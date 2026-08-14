//! Anonymous GitHub API and raw-content renderer.

use omp_core::{Str, base64};
use serde::Deserialize;
use smallvec::SmallVec;
use url::Url;

use super::{
	super::types::{HttpClient, HttpRequest, HttpResponse, RenderResult, WebError, finalize_output},
	utils::build_result,
};

const API: &str = "https://api.github.com";

#[derive(Clone, Debug, Eq, PartialEq)]
enum Kind {
	Blob,
	Tree,
	Repo,
	Commit,
	Issue(u64),
	Issues,
	Pull(u64),
	Other,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Target {
	owner:     String,
	repo:      String,
	kind:      Kind,
	reference: Option<String>,
	path:      Option<String>,
}

/// Returns whether the URL belongs to a GitHub repository path.
pub(super) fn matches(url: &Url) -> bool {
	parse(url).is_some()
}

/// Renders supported GitHub repository URLs through anonymous API and raw
/// requests.
pub(super) async fn render<C: HttpClient + Sync>(
	client: &C,
	url: &Url,
) -> Result<Option<RenderResult>, WebError> {
	let Some(target) = parse(url) else {
		return Ok(None);
	};

	match &target.kind {
		Kind::Blob => render_blob(client, &target).await,
		Kind::Tree => render_tree(client, &target).await,
		Kind::Repo => render_repo(client, &target).await,
		Kind::Commit => render_commit(client, &target).await,
		Kind::Issue(_) | Kind::Pull(_) => render_issue(client, &target).await,
		Kind::Issues => render_issues(client, &target).await,
		Kind::Other => Ok(None),
	}
}

fn parse(url: &Url) -> Option<Target> {
	if url.host_str()? != "github.com" {
		return None;
	}
	let parts: Vec<_> = url
		.path_segments()?
		.filter(|part| !part.is_empty())
		.collect();
	if parts.len() < 2 {
		return None;
	}

	let owner = parts[0].to_owned();
	let repo = parts[1].to_owned();
	let rest = &parts[2..];
	if rest.is_empty() {
		return Some(Target { owner, repo, kind: Kind::Repo, reference: None, path: None });
	}

	let section = rest[0];
	let sub = &rest[1..];
	let (kind, reference, path) = match section {
		"blob" => (
			Kind::Blob,
			sub.first().map(|value| (*value).to_owned()),
			Some(sub.get(1..).unwrap_or_default().join("/")),
		),
		"tree" => (
			Kind::Tree,
			sub.first().map(|value| (*value).to_owned()),
			Some(sub.get(1..).unwrap_or_default().join("/")),
		),
		"commit" if !sub.is_empty() => (Kind::Commit, Some(sub[0].to_owned()), None),
		"issues" => (
			sub.first()
				.and_then(|number| number.parse().ok())
				.map_or(Kind::Issues, Kind::Issue),
			None,
			None,
		),
		"pull" => (
			sub.first()
				.and_then(|number| number.parse().ok())
				.map_or(Kind::Other, Kind::Pull),
			None,
			None,
		),
		"pulls" => (Kind::Other, None, None),
		_ => (Kind::Other, None, None),
	};
	Some(Target { owner, repo, kind, reference, path })
}

fn filename_order(left: &str, right: &str) -> std::cmp::Ordering {
	let primary = left
		.chars()
		.flat_map(char::to_lowercase)
		.cmp(right.chars().flat_map(char::to_lowercase));
	if primary != std::cmp::Ordering::Equal {
		return primary;
	}
	left
		.chars()
		.map(|character| character.is_uppercase())
		.cmp(right.chars().map(|character| character.is_uppercase()))
		.then_with(|| left.cmp(right))
}

async fn render_blob<C: HttpClient + Sync>(
	client: &C,
	target: &Target,
) -> Result<Option<RenderResult>, WebError> {
	let Some(reference) = target.reference.as_deref() else {
		return Ok(None);
	};
	let Some(path) = target.path.as_deref() else {
		return Ok(None);
	};
	let raw_url = format!(
		"https://raw.githubusercontent.com/{}/{}/{reference}/{path}",
		target.owner, target.repo
	);
	let Some(response) = get(client, &raw_url, None).await? else {
		return Ok(None);
	};
	let (content, truncated) = finalize_output(response.text().as_ref());
	let mut notes = SmallVec::<Str, 4>::new();
	notes.push(Str::from(format!("Fetched raw: {raw_url}")));
	if truncated {
		notes.push(Str::new_static("Output truncated to 500000 characters"));
	}
	Ok(Some(RenderResult {
		content,
		content_type: Some(Str::new_static("text/plain")),
		method: Str::new_static("github-raw"),
		notes,
	}))
}

async fn render_tree<C: HttpClient + Sync>(
	client: &C,
	target: &Target,
) -> Result<Option<RenderResult>, WebError> {
	let Some(repo): Option<RepositoryIdentity> =
		api_json(client, &repo_endpoint(target, "")).await?
	else {
		return Ok(None);
	};
	let reference = target.reference.as_deref().unwrap_or(&repo.default_branch);
	let directory = target.path.as_deref().unwrap_or("");
	let mut markdown = format!(
		"# {}/{}\n\n**Branch:** {reference}\n\n",
		repo.full_name,
		if directory.is_empty() {
			"(root)"
		} else {
			directory
		}
	);

	let endpoint =
		format!("{API}/repos/{}/{}/contents/{directory}?ref={reference}", target.owner, target.repo);
	if let Some(mut items) = api_json::<_, Vec<ContentItem>>(client, &endpoint).await? {
		items.sort_by(|left, right| match (left.kind.as_str(), right.kind.as_str()) {
			("dir", "dir") | ("file", "file") => filename_order(&left.name, &right.name),
			("dir", _) => std::cmp::Ordering::Less,
			(_, "dir") => std::cmp::Ordering::Greater,
			_ => filename_order(&left.name, &right.name),
		});
		markdown.push_str("## Contents\n\n```\n");
		for item in &items {
			let prefix = if item.kind == "dir" {
				"[dir] "
			} else {
				"      "
			};
			let size = if item.kind == "file" && item.size.unwrap_or(0) != 0 {
				format!(" ({} bytes)", item.size.unwrap_or(0))
			} else {
				String::new()
			};
			markdown.push_str(&format!("{prefix}{}{size}\n", item.name));
		}
		markdown.push_str("```\n\n");

		if let Some(readme) = items
			.iter()
			.find(|item| item.kind == "file" && item.name.eq_ignore_ascii_case("readme.md"))
		{
			let readme_path = if directory.is_empty() {
				readme.name.clone()
			} else {
				format!("{directory}/{}", readme.name)
			};
			let raw_url = format!(
				"https://raw.githubusercontent.com/{}/{}/{reference}/{readme_path}",
				target.owner, target.repo
			);
			if let Some(response) = get(client, &raw_url, None).await? {
				markdown.push_str("---\n\n## README\n\n");
				markdown.push_str(response.text().as_ref());
			}
		}
	}

	Ok(Some(markdown_result(markdown, "github-tree", "Fetched via GitHub API")))
}

async fn render_repo<C: HttpClient + Sync>(
	client: &C,
	target: &Target,
) -> Result<Option<RenderResult>, WebError> {
	let Some(repo): Option<Repository> = api_json(client, &repo_endpoint(target, "")).await? else {
		return Ok(None);
	};
	let mut markdown = format!("# {}\n\n", repo.full_name);
	if let Some(description) = repo
		.description
		.as_deref()
		.filter(|value| !value.is_empty())
	{
		markdown.push_str(description);
		markdown.push_str("\n\n");
	}
	markdown.push_str(&format!(
		"Stars: {} · Forks: {} · Issues: {}\n",
		repo.stargazers_count, repo.forks_count, repo.open_issues_count
	));
	if let Some(language) = repo.language.as_deref().filter(|value| !value.is_empty()) {
		markdown.push_str(&format!("Language: {language}\n"));
	}
	if let Some(license) = repo.license.as_ref() {
		markdown.push_str(&format!("License: {}\n", license.name));
	}
	markdown.push_str("\n---\n\n");

	let tree_endpoint =
		repo_endpoint(target, &format!("/git/trees/{}?recursive=1", repo.default_branch));
	if let Some(tree) = api_json::<_, GitTree>(client, &tree_endpoint).await? {
		markdown.push_str("## Files\n\n```\n");
		for item in tree.tree.iter().take(100) {
			let prefix = if item.kind == "tree" {
				"[dir] "
			} else {
				"      "
			};
			markdown.push_str(&format!("{prefix}{}\n", item.path));
		}
		if tree.tree.len() > 100 {
			markdown.push_str(&format!("[…{} files elided…]\n", tree.tree.len() - 100));
		}
		markdown.push_str("```\n\n");
	}

	if let Some(readme) = api_json::<_, Readme>(client, &repo_endpoint(target, "/readme")).await? {
		if readme.encoding == "base64" {
			if let Some(decoded) = decode_base64(&readme.content) {
				markdown.push_str("## README\n\n");
				markdown.push_str(&String::from_utf8_lossy(&decoded));
			}
		}
	}

	Ok(Some(markdown_result(markdown, "github-repo", "Fetched via GitHub API")))
}

async fn render_issue<C: HttpClient + Sync>(
	client: &C,
	target: &Target,
) -> Result<Option<RenderResult>, WebError> {
	let (number, suffix, method) = match &target.kind {
		Kind::Pull(number) => (*number, format!("/pulls/{number}"), "github-pr"),
		Kind::Issue(number) => (*number, format!("/issues/{number}"), "github-issue"),
		_ => return Ok(None),
	};
	let Some(issue): Option<Issue> = api_json(client, &repo_endpoint(target, &suffix)).await? else {
		return Ok(None);
	};

	let mut markdown = format!(
		"# {}\n\n**#{}** · {} · opened by @{}\nCreated: {} · Updated: {}\n",
		issue.title, issue.number, issue.state, issue.user.login, issue.created_at, issue.updated_at
	);
	if !issue.labels.is_empty() {
		markdown.push_str(&format!(
			"Labels: {}\n",
			issue
				.labels
				.iter()
				.map(|label| label.name.as_str())
				.collect::<Vec<_>>()
				.join(", ")
		));
	}
	markdown.push_str("\n---\n\n");
	markdown.push_str(
		issue
			.body
			.as_deref()
			.filter(|body| !body.is_empty())
			.unwrap_or("*No description provided.*"),
	);
	markdown.push_str("\n\n---\n\n");

	if issue.comments > 0 {
		let comments = fetch_comments(client, target, number, issue.comments).await?;
		if !comments.is_empty() {
			let count = if issue.comments > comments.len() as u64 {
				format!("{} of {}", comments.len(), issue.comments)
			} else {
				comments.len().to_string()
			};
			markdown.push_str(&format!("## Comments ({count})\n\n"));
			for comment in comments {
				markdown.push_str(&format!(
					"### @{} · {}\n\n{}\n\n---\n\n",
					comment.user.login, comment.created_at, comment.body
				));
			}
		}
	}

	Ok(Some(markdown_result(markdown, method, "Fetched via GitHub API")))
}

async fn fetch_comments<C: HttpClient + Sync>(
	client: &C,
	target: &Target,
	number: u64,
	expected: u64,
) -> Result<Vec<IssueComment>, WebError> {
	let mut comments = Vec::new();
	let mut page = 1_u64;
	while comments.len() < expected as usize {
		let endpoint =
			repo_endpoint(target, &format!("/issues/{number}/comments?per_page=100&page={page}"));
		let Some(mut batch): Option<Vec<IssueComment>> = api_json(client, &endpoint).await? else {
			break;
		};
		let count = batch.len();
		if count == 0 {
			break;
		}
		comments.append(&mut batch);
		if count < 100 {
			break;
		}
		page += 1;
	}
	Ok(comments)
}

async fn render_issues<C: HttpClient + Sync>(
	client: &C,
	target: &Target,
) -> Result<Option<RenderResult>, WebError> {
	let endpoint = repo_endpoint(target, "/issues?state=open&per_page=30");
	let Some(issues): Option<Vec<IssueListItem>> = api_json(client, &endpoint).await? else {
		return Ok(None);
	};
	let mut markdown = format!("# {}/{} - Open Issues\n\n", target.owner, target.repo);
	for issue in issues
		.into_iter()
		.filter(|issue| issue.pull_request.is_none())
	{
		let labels = if issue.labels.is_empty() {
			String::new()
		} else {
			format!(
				" [{}]",
				issue
					.labels
					.iter()
					.map(|label| label.name.as_str())
					.collect::<Vec<_>>()
					.join(", ")
			)
		};
		markdown.push_str(&format!(
			"- **#{}** {}{labels}\n  by @{} · {} comments · {}\n\n",
			issue.number, issue.title, issue.user.login, issue.comments, issue.created_at
		));
	}
	Ok(Some(markdown_result(markdown, "github-issues", "Fetched via GitHub API")))
}

async fn render_commit<C: HttpClient + Sync>(
	client: &C,
	target: &Target,
) -> Result<Option<RenderResult>, WebError> {
	let Some(reference) = target.reference.as_deref() else {
		return Ok(None);
	};
	let Some(commit): Option<Commit> =
		api_json(client, &repo_endpoint(target, &format!("/commits/{reference}"))).await?
	else {
		return Ok(None);
	};
	let mut lines = commit.inner.message.split('\n');
	let subject = lines
		.next()
		.filter(|value| !value.is_empty())
		.unwrap_or_else(|| commit.sha.get(..7).unwrap_or(commit.sha.as_str()));
	let author = commit
		.author
		.as_ref()
		.and_then(|author| (!author.login.is_empty()).then(|| format!("@{}", author.login)))
		.or_else(|| {
			commit
				.inner
				.author
				.as_ref()
				.and_then(|author| author.name.clone())
		})
		.unwrap_or_else(|| "unknown".to_owned());
	let short_sha = commit.sha.get(..12).unwrap_or(commit.sha.as_str());
	let mut markdown = format!("# {subject}\n\n**{short_sha}** · authored by {author}");
	if let Some(date) = commit
		.inner
		.author
		.as_ref()
		.and_then(|author| author.date.as_deref())
		.filter(|date| !date.is_empty())
	{
		markdown.push_str(&format!(" · {date}"));
	}
	markdown.push('\n');
	if let Some(stats) = commit.stats.as_ref() {
		let files = commit.files.len();
		markdown.push_str(&format!(
			"{files} file{} changed · +{} −{}\n",
			if files == 1 { "" } else { "s" },
			stats.additions.unwrap_or(0),
			stats.deletions.unwrap_or(0)
		));
	}
	if !commit.parents.is_empty() {
		markdown.push_str("Parents: ");
		markdown.push_str(
			&commit
				.parents
				.iter()
				.map(|parent| parent.sha.get(..12).unwrap_or(parent.sha.as_str()))
				.collect::<Vec<_>>()
				.join(", "),
		);
		markdown.push('\n');
	}
	let body = lines.collect::<Vec<_>>().join("\n");
	if !body.trim().is_empty() {
		markdown.push('\n');
		markdown.push_str(body.trim());
		markdown.push('\n');
	}
	if !commit.files.is_empty() {
		markdown.push_str(&format!("\n---\n\n## Files ({})\n\n", commit.files.len()));
		for file in commit.files {
			let name = file
				.previous_filename
				.as_deref()
				.filter(|name| !name.is_empty())
				.map_or_else(
					|| file.filename.clone(),
					|previous| format!("{previous} → {}", file.filename),
				);
			markdown.push_str(&format!(
				"### {name}\n\n{} · +{} −{}\n\n",
				file.status, file.additions, file.deletions
			));
			if let Some(patch) = file.patch.as_deref().filter(|patch| !patch.is_empty()) {
				markdown.push_str(&format!("```diff\n{patch}\n```\n\n"));
			} else {
				markdown.push_str("*No textual diff (binary or too large).*\n\n");
			}
		}
	}
	Ok(Some(markdown_result(markdown, "github-commit", "Fetched via GitHub API")))
}

pub(super) async fn api_json<C, T>(client: &C, endpoint: &str) -> Result<Option<T>, WebError>
where
	C: HttpClient + Sync,
	T: for<'de> Deserialize<'de>,
{
	let Some(response) = get(client, endpoint, Some("application/vnd.github.v3+json")).await? else {
		return Ok(None);
	};
	Ok(serde_json::from_slice(&response.body).ok())
}

async fn get<C: HttpClient + Sync>(
	client: &C,
	endpoint: &str,
	accept: Option<&'static str>,
) -> Result<Option<HttpResponse>, WebError> {
	let request = if let Some(accept) = accept {
		HttpRequest::new(endpoint)
			.with_header("Accept", accept)
			.with_header("User-Agent", concat!("omp/", env!("CARGO_PKG_VERSION")))
	} else {
		HttpRequest::new(endpoint)
	};
	let Ok(response) = client.get(request).await else {
		return Ok(None);
	};
	Ok(response.is_success().then_some(response))
}

fn repo_endpoint(target: &Target, suffix: &str) -> String {
	format!("{API}/repos/{}/{}{suffix}", target.owner, target.repo)
}

fn markdown_result(content: String, method: &'static str, note: &'static str) -> RenderResult {
	let mut result = build_result(&content, method);
	result.notes.insert(0, Str::new_static(note));
	result
}

fn decode_base64(input: &str) -> Option<Vec<u8>> {
	let compact = input
		.bytes()
		.filter(|byte| !byte.is_ascii_whitespace())
		.collect::<Vec<_>>();
	base64::decode(&compact).into_vec().ok()
}

#[derive(Deserialize)]
struct RepositoryIdentity {
	full_name:      String,
	default_branch: String,
}

#[derive(Deserialize)]
struct Repository {
	full_name:         String,
	description:       Option<String>,
	stargazers_count:  u64,
	forks_count:       u64,
	open_issues_count: u64,
	default_branch:    String,
	language:          Option<String>,
	license:           Option<License>,
}

#[derive(Deserialize)]
struct License {
	name: String,
}

#[derive(Deserialize)]
struct ContentItem {
	name: String,
	#[serde(rename = "type")]
	kind: String,
	size: Option<u64>,
}

#[derive(Deserialize)]
struct GitTree {
	#[serde(default)]
	tree: Vec<GitTreeItem>,
}

#[derive(Deserialize)]
struct GitTreeItem {
	path: String,
	#[serde(rename = "type")]
	kind: String,
}

#[derive(Deserialize)]
struct Readme {
	content:  String,
	encoding: String,
}

#[derive(Deserialize)]
struct User {
	login: String,
}

#[derive(Deserialize)]
struct Label {
	name: String,
}

#[derive(Deserialize)]
struct Issue {
	title:      String,
	number:     u64,
	state:      String,
	user:       User,
	created_at: String,
	updated_at: String,
	body:       Option<String>,
	#[serde(default)]
	labels:     Vec<Label>,
	#[serde(default)]
	comments:   u64,
}

#[derive(Deserialize)]
struct IssueComment {
	user:       User,
	created_at: String,
	body:       String,
}

#[derive(Deserialize)]
struct IssueListItem {
	number:       u64,
	title:        String,
	user:         User,
	created_at:   String,
	comments:     u64,
	#[serde(default)]
	labels:       Vec<Label>,
	pull_request: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct Commit {
	sha:     String,
	#[serde(rename = "commit")]
	inner:   CommitInner,
	author:  Option<User>,
	#[serde(default)]
	parents: Vec<CommitParent>,
	stats:   Option<CommitStats>,
	#[serde(default)]
	files:   Vec<CommitFile>,
}

#[derive(Deserialize)]
struct CommitInner {
	message: String,
	author:  Option<CommitAuthor>,
}

#[derive(Deserialize)]
struct CommitAuthor {
	name: Option<String>,
	date: Option<String>,
}

#[derive(Deserialize)]
struct CommitParent {
	sha: String,
}

#[derive(Deserialize)]
struct CommitStats {
	additions: Option<u64>,
	deletions: Option<u64>,
}

#[derive(Deserialize)]
struct CommitFile {
	filename:          String,
	status:            String,
	additions:         u64,
	deletions:         u64,
	patch:             Option<String>,
	previous_filename: Option<String>,
}
