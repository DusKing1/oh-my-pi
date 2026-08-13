//! Production built-in tool registry assembly.

use std::sync::Arc;

use omp_core::Str;
use omp_proto::toolhost::v1::{
	GrammarSyntax as WorkerGrammarSyntax, ToolDecl, tool_constraint,
};
use omp_tool::{Constraint, GrammarSyntax, Registry, Rev, ToolSpec};
use omp_tools::edit::FormatPolicy;

use super::{
	EnvdError,
	blobs::BlobHost,
	docs::DocumentHost,
	exec::ExecHost,
	tool_search::WorkspaceSearchAdapter,
	tool_shell::ShellExecHost,
	worker::ToolWorkerSupervisor,
	workspace::WorkspaceHost,
};

/// Builds the complete registry shared by environment dispatch and the agent.
///
/// Resource adapters are cloned into their typed executors. Worker declarations
/// occupy explicit worker routes: they are advertised by the registry but can
/// only be invoked by the environment's worker supervisor.
pub(crate) fn production_registry(
	documents: &DocumentHost,
	blobs: &BlobHost,
	exec: &ExecHost,
	workspace: &WorkspaceHost,
	root_uri: &Str,
	workers: &ToolWorkerSupervisor,
	mut registry: Registry,
) -> Result<Arc<Registry>, EnvdError> {
	for name in ["read", "edit", "shell", "grep", "glob"] {
		ensure_name_absent(&registry, name)?;
	}
	registry.register(omp_tools::read::tool(documents.clone(), blobs.clone()))?;
	registry.register(omp_tools::edit::tool(documents.clone(), FormatPolicy::Configured))?;
	registry.register(omp_tools::shell::shell(ShellExecHost::new(
		exec.clone(),
		root_uri.clone(),
	)))?;
	let search = WorkspaceSearchAdapter::new(workspace.clone());
	registry.register(omp_tools::grep::tool(search.clone()))?;
	registry.register(omp_tools::glob::tool(search))?;
	for declaration in workers.registrations() {
		let spec = worker_spec(declaration)?;
		ensure_name_absent(&registry, &spec.name)?;
		registry.register_worker(spec)?;
	}
	Ok(Arc::new(registry))
}

fn ensure_name_absent(registry: &Registry, name: &str) -> Result<(), EnvdError> {
	if registry.live_identity(name).is_some() {
		return Err(EnvdError::DuplicateToolName(Str::from(name)));
	}
	Ok(())
}

fn worker_spec(declaration: &ToolDecl) -> Result<ToolSpec, EnvdError> {
	let definition = declaration.definition.as_ref().ok_or_else(|| {
		EnvdError::WorkerDeclaration(Str::new_static("worker tool declaration has no definition"))
	})?;
	Ok(ToolSpec {
		name: Str::from(definition.name.as_str()),
		rev: parse_revision(&declaration.rev)?,
		description: Str::from(definition.description.as_str()),
		schema: definition.schema_json.clone(),
		constraint: worker_constraint(declaration)?,
	})
}

fn parse_revision(value: &str) -> Result<Rev, EnvdError> {
	let (family, number) = match value.rsplit_once('.') {
		Some((family, number)) if !family.is_empty() => (family, number),
		Some(_) => return Err(worker_declaration_error("worker revision family is empty")),
		None => ("", value),
	};
	let n = number.parse::<u16>().map_err(|_| {
		worker_declaration_error("worker revision must be `N` or `family.N` with a u16 N")
	})?;
	Ok(Rev { family: Str::from(family), n })
}

fn worker_constraint(declaration: &ToolDecl) -> Result<Constraint, EnvdError> {
	let Some(kind) = declaration.constraint.as_ref().and_then(|value| value.kind.as_ref()) else {
		let strict = declaration
			.definition
			.as_ref()
			.and_then(|definition| definition.strict)
			.unwrap_or(false);
		return Ok(if strict {
			Constraint::Schema { priority: 100 }
		} else {
			Constraint::None
		});
	};
	match kind {
		tool_constraint::Kind::Schema(schema) => Ok(Constraint::Schema {
			priority: constraint_priority(schema.priority)?,
		}),
		tool_constraint::Kind::Grammar(grammar) => {
			let syntax = match WorkerGrammarSyntax::try_from(grammar.syntax) {
				Ok(WorkerGrammarSyntax::Lark) => GrammarSyntax::Lark,
				Ok(WorkerGrammarSyntax::Regex) => GrammarSyntax::Regex,
				_ => {
					return Err(worker_declaration_error(
						"worker grammar constraint has an unsupported syntax",
					));
				},
			};
			Ok(Constraint::Grammar {
				syntax,
				definition: Str::from(grammar.definition.as_str()),
				priority: constraint_priority(grammar.priority)?,
			})
		},
	}
}

fn constraint_priority(priority: u32) -> Result<u8, EnvdError> {
	u8::try_from(priority)
		.map_err(|_| worker_declaration_error("worker constraint priority exceeds u8"))
}

fn worker_declaration_error(message: &'static str) -> EnvdError {
	EnvdError::WorkerDeclaration(Str::new_static(message))
}
