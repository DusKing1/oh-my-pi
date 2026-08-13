//! One-time type erasure, live advertisement, and historical lift composition.

use std::{collections::BTreeMap, pin::Pin, sync::Arc};

use async_stream::stream;
use bytes::Bytes;
use futures::{Stream, StreamExt, pin_mut};
use omp_core::Str;
use omp_llm_catalog::GrammarBits;
use omp_llm_inference::{
	Adjustment, FeatureId, OpaqueJson, ReasonId, ToolDefinition, ToolGrammar, ToolGrammarSyntax,
	ToolInputConstraint,
};
use omp_proto::inference::v1::InvokeInput;
use serde_json::Value;
use thiserror::Error;

use crate::{
	Abort, Constraint, GrammarSyntax, IncomingParams, LiftedCall, Part, PromptCaps, RecordedCall,
	RecordedCallOwned, Rev, Tool, ToolIdentity, Verdict,
};

/// Catalog capabilities needed for deterministic tool lowering.
#[derive(Clone, Copy, Debug)]
pub struct LoweringCaps {
	/// Whether per-tool strict JSON Schema is supported.
	pub strict_schema: bool,
	/// Supported freeform grammar languages.
	pub grammar:       GrammarBits,
}

/// Strength retained after capability-aware constraint lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstraintDisposition {
	/// Route can honor the requested constraint.
	Required,
	/// Request remains a preference and is receipted when unavailable.
	Prefer,
}
/// Execution route associated with a live registry entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolRoute {
	/// In-process typed Rust executor erased at registration.
	Native,
	/// Externally supervised worker executor.
	Worker,
}


/// One live tool declaration ready for inference request construction.
#[derive(Clone, Debug)]
pub struct LoweredTool {
	/// Durable live identity.
	pub identity:    ToolIdentity,
	/// Canonical inference declaration.
	pub definition:  ToolDefinition,
	/// Constraint strength after catalog-aware lowering, if requested.
	pub disposition: Option<ConstraintDisposition>,
	/// Original constraint priority, if requested.
	pub priority:    Option<u8>,
	/// Explicit degradation receipts; unsupported constraints are never silent.
	pub adjustments: Vec<Adjustment>,
}

/// Type-erased event emitted across the environment dispatch boundary.
#[derive(Clone, Debug)]
pub enum ErasedEv {
	/// Serialized typed update.
	Update(Bytes),
	/// Terminal serialized outcome.
	Done(ErasedOutcome),
}

/// Type-erased terminal tool outcome.
#[derive(Clone, Debug)]
pub enum ErasedOutcome {
	/// Structured journal verdict with compaction metadata.
	Done {
		/// Exact serialized [`Verdict`] JSON.
		verdict: Bytes,
		/// Whether projected parts may be compacted.
		useless: bool,
	},
	/// Detached work.
	Detached(crate::JobRef),
}

/// Cold dispatch stream allocated once for an erased invocation.
pub type ErasedStream<'a> =
	Pin<Box<dyn Stream<Item = Result<ErasedEv, RegistryError>> + Send + 'a>>;

/// Projection result for a durable historical call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProjectedCall {
	/// Call is expressed in the live revision and may be emitted as a tool item.
	Live(RecordedCallOwned),
	/// No complete lift path exists; preserve the original call as transcript
	/// data.
	Data(RecordedCallOwned),
}

/// Authoritative model projection and branch metadata decoded from one verdict.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectedVerdict {
	/// Model-facing parts under the supplied current-model capabilities.
	pub parts:    Vec<Part>,
	/// Whether the decoded verdict branch is a fault, argument error, or abort.
	pub is_error: bool,
	/// Durable compaction hint, forced false for argument errors and aborts.
	pub useless:  bool,
}

/// Registry construction, dispatch, serialization, or projection failure.
#[derive(Debug, Error)]
pub enum RegistryError {
	/// `(name, revision)` was registered twice.
	#[error("tool revision already registered: {0}@{1}")]
	Duplicate(Str, Rev),
	/// Tool name is not registered.
	#[error("unknown tool: {0}")]
	UnknownTool(Str),
	/// Operation requires a native pure or execution surface unavailable for a
	/// worker declaration.
	#[error("tool {name}@{rev} is worker-routed and cannot perform registry operation {operation}")]
	UnsupportedExternal {
		/// Tool name.
		name:      Str,
		/// Exact registered revision.
		rev:       Rev,
		/// Requested registry operation.
		operation: &'static str,
	},
	/// Registered schema is not one complete JSON value.
	#[error("invalid JSON Schema for {name}@{rev}: {source}")]
	InvalidSchema {
		/// Tool name.
		name:   Str,
		/// Tool revision.
		rev:    Rev,
		/// Parser failure.
		source: serde_json::Error,
	},
	/// Typed event or verdict serialization failed.
	#[error("tool value serialization failed: {0}")]
	Serialize(#[from] serde_json::Error),
	/// Stored verdict does not match its registered typed revision.
	#[error("stored verdict does not match registered tool revision: {0}")]
	VerdictShape(Str),
	/// Serialized update does not match its registered typed revision.
	#[error("tool update does not match registered revision {name}@{rev}: {source}")]
	UpdateShape {
		/// Tool name.
		name: Str,
		/// Exact registered revision.
		rev: Rev,
		/// Typed update decoder failure.
		source: serde_json::Error,
	},
}

trait ErasedTool: Send + Sync {
	fn spec(&self) -> &crate::ToolSpec;
	fn route(&self) -> ToolRoute;
	fn schema(&self) -> &OpaqueJson;
	fn call<'a>(&'a self, params: IncomingParams<'a>) -> ErasedStream<'a>;
	fn project_verdict(
		&self,
		verdict: &[u8],
		recorded_useless: bool,
		caps: &PromptCaps,
	) -> Result<ProjectedVerdict, RegistryError>;
	fn invoke_input(
		&self,
		invocation_id: &str,
		json: &[u8],
	) -> Result<Option<InvokeInput>, RegistryError>;
	fn lift(&self, from: &Rev, call: RecordedCall<'_>) -> Option<LiftedCall>;

}

struct Worker {
	spec:   crate::ToolSpec,
	schema: OpaqueJson,
}

impl ErasedTool for Worker {
	fn spec(&self) -> &crate::ToolSpec {
		&self.spec
	}

	fn route(&self) -> ToolRoute {
		ToolRoute::Worker
	}

	fn schema(&self) -> &OpaqueJson {
		&self.schema
	}


	fn call<'a>(&'a self, _params: IncomingParams<'a>) -> ErasedStream<'a> {
		let error = external_error(&self.spec, "invoke");
		Box::pin(futures::stream::once(async move { Err(error) }))
	}

	fn project_verdict(
		&self,
		_verdict: &[u8],
		_recorded_useless: bool,
		_caps: &PromptCaps,
	) -> Result<ProjectedVerdict, RegistryError> {
		Err(external_error(&self.spec, "project_verdict"))
	}

	fn invoke_input(
		&self,
		_invocation_id: &str,
		_json: &[u8],
	) -> Result<Option<InvokeInput>, RegistryError> {
		Err(external_error(&self.spec, "invoke_input"))
	}

	fn lift(&self, _from: &Rev, _call: RecordedCall<'_>) -> Option<LiftedCall> {
		None
	}
}

struct Registered<T> {
	tool:   T,
	schema: OpaqueJson,
}

impl<T: Tool> ErasedTool for Registered<T> {
	fn spec(&self) -> &crate::ToolSpec {
		self.tool.spec()
	}

	fn route(&self) -> ToolRoute {
		ToolRoute::Native
	}

	fn schema(&self) -> &OpaqueJson {
		&self.schema
	}

	fn call<'a>(&'a self, params: IncomingParams<'a>) -> ErasedStream<'a> {
		Box::pin(stream! {
			let events = self.tool.call(params);
			pin_mut!(events);
			let mut terminal = false;
			while let Some(event) = events.next().await {
				match event {
					crate::Ev::Update(update) => match serde_json::to_vec(&update) {
						Ok(json) => yield Ok(ErasedEv::Update(Bytes::from(json))),
						Err(error) => {
							terminal = true;
							yield Err(RegistryError::Serialize(error));
							break;
						},
					},
					crate::Ev::Args(issue) => {
						terminal = true;
						let verdict = Verdict::<T::Payload, T::Fault>::Args(issue);
						match serde_json::to_vec(&verdict) {
							Ok(json) => yield Ok(ErasedEv::Done(ErasedOutcome::Done {
								verdict: Bytes::from(json),
								useless: false,
							})),
							Err(error) => yield Err(RegistryError::Serialize(error)),
						}
						break;
					},
					crate::Ev::Aborted(abort) => {
						terminal = true;
						let verdict = Verdict::<T::Payload, T::Fault>::Aborted(abort);
						match serde_json::to_vec(&verdict) {
							Ok(json) => yield Ok(ErasedEv::Done(ErasedOutcome::Done {
								verdict: Bytes::from(json),
								useless: false,
							})),
							Err(error) => yield Err(RegistryError::Serialize(error)),
						}
						break;
					},
					crate::Ev::Done(outcome) => {
						terminal = true;
						let erased = match outcome {
							crate::Outcome::Done { result, useless } => {
								let verdict = match result {
									Ok(payload) => Verdict::<T::Payload, T::Fault>::Ok(payload),
									Err(fault) => Verdict::<T::Payload, T::Fault>::Fault(fault),
								};
								match serde_json::to_vec(&verdict) {
									Ok(json) => ErasedOutcome::Done {
										verdict: Bytes::from(json),
										useless,
									},
									Err(error) => {
										yield Err(RegistryError::Serialize(error));
										break;
									},
								}
							},
							crate::Outcome::Detached(job) => ErasedOutcome::Detached(job),
						};
						yield Ok(ErasedEv::Done(erased));
						break;
					},
				}
			}
			if !terminal {
				let verdict = Verdict::<Value, Value>::Aborted(Abort::MissingOutcome);
				match serde_json::to_vec(&verdict) {
					Ok(json) => yield Ok(ErasedEv::Done(ErasedOutcome::Done {
						verdict: Bytes::from(json),
						useless: false,
					})),
					Err(error) => yield Err(RegistryError::Serialize(error)),
				}
			}
		})
	}

	fn project_verdict(
		&self,
		verdict: &[u8],
		recorded_useless: bool,
		caps: &PromptCaps,
	) -> Result<ProjectedVerdict, RegistryError> {
		let verdict: Verdict<T::Payload, T::Fault> = serde_json::from_slice(verdict)
			.map_err(|_| RegistryError::VerdictShape(self.tool.spec().name.clone()))?;
		Ok(match &verdict {
			Verdict::Ok(payload) => ProjectedVerdict {
				parts:    self.tool.prompt(Ok(payload), caps),
				is_error: false,
				useless:  recorded_useless,
			},
			Verdict::Fault(fault) => ProjectedVerdict {
				parts:    self.tool.prompt(Err(fault), caps),
				is_error: true,
				useless:  recorded_useless,
			},
			Verdict::Args(issue) => ProjectedVerdict {
				parts:    vec![Part::Text { text: render_arg_issue(issue) }],
				is_error: true,
				useless:  false,
			},
			Verdict::Aborted(abort) => ProjectedVerdict {
				parts:    vec![Part::Text { text: render_abort(abort) }],
				is_error: true,
				useless:  false,
			},
		})
	}

	fn invoke_input(
		&self,
		invocation_id: &str,
		json: &[u8],
	) -> Result<Option<InvokeInput>, RegistryError> {
		let update: T::Update = serde_json::from_slice(json).map_err(|source| {
			RegistryError::UpdateShape {
				name: self.tool.spec().name.clone(),
				rev: self.tool.spec().rev.clone(),
				source,
			}
		})?;
		Ok(self.tool.invoke_input(&update, invocation_id))
	}

	fn lift(&self, from: &Rev, call: RecordedCall<'_>) -> Option<LiftedCall> {
		self.tool.lift(from, call)
	}
}

/// Revision-aware tool registry.
///
/// Concrete associated types are erased exactly once by
/// [`register`](Self::register). Every revision remains available only for pure
/// projection/lift code; dispatch and advertisement always select the one live
/// revision per stable name.
#[derive(Default)]
pub struct Registry {
	versions: BTreeMap<Str, BTreeMap<Rev, Arc<dyn ErasedTool>>>,
	live:     BTreeMap<Str, Rev>,
}

impl Registry {
	/// Creates an empty registry.
	pub fn new() -> Self {
		Self::default()
	}

	/// Registers a typed tool and makes this revision live for its name.
	///
	/// Older registered revisions remain only as pure lift steps; they are never
	/// dispatched or advertised.
	pub fn register<T: Tool>(&mut self, tool: T) -> Result<(), RegistryError> {
		let spec = tool.spec();
		let name = spec.name.clone();
		let rev = spec.rev.clone();
		let value = serde_json::from_slice(&spec.schema).map_err(|source| {
			RegistryError::InvalidSchema { name: name.clone(), rev: rev.clone(), source }
		})?;
		let versions = self.versions.entry(name.clone()).or_default();
		if versions.contains_key(&rev) {
			return Err(RegistryError::Duplicate(name, rev));
		}
		versions.insert(rev.clone(), Arc::new(Registered { tool, schema: OpaqueJson::new(value) }));
		self.live.insert(name, rev);
		Ok(())
	}
	/// Registers an externally supervised worker declaration and makes it live.
	///
	/// Worker declarations participate in identity, hashing, and advertisement,
	/// but execution and pure typed projection remain owned by the worker route.
	pub fn register_worker(&mut self, spec: crate::ToolSpec) -> Result<(), RegistryError> {
		let name = spec.name.clone();
		let rev = spec.rev.clone();
		let value = serde_json::from_slice(&spec.schema).map_err(|source| {
			RegistryError::InvalidSchema { name: name.clone(), rev: rev.clone(), source }
		})?;
		let versions = self.versions.entry(name.clone()).or_default();
		if versions.contains_key(&rev) {
			return Err(RegistryError::Duplicate(name, rev));
		}
		versions.insert(rev.clone(), Arc::new(Worker { spec, schema: OpaqueJson::new(value) }));
		self.live.insert(name, rev);
		Ok(())
	}


	/// Borrows the exact live `(name, revision)` identity for `name`.
	///
	/// The returned values are owned by this registry and remain valid for the
	/// duration of the borrow. No transcript-facing identity is synthesized.
	#[must_use]
	pub fn live_identity(&self, name: &str) -> Option<(&Str, &Rev)> {
		self.live.get_key_value(name)
	}
	/// Returns the execution route of the live declaration named `name`.
	pub fn route(&self, name: &str) -> Result<ToolRoute, RegistryError> {
		Ok(self.live_entry(name)?.route())
	}


	/// Hashes the ordered live tool identities without allocation or serialization.
	///
	/// Every identity field is length-delimited with a little-endian `u64`
	/// length; revision numbers are encoded as little-endian `u16` bytes. The
	/// live map's `BTreeMap` order makes the digest registration-order independent.
	#[must_use]
	pub fn live_hash(&self) -> [u8; 32] {
		let mut hasher = blake3::Hasher::new();
		hasher.update(b"omp-tool/live/v1\0");
		for (name, rev) in &self.live {
			hash_field(&mut hasher, name.as_bytes());
			hash_field(&mut hasher, rev.family.as_bytes());
			hash_field(&mut hasher, &rev.n.to_le_bytes());
		}
		*hasher.finalize().as_bytes()
	}

	/// Dispatches only the live registered revision.
	pub fn invoke<'a>(
		&'a self,
		name: &str,
		params: IncomingParams<'a>,
	) -> Result<ErasedStream<'a>, RegistryError> {
		let entry = self.live_entry(name)?;
		if entry.route() == ToolRoute::Worker {
			return Err(external_error(entry.spec(), "invoke"));
		}
		Ok(entry.call(params))
	}

	/// Lowers every live spec and no historical spec for one selected route.
	pub fn advertise(&self, caps: LoweringCaps) -> Vec<LoweredTool> {
		self
			.live
			.iter()
			.filter_map(|(name, rev)| {
				let entry = self.versions.get(name)?.get(rev)?;
				Some(lower(entry.as_ref(), caps))
			})
			.collect()
	}

	/// Deterministically projects a structured live verdict through its tool.
	pub fn prompt(
		&self,
		identity: &ToolIdentity,
		verdict: &[u8],
		caps: &PromptCaps,
	) -> Result<Option<Vec<Part>>, RegistryError> {
		Ok(Some(self.project_verdict(identity, verdict, false, caps)?.parts))
	}

	/// Decodes one recorded verdict into current model parts and branch
	/// metadata.
	///
	/// The durable `recorded_useless` hint is preserved for tool-owned `Ok` and
	/// `Fault` branches. Harness-owned `Args` and `Aborted` branches always
	/// force it false.
	pub fn project_verdict(
		&self,
		identity: &ToolIdentity,
		verdict: &[u8],
		recorded_useless: bool,
		caps: &PromptCaps,
	) -> Result<ProjectedVerdict, RegistryError> {
		let entry = self
			.versions
			.get(&identity.name)
			.and_then(|versions| versions.get(&identity.rev))
			.ok_or_else(|| RegistryError::UnknownTool(identity.name.clone()))?;
		entry.project_verdict(verdict, recorded_useless, caps)
	}

	/// Projects one exact serialized update through its registered typed tool.
	pub fn invoke_input(
		&self,
		identity: &ToolIdentity,
		invocation_id: &str,
		json: &[u8],
	) -> Result<Option<InvokeInput>, RegistryError> {
		let entry = self
			.versions
			.get(&identity.name)
			.and_then(|versions| versions.get(&identity.rev))
			.ok_or_else(|| RegistryError::UnknownTool(identity.name.clone()))?;
		entry.invoke_input(invocation_id, json)
	}

	/// Composes registered adjacent lift steps toward the live revision.
	///
	/// Failure of any step returns the exact original bytes as `Data`; partially
	/// migrated history is never exposed or mistaken for a live schema.
	pub fn project(&self, original: RecordedCallOwned) -> ProjectedCall {
		let Some(live_rev) = self.live.get(&original.identity.name) else {
			return ProjectedCall::Data(original);
		};
		if &original.identity.rev == live_rev {
			return ProjectedCall::Live(original);
		}
		let Some(versions) = self.versions.get(&original.identity.name) else {
			return ProjectedCall::Data(original);
		};

		let mut current_rev = original.identity.rev.clone();
		let mut current =
			LiftedCall { raw_args: original.raw_args.clone(), verdict: original.verdict.clone() };
		while &current_rev != live_rev {
			let next_rev = if current_rev.family == live_rev.family && current_rev.n < live_rev.n {
				Rev { family: current_rev.family.clone(), n: current_rev.n.saturating_add(1) }
			} else {
				live_rev.clone()
			};
			let Some(step) = versions.get(&next_rev) else {
				return ProjectedCall::Data(original);
			};
			let Some(lifted) = step.lift(&current_rev, RecordedCall {
				raw_args: &current.raw_args,
				verdict:  &current.verdict,
			}) else {
				return ProjectedCall::Data(original);
			};
			current = lifted;
			current_rev = next_rev;
		}
		ProjectedCall::Live(RecordedCallOwned {
			identity: ToolIdentity { name: original.identity.name, rev: current_rev },
			raw_args: current.raw_args,
			verdict:  current.verdict,
		})
	}

	fn live_entry(&self, name: &str) -> Result<&dyn ErasedTool, RegistryError> {
		let rev = self
			.live
			.get(name)
			.ok_or_else(|| RegistryError::UnknownTool(Str::from(name)))?;
		self
			.versions
			.get(name)
			.and_then(|versions| versions.get(rev))
			.map(|entry| entry.as_ref())
			.ok_or_else(|| RegistryError::UnknownTool(Str::from(name)))
	}
}

fn hash_field(hasher: &mut blake3::Hasher, field: &[u8]) {
	let len = u64::try_from(field.len()).expect("tool identity length fits in u64");
	hasher.update(&len.to_le_bytes());
	hasher.update(field);
}

fn render_arg_issue(issue: &crate::ArgIssue) -> Str {
	let mut path = String::from("$");
	for segment in &issue.path {
		match segment {
			crate::ArgPath::Key(key) => {
				path.push('[');
				path.push_str(
					&serde_json::to_string(key.as_str()).unwrap_or_else(|_| "\"?\"".into()),
				);
				path.push(']');
			},
			crate::ArgPath::Index(index) => {
				path.push('[');
				path.push_str(&index.to_string());
				path.push(']');
			},
		}
	}
	let kind_json = serde_json::to_string(&issue.kind)
		.expect("serializing a fieldless argument issue kind cannot fail");
	let kind = kind_json.trim_matches('"');
	let mut text = format!("invalid arguments at {path}: expected {} ({kind})", issue.expected);
	if let Some(found) = &issue.found {
		text.push_str("; found ");
		text.push_str(found);
	}
	if let Some(example) = &issue.example {
		text.push_str("; example ");
		text.push_str(example);
	}
	Str::from(text)
}

fn render_abort(abort: &Abort) -> Str {
	match abort {
		Abort::Skipped { reason } => Str::from(format!("skipped: {reason}")),
		Abort::Interrupted { reason } => Str::from(format!("interrupted: {reason}")),
		Abort::EffectsUnknown { reason } => {
			Str::from(format!("aborted with effects unknown: {reason}"))
		},
		Abort::InputDropped => {
			Str::new_static("aborted: invocation input dropped before commit")
		},
		Abort::MissingOutcome => {
			Str::new_static("aborted: executor ended without a terminal outcome")
		},
	}
}


fn lower(entry: &dyn ErasedTool, caps: LoweringCaps) -> LoweredTool {
	let spec = entry.spec();
	let mut adjustments = Vec::new();
	let (input, disposition, priority) = match &spec.constraint {
		Constraint::None => (
			ToolInputConstraint::JsonSchema { parameters: entry.schema().clone(), strict: false },
			None,
			None,
		),
		Constraint::Schema { priority } if caps.strict_schema => (
			ToolInputConstraint::JsonSchema { parameters: entry.schema().clone(), strict: true },
			Some(ConstraintDisposition::Required),
			Some(*priority),
		),
		Constraint::Schema { priority } => {
			adjustments.push(dropped(&spec.name, "schema", "catalog.strict-schema-unsupported"));
			(
				ToolInputConstraint::JsonSchema {
					parameters: entry.schema().clone(),
					strict:     false,
				},
				Some(ConstraintDisposition::Prefer),
				Some(*priority),
			)
		},
		Constraint::Grammar { syntax, definition, priority }
			if caps.grammar.contains(grammar_bit(*syntax)) =>
		{
			(
				ToolInputConstraint::Grammar(ToolGrammar {
					syntax:     grammar_syntax(*syntax),
					definition: definition.clone(),
				}),
				Some(ConstraintDisposition::Required),
				Some(*priority),
			)
		},
		Constraint::Grammar { syntax, priority, .. } => {
			adjustments.push(dropped(
				&spec.name,
				grammar_name(*syntax),
				"catalog.grammar-unsupported",
			));
			(
				ToolInputConstraint::JsonSchema {
					parameters: entry.schema().clone(),
					strict:     false,
				},
				Some(ConstraintDisposition::Prefer),
				Some(*priority),
			)
		},
	};
	LoweredTool {
		identity: spec.identity(),
		definition: ToolDefinition {
			name: spec.name.clone(),
			description: Some(spec.description.clone()),
			input,
		},
		disposition,
		priority,
		adjustments,
	}
}

fn external_error(spec: &crate::ToolSpec, operation: &'static str) -> RegistryError {
	RegistryError::UnsupportedExternal {
		name: spec.name.clone(),
		rev: spec.rev.clone(),
		operation,
	}
}

fn grammar_syntax(syntax: GrammarSyntax) -> ToolGrammarSyntax {
	match syntax {
		GrammarSyntax::Lark => ToolGrammarSyntax::Lark,
		GrammarSyntax::Regex => ToolGrammarSyntax::Regex,
		GrammarSyntax::Ebnf => ToolGrammarSyntax::Ebnf,
	}
}

fn grammar_bit(syntax: GrammarSyntax) -> GrammarBits {
	match syntax {
		GrammarSyntax::Lark => GrammarBits::LARK,
		GrammarSyntax::Regex => GrammarBits::REGEX,
		GrammarSyntax::Ebnf => GrammarBits::EBNF,
	}
}

fn grammar_name(syntax: GrammarSyntax) -> &'static str {
	match syntax {
		GrammarSyntax::Lark => "lark",
		GrammarSyntax::Regex => "regex",
		GrammarSyntax::Ebnf => "ebnf",
	}
}

fn dropped(name: &Str, feature: &str, reason: &'static str) -> Adjustment {
	Adjustment::Dropped {
		feature: FeatureId(Str::from(format!("tool.{}.{}", name, feature))),
		reason:  ReasonId(Str::from(reason)),
	}
}
