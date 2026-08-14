//! Derive macro backing `omp_tool::ToolParam` model-facing JSON Schemas.
//!
//! The derive reads the same field metadata serde uses to deserialize tool
//! arguments and emits a deterministic, inline JSON Schema:
//!
//! - Doc comments become property `description`s (lines joined with spaces,
//!   blank doc lines becoming newlines).
//! - `Option<T>` fields follow the absent-property convention: they are left
//!   out of `required` and never accept `null` (opt back in with
//!   `#[param(nullable)]`).
//! - `#[serde(rename, rename_all, default, skip, deny_unknown_fields)]` are
//!   honored so the schema always describes deserialization.
//! - `#[param(...)]` carries the schema-only knobs: `description` overrides
//!   (empty string suppresses), `minimum`/`maximum`/`min_length`/`max_length`
//!   bounds, `nullable`, and `extend({ ... })` merging raw JSON Schema into
//!   the surrounding object.
//!
//! Unit-variant enums derive as `{"type": "string", "enum": [...]}`.

use proc_macro2::{Delimiter, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::{Attribute, Data, DeriveInput, Expr, ExprLit, Fields, Lit, LitStr, Meta, Token};

/// Expands `#[derive(ToolParam)]`, lowering parse failures to compile errors.
pub fn derive(input: TokenStream2) -> TokenStream2 {
	syn::parse2(input)
		.and_then(|input: DeriveInput| expand(&input))
		.unwrap_or_else(|error| error.to_compile_error())
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
	let body = match &input.data {
		Data::Struct(data) => match &data.fields {
			Fields::Named(fields) => expand_struct(input, fields)?,
			_ => {
				return Err(syn::Error::new_spanned(
					&input.ident,
					"ToolParam structs must have named fields",
				));
			},
		},
		Data::Enum(data) => expand_enum(input, data)?,
		Data::Union(_) => {
			return Err(syn::Error::new_spanned(&input.ident, "ToolParam cannot derive unions"));
		},
	};
	let ident = &input.ident;
	let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
	Ok(quote! {
		#[automatically_derived]
		impl #impl_generics ::omp_tool::ToolParam for #ident #ty_generics #where_clause {
			fn schema() -> ::omp_tool::__private::serde_json::Value {
				use ::omp_tool::__private::serde_json as __json;
				#body
			}
		}
	})
}

fn expand_struct(input: &DeriveInput, fields: &syn::FieldsNamed) -> syn::Result<TokenStream2> {
	let container_serde = container_serde(&input.attrs)?;
	let container_param = param_attrs(&input.attrs)?;
	let mut blocks = Vec::new();
	for field in &fields.named {
		let serde = field_serde(&field.attrs)?;
		if serde.skip {
			continue;
		}
		let param = param_attrs(&field.attrs)?;
		let ident = field.ident.as_ref().expect("named field has an ident");
		let wire = match serde.rename {
			Some(rename) => rename,
			None => match container_serde.rename_all {
				Some(rule) => rule.apply_to_field(&ident.to_string()),
				None => ident.to_string(),
			},
		};
		let mut steps = TokenStream2::new();
		let description = match &param.description {
			Some(text) => (!text.is_empty()).then(|| text.clone()),
			None => doc_text(&field.attrs),
		};
		if let Some(text) = description {
			steps.extend(quote! {
				prop.insert(
					::std::string::String::from("description"),
					__json::Value::String(::std::string::String::from(#text)),
				);
			});
		}
		let ty = &field.ty;
		steps.extend(quote! {
			::omp_tool::__private::merge_defaults(
				&mut prop,
				<#ty as ::omp_tool::ToolParam>::schema(),
			);
		});
		for (key, value) in &param.bounds {
			steps.extend(quote! {
				prop.insert(::std::string::String::from(#key), __json::json!(#value));
			});
		}
		if let Some(value) = &param.default {
			steps.extend(quote! {
				prop.insert(::std::string::String::from("default"), __json::json!(#value));
			});
		}
		if param.nullable {
			steps.extend(quote! { ::omp_tool::__private::nullable(&mut prop); });
		}
		if let Some(extension) = &param.extend {
			steps.extend(quote! {
				::omp_tool::__private::merge_override(&mut prop, __json::json!(#extension));
			});
		}
		if !serde.default {
			steps.extend(quote! {
				if !<#ty as ::omp_tool::ToolParam>::OPTIONAL {
					required.push(__json::Value::String(::std::string::String::from(#wire)));
				}
			});
		}
		blocks.push(quote! {
			{
				let mut prop = __json::Map::new();
				#steps
				properties.insert(::std::string::String::from(#wire), __json::Value::Object(prop));
			}
		});
	}
	let description_step = doc_text(&input.attrs).map(|text| {
		quote! {
			root.insert(
				::std::string::String::from("description"),
				__json::Value::String(::std::string::String::from(#text)),
			);
		}
	});
	let deny_step = container_serde.deny_unknown_fields.then(|| {
		quote! {
			root.insert(::std::string::String::from("additionalProperties"), __json::Value::Bool(false));
		}
	});
	let extend_step = container_param.extend.as_ref().map(|extension| {
		quote! {
			::omp_tool::__private::merge_override(&mut root, __json::json!(#extension));
		}
	});
	Ok(quote! {
		let mut properties = __json::Map::new();
		let mut required = ::std::vec::Vec::new();
		#(#blocks)*
		let mut root = __json::Map::new();
		#description_step
		root.insert(
			::std::string::String::from("type"),
			__json::Value::String(::std::string::String::from("object")),
		);
		root.insert(::std::string::String::from("properties"), __json::Value::Object(properties));
		if !required.is_empty() {
			root.insert(::std::string::String::from("required"), __json::Value::Array(required));
		}
		#deny_step
		#extend_step
		__json::Value::Object(root)
	})
}

fn expand_enum(input: &DeriveInput, data: &syn::DataEnum) -> syn::Result<TokenStream2> {
	let container_serde = container_serde(&input.attrs)?;
	let mut values = Vec::new();
	for variant in &data.variants {
		if !matches!(variant.fields, Fields::Unit) {
			return Err(syn::Error::new_spanned(
				variant,
				"ToolParam enums must contain only unit variants",
			));
		}
		let serde = field_serde(&variant.attrs)?;
		if serde.skip {
			continue;
		}
		values.push(match serde.rename {
			Some(rename) => rename,
			None => match container_serde.rename_all {
				Some(rule) => rule.apply_to_variant(&variant.ident.to_string()),
				None => variant.ident.to_string(),
			},
		});
	}
	let description_step = doc_text(&input.attrs).map(|text| {
		quote! {
			root.insert(
				::std::string::String::from("description"),
				__json::Value::String(::std::string::String::from(#text)),
			);
		}
	});
	Ok(quote! {
		let mut root = __json::Map::new();
		#description_step
		root.insert(
			::std::string::String::from("type"),
			__json::Value::String(::std::string::String::from("string")),
		);
		root.insert(::std::string::String::from("enum"), __json::json!([#(#values),*]));
		__json::Value::Object(root)
	})
}

/// Container-level serde metadata the schema must mirror.
#[derive(Default)]
struct ContainerSerde {
	deny_unknown_fields: bool,
	rename_all:          Option<RenameRule>,
}

fn container_serde(attrs: &[Attribute]) -> syn::Result<ContainerSerde> {
	let mut out = ContainerSerde::default();
	for attr in attrs {
		if !attr.path().is_ident("serde") {
			continue;
		}
		attr.parse_nested_meta(|meta| {
			if meta.path.is_ident("deny_unknown_fields") {
				out.deny_unknown_fields = true;
			} else if meta.path.is_ident("rename_all") {
				let rule: LitStr = meta.value()?.parse()?;
				out.rename_all = Some(RenameRule::parse(&rule)?);
			} else {
				skip_unknown(&meta)?;
			}
			Ok(())
		})?;
	}
	Ok(out)
}

/// Field- or variant-level serde metadata the schema must mirror.
#[derive(Default)]
struct FieldSerde {
	rename:  Option<String>,
	default: bool,
	skip:    bool,
}

fn field_serde(attrs: &[Attribute]) -> syn::Result<FieldSerde> {
	let mut out = FieldSerde::default();
	for attr in attrs {
		if !attr.path().is_ident("serde") {
			continue;
		}
		attr.parse_nested_meta(|meta| {
			if meta.path.is_ident("rename") {
				let rename: LitStr = meta.value()?.parse()?;
				out.rename = Some(rename.value());
			} else if meta.path.is_ident("default") {
				out.default = true;
				skip_unknown(&meta)?;
			} else if meta.path.is_ident("skip") || meta.path.is_ident("skip_deserializing") {
				out.skip = true;
			} else if meta.path.is_ident("flatten") {
				return Err(meta.error("ToolParam does not support #[serde(flatten)]"));
			} else {
				skip_unknown(&meta)?;
			}
			Ok(())
		})?;
	}
	Ok(out)
}

/// Schema-only knobs carried by `#[param(...)]`.
#[derive(Default)]
struct ParamAttrs {
	description: Option<String>,
	nullable:    bool,
	bounds:      Vec<(&'static str, Lit)>,
	default:     Option<Lit>,
	extend:      Option<TokenStream2>,
}

fn param_attrs(attrs: &[Attribute]) -> syn::Result<ParamAttrs> {
	let mut out = ParamAttrs::default();
	for attr in attrs {
		if !attr.path().is_ident("param") {
			continue;
		}
		attr.parse_nested_meta(|meta| {
			let bound_key = ["minimum", "maximum", "min_length", "max_length"]
				.into_iter()
				.find(|key| meta.path.is_ident(key));
			if meta.path.is_ident("description") {
				let text: LitStr = meta.value()?.parse()?;
				out.description = Some(text.value());
			} else if meta.path.is_ident("nullable") {
				out.nullable = true;
			} else if let Some(key) = bound_key {
				let value: Lit = meta.value()?.parse()?;
				if !matches!(value, Lit::Int(_) | Lit::Float(_)) {
					return Err(meta.error("schema bounds take a numeric literal"));
				}
				let json_key = match key {
					"min_length" => "minLength",
					"max_length" => "maxLength",
					other => other,
				};
				out.bounds.push((json_key, value));
			} else if meta.path.is_ident("default") {
				out.default = Some(meta.value()?.parse()?);
			} else if meta.path.is_ident("extend") {
				let content;
				syn::parenthesized!(content in meta.input);
				let tokens: TokenStream2 = content.parse()?;
				let mut trees = tokens.clone().into_iter();
				match (trees.next(), trees.next()) {
					(Some(TokenTree::Group(group)), None)
						if group.delimiter() == Delimiter::Brace => {},
					_ => return Err(meta.error("extend takes a single JSON object: extend({ ... })")),
				}
				out.extend = Some(tokens);
			} else {
				return Err(meta.error(
					"unknown param attribute; expected description, nullable, minimum, maximum, \
					 min_length, max_length, default, or extend",
				));
			}
			Ok(())
		})?;
	}
	Ok(out)
}

/// Consumes an unrecognized nested-meta value or list so foreign serde
/// attributes never break the derive.
fn skip_unknown(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<()> {
	if meta.input.peek(Token![=]) {
		meta.value()?.parse::<Expr>()?;
	} else if meta.input.peek(syn::token::Paren) {
		let content;
		syn::parenthesized!(content in meta.input);
		content.parse::<TokenStream2>()?;
	}
	Ok(())
}

/// Joins doc-comment lines into one model-facing description.
///
/// Lines are trimmed and joined with single spaces; blank doc lines become
/// newlines so intentional paragraph breaks survive.
fn doc_text(attrs: &[Attribute]) -> Option<String> {
	let mut out = String::new();
	let mut pending_break = false;
	for attr in attrs {
		if !attr.path().is_ident("doc") {
			continue;
		}
		let Meta::NameValue(pair) = &attr.meta else { continue };
		let Expr::Lit(ExprLit { lit: Lit::Str(text), .. }) = &pair.value else { continue };
		let line = text.value();
		let line = line.trim();
		if line.is_empty() {
			pending_break = !out.is_empty();
			continue;
		}
		if pending_break {
			out.push('\n');
			pending_break = false;
		} else if !out.is_empty() {
			out.push(' ');
		}
		out.push_str(line);
	}
	(!out.is_empty()).then_some(out)
}

/// Serde `rename_all` case conventions.
#[derive(Clone, Copy, strum::EnumString)]
enum RenameRule {
	#[strum(serialize = "lowercase")]
	Lower,
	#[strum(serialize = "UPPERCASE")]
	Upper,
	#[strum(serialize = "PascalCase")]
	Pascal,
	#[strum(serialize = "camelCase")]
	Camel,
	#[strum(serialize = "snake_case")]
	Snake,
	#[strum(serialize = "SCREAMING_SNAKE_CASE")]
	ScreamingSnake,
	#[strum(serialize = "kebab-case")]
	Kebab,
	#[strum(serialize = "SCREAMING-KEBAB-CASE")]
	ScreamingKebab,
}

impl RenameRule {
	fn parse(rule: &LitStr) -> syn::Result<Self> {
		rule.value().parse().map_err(|_| {
			syn::Error::new_spanned(
				rule,
				format!("unsupported rename_all rule `{}`", rule.value()),
			)
		})
	}

	/// Applies this rule to a `PascalCase` variant name, mirroring serde.
	fn apply_to_variant(self, name: &str) -> String {
		match self {
			Self::Lower => name.to_ascii_lowercase(),
			Self::Upper => name.to_ascii_uppercase(),
			Self::Pascal => name.to_owned(),
			Self::Camel => decapitalize(name),
			Self::Snake => separate_words(name, '_', false),
			Self::ScreamingSnake => separate_words(name, '_', true),
			Self::Kebab => separate_words(name, '-', false),
			Self::ScreamingKebab => separate_words(name, '-', true),
		}
	}

	/// Applies this rule to a `snake_case` field name, mirroring serde.
	fn apply_to_field(self, name: &str) -> String {
		match self {
			Self::Lower | Self::Snake => name.to_owned(),
			Self::Upper | Self::ScreamingSnake => name.to_ascii_uppercase(),
			Self::Pascal => capitalize_words(name),
			Self::Camel => decapitalize(&capitalize_words(name)),
			Self::Kebab => name.replace('_', "-"),
			Self::ScreamingKebab => name.to_ascii_uppercase().replace('_', "-"),
		}
	}
}

fn decapitalize(name: &str) -> String {
	let mut chars = name.chars();
	match chars.next() {
		Some(first) => first.to_ascii_lowercase().to_string() + chars.as_str(),
		None => String::new(),
	}
}

fn capitalize_words(snake: &str) -> String {
	snake
		.split('_')
		.map(|word| {
			let mut chars = word.chars();
			match chars.next() {
				Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
				None => String::new(),
			}
		})
		.collect()
}

fn separate_words(pascal: &str, separator: char, screaming: bool) -> String {
	let mut out = String::with_capacity(pascal.len() + 4);
	for (index, ch) in pascal.char_indices() {
		if ch.is_ascii_uppercase() && index > 0 {
			out.push(separator);
		}
		out.push(if screaming { ch.to_ascii_uppercase() } else { ch.to_ascii_lowercase() });
	}
	out
}
