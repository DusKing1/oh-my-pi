use omp_core::Str;
use omp_proto::thread::v1::{Item, Message, Part, Role, item, part};

use super::now_ms;

/// Actions parsed from user input in the chat shell.
#[derive(Debug, PartialEq)]
pub enum ChatCommand {
	/// Show the commands implemented by this chat shell.
	Help,
	/// Start provider authentication, defaulting to the selected model's
	/// provider.
	Login(Option<Str>),
	/// Update the session model targeting the given identifier.
	Model(Str),
	/// Open the project-local durable-session picker.
	Resume,
	/// Exit the application cleanly.
	Quit,
	/// A plain text message to append as a user turn.
	Submit(Box<Item>),
}

/// Structured parsing failure for interactive input.
#[derive(Debug, PartialEq, Eq)]
pub enum InputError {
	/// The `/model` command was provided without an argument.
	EmptyModel,
	/// An unrecognized slash command was entered.
	UnknownCommand(Str),
}

impl std::fmt::Display for InputError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::EmptyModel => write!(f, "missing model name for /model command"),
			Self::UnknownCommand(cmd) => write!(f, "unknown slash command: {cmd}"),
		}
	}
}
impl std::error::Error for InputError {}

/// Parses raw text from the composer buffer into an actionable command.
pub fn parse_input(text: &str) -> Result<ChatCommand, InputError> {
	if text.starts_with('/') {
		let text = text.trim();
		if let Some(rest) = text.strip_prefix("/model ") {
			let model = rest.trim();
			if model.is_empty() {
				return Err(InputError::EmptyModel);
			}
			return Ok(ChatCommand::Model(Str::from(model)));
		}
		if text == "/model" {
			return Err(InputError::EmptyModel);
		}
		if text == "/help" {
			return Ok(ChatCommand::Help);
		}
		if text == "/login" {
			return Ok(ChatCommand::Login(None));
		}
		if let Some(rest) = text.strip_prefix("/login ") {
			let provider = rest.trim();
			return Ok(ChatCommand::Login((!provider.is_empty()).then(|| Str::from(provider))));
		}
		if text == "/resume" {
			return Ok(ChatCommand::Resume);
		}
		if text == "/quit" {
			return Ok(ChatCommand::Quit);
		}

		let cmd = text.split_whitespace().next().unwrap_or(text);
		return Err(InputError::UnknownCommand(Str::from(cmd)));
	}

	Ok(ChatCommand::Submit(Box::new(Item {
		seq:           0,
		created_at_ms: now_ms(),
		kind:          Some(item::Kind::Message(Message {
			role:  i32::from(Role::User),
			parts: vec![Part { kind: Some(part::Kind::Text(text.to_owned())) }],
		})),
		props:         None,
	})))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_slash_commands() {
		assert_eq!(parse_input("/model smol"), Ok(ChatCommand::Model(Str::from("smol"))));
		assert_eq!(parse_input("/help"), Ok(ChatCommand::Help));
		assert_eq!(parse_input("/login"), Ok(ChatCommand::Login(None)));
		assert_eq!(
			parse_input("/login kimi-code"),
			Ok(ChatCommand::Login(Some(Str::from("kimi-code"))))
		);
		assert_eq!(parse_input("/resume"), Ok(ChatCommand::Resume));
		assert_eq!(parse_input("/quit"), Ok(ChatCommand::Quit));
	}

	#[test]
	fn test_invalid_slash_commands() {
		assert_eq!(parse_input("/model  "), Err(InputError::EmptyModel));
		assert_eq!(parse_input("/model"), Err(InputError::EmptyModel));
		assert_eq!(
			parse_input("/unknown arg"),
			Err(InputError::UnknownCommand(Str::from("/unknown")))
		);
	}

	#[test]
	fn test_plain_text() {
		let result = parse_input("hello world").unwrap();
		match result {
			ChatCommand::Submit(item) => {
				assert_eq!(item.seq, 0);
				assert!(item.created_at_ms > 0);
				let kind = item.kind.unwrap();
				match kind {
					item::Kind::Message(msg) => {
						assert_eq!(msg.role, Role::User as i32);
						let part = &msg.parts[0];
						match part.kind.as_ref().unwrap() {
							part::Kind::Text(t) => assert_eq!(t, "hello world"),
							_ => panic!("wrong part kind"),
						}
					},
					_ => panic!("wrong item kind"),
				}
			},
			_ => panic!("expected Submit"),
		}
	}
}
