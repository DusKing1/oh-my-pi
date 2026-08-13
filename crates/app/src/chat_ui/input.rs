use omp_core::Str;
use std::time::{SystemTime, UNIX_EPOCH};
use omp_proto::thread::v1::{Item, Message, Part, Role, item, part};

/// Actions parsed from user input in the chat shell.
#[derive(Debug, PartialEq)]
pub enum ChatCommand {
	/// Update the session model targeting the given identifier.
	Model(Str),
	/// Manually trigger a resume attempt.
	Resume,
	/// Exit the application cleanly.
	Quit,
	/// A plain text message to append as a user turn.
	Submit(Item),
}

/// Structured parsing failure for interactive input.
#[derive(Debug, PartialEq)]
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
			Self::UnknownCommand(cmd) => write!(f, "unknown slash command: {}", cmd),
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
		if text == "/resume" {
			return Ok(ChatCommand::Resume);
		}
		if text == "/quit" {
			return Ok(ChatCommand::Quit);
		}
		
		let cmd = text.split_whitespace().next().unwrap_or(text);
		return Err(InputError::UnknownCommand(Str::from(cmd)));
	}

	Ok(ChatCommand::Submit(Item {
		seq: 0,
		created_at_ms: now_ms(),
		kind: Some(item::Kind::Message(Message {
			role: i32::from(Role::User),
			parts: vec![Part {
				kind: Some(part::Kind::Text(text.to_owned())),
			}],
		})),
		props: None,
	}))
}
fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis()
		.try_into()
		.unwrap_or(u64::MAX)
}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_slash_commands() {
		assert_eq!(parse_input("/model smol"), Ok(ChatCommand::Model(Str::from("smol"))));
		assert_eq!(parse_input("/resume"), Ok(ChatCommand::Resume));
		assert_eq!(parse_input("/quit"), Ok(ChatCommand::Quit));
	}

	#[test]
	fn test_invalid_slash_commands() {
		assert_eq!(parse_input("/model  "), Err(InputError::EmptyModel));
		assert_eq!(parse_input("/model"), Err(InputError::EmptyModel));
		assert_eq!(parse_input("/unknown arg"), Err(InputError::UnknownCommand(Str::from("/unknown"))));
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
					}
					_ => panic!("wrong item kind"),
				}
			}
			_ => panic!("expected Submit"),
		}
	}
}