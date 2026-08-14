use omp_core::Str;
use omp_proto::thread::v1::{Item, Message, Part, Role, item, part};
use omp_tui::Command;

use super::now_ms;

/// Metadata shared by slash-command completion and `/help`.
pub struct CommandSpec {
	/// Command token without the leading slash.
	pub(crate) name:        &'static str,
	/// Human-readable completion and help text.
	pub(crate) description: &'static str,
	/// Optional argument hint appended by help and completion.
	pub(crate) usage:       &'static str,
}

/// Canonical slash-command vocabulary shared by completion, help, and the
/// command palette.
pub const COMMANDS: &[CommandSpec] = &[
	CommandSpec {
		name:        "help",
		description: "Show commands and keyboard controls",
		usage:       "",
	},
	CommandSpec {
		name:        "login",
		description: "Authenticate a provider",
		usage:       "[provider]",
	},
	CommandSpec {
		name:        "model",
		description: "Change the selected model",
		usage:       "<model>",
	},
	CommandSpec { name: "models", description: "Browse available models", usage: "" },
	CommandSpec {
		name:        "resume",
		description: "Open another project session",
		usage:       "",
	},
	CommandSpec { name: "quit", description: "Exit the application", usage: "" },
];

/// Slash commands offered by the chat composer's completion palette.
pub fn commands() -> Vec<Command> {
	COMMANDS
		.iter()
		.map(|spec| {
			let command = Command::new(spec.name, spec.description, &[]);
			if spec.usage.is_empty() {
				command
			} else {
				command.with_hint(spec.usage)
			}
		})
		.collect()
}

/// Renders the discoverable slash-command reference from the completion
/// vocabulary.
pub fn help_text() -> String {
	let mut help = String::from("**Commands**\n");
	for spec in COMMANDS {
		help.push_str("- `/");
		help.push_str(spec.name);
		if !spec.usage.is_empty() {
			help.push(' ');
			help.push_str(spec.usage);
		}
		help.push_str("` — ");
		help.push_str(spec.description);
		help.push('\n');
	}
	help.push_str(
		"\n**Keys**\nesc interrupt · esc esc rewind · enter enter interrupt+send · alt+enter \
		 follow-up",
	);
	help
}

/// Actions parsed from user input in the chat shell.
#[derive(Debug, PartialEq)]
pub enum ChatCommand {
	/// Ignore an empty composer submission.
	Nothing,
	/// Show the commands implemented by this chat shell.
	Help,
	/// Start provider authentication, defaulting to the selected model's
	/// provider.
	Login(Option<Str>),
	/// Update the session model targeting the given identifier.
	Model(Str),
	/// Open the catalog model picker.
	ModelPicker,
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
	/// An unrecognized slash command was entered.
	UnknownCommand(Str),
}

impl std::fmt::Display for InputError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::UnknownCommand(cmd) => write!(f, "unknown slash command: {cmd}"),
		}
	}
}
impl std::error::Error for InputError {}

/// Parses raw text from the composer buffer into an actionable command.
pub fn parse_input(text: &str) -> Result<ChatCommand, InputError> {
	if text.trim().is_empty() {
		return Ok(ChatCommand::Nothing);
	}

	// A command token never contains a second `/`: an expanded attachment
	// payload like `/tmp/pic.png describe this` is a message, not a command.
	let first = text.split_whitespace().next().unwrap_or_default();
	if text.starts_with('/') && !first[1..].contains('/') {
		let text = text.trim();
		if let Some(rest) = text.strip_prefix("/model ") {
			let model = rest.trim();
			if !model.is_empty() {
				return Ok(ChatCommand::Model(Str::from(model)));
			}
		}
		if text == "/model" || text == "/models" || text.starts_with("/model ") {
			return Ok(ChatCommand::ModelPicker);
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

	Ok(ChatCommand::Submit(Box::new(user_message(text))))
}

/// Builds the canonical user-message item used by submissions and steering.
pub(super) fn user_message(text: impl Into<String>) -> Item {
	Item {
		seq:           0,
		created_at_ms: now_ms(),
		kind:          Some(item::Kind::Message(Message {
			role:  i32::from(Role::User),
			parts: vec![Part { kind: Some(part::Kind::Text(text.into())) }],
		})),
		props:         None,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_slash_commands() {
		assert_eq!(parse_input("/model smol"), Ok(ChatCommand::Model(Str::from("smol"))));
		assert_eq!(parse_input("/model"), Ok(ChatCommand::ModelPicker));
		assert_eq!(parse_input("/model  "), Ok(ChatCommand::ModelPicker));
		assert_eq!(parse_input("/models"), Ok(ChatCommand::ModelPicker));
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
	fn help_uses_the_completion_vocabulary_and_usages() {
		let help = help_text();
		for spec in COMMANDS {
			let usage = if spec.usage.is_empty() {
				format!("/{}", spec.name)
			} else {
				format!("/{} {}", spec.name, spec.usage)
			};
			assert!(help.contains(&usage), "help omitted {usage}");
			assert!(help.contains(spec.description), "help omitted description for {}", spec.name);
		}
	}

	#[test]
	fn test_invalid_slash_commands() {
		assert_eq!(
			parse_input("/unknown arg"),
			Err(InputError::UnknownCommand(Str::from("/unknown")))
		);
	}

	#[test]
	fn path_payloads_submit_as_plain_messages() {
		let parsed = parse_input("/tmp/pic.png describe this image");
		let Ok(ChatCommand::Submit(item)) = parsed else {
			panic!("expanded attachment payload parsed as {parsed:?}");
		};
		let Some(item::Kind::Message(message)) = item.kind else {
			panic!("submit item lost its message");
		};
		assert_eq!(
			message.parts[0].kind,
			Some(part::Kind::Text("/tmp/pic.png describe this image".to_owned()))
		);
	}

	#[test]
	fn blank_input_is_nothing() {
		assert_eq!(parse_input(""), Ok(ChatCommand::Nothing));
		assert_eq!(parse_input(" \t\n"), Ok(ChatCommand::Nothing));
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
