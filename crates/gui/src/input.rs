//! winit → omp-tui input translation.
//!
//! Chords map through the *physical* key code so `Ctrl+P` means the same
//! keycap a terminal would report regardless of layout or Option-modified
//! characters; plain text rides the event's resolved `text`.

use omp_tui::{Key, Mods, MouseButton};
use winit::{
	event::MouseButton as WinitButton,
	keyboard::{Key as WinitKey, KeyCode, ModifiersState, NamedKey, PhysicalKey},
};

/// winit modifiers → the report's modifier bits.
pub fn modifiers(state: ModifiersState) -> Mods {
	Mods {
		shift:     state.shift_key(),
		alt:       state.alt_key(),
		ctrl:      state.control_key(),
		super_key: state.super_key(),
		hyper:     false,
		meta:      false,
	}
}

/// Maps one key press to the terminal-vocabulary key, or `None` for host
/// chrome chords (⌘-prefixed) and unrecognized input.
pub fn map_key(event: &winit::event::KeyEvent, mods: ModifiersState) -> Option<Key> {
	if mods.super_key() {
		return None;
	}
	match &event.logical_key {
		WinitKey::Named(named) => map_named(*named, mods),
		WinitKey::Character(text) => {
			let letter = letter_of(&event.physical_key);
			match (mods.control_key(), mods.alt_key()) {
				(true, true) => return letter.map(Key::CtrlAlt),
				(true, false) => {
					if mods.shift_key() && letter == Some('v') {
						return Some(Key::PasteRaw);
					}
					if letter == Some('v') {
						return Some(Key::Paste);
					}
					return letter.map(Key::Ctrl);
				},
				(false, true) => return letter.map(Key::Alt),
				(false, false) => {},
			}
			let mut chars = text.chars();
			let c = chars.next()?;
			Some(if c == ' ' && chars.next().is_none() {
				Key::Space
			} else {
				Key::Char(c)
			})
		},
		_ => None,
	}
}

/// Maps a named (non-character) key with its modifiers; shift promotes
/// motions to their selection-extending counterparts.
fn map_named(named: NamedKey, mods: ModifiersState) -> Option<Key> {
	Some(match named {
		NamedKey::ArrowUp if mods.shift_key() => Key::SelectUp,
		NamedKey::ArrowUp => Key::Up,
		NamedKey::ArrowDown if mods.shift_key() => Key::SelectDown,
		NamedKey::ArrowDown => Key::Down,
		NamedKey::ArrowLeft if mods.shift_key() && (mods.alt_key() || mods.control_key()) => {
			Key::SelectWordLeft
		},
		NamedKey::ArrowRight if mods.shift_key() && (mods.alt_key() || mods.control_key()) => {
			Key::SelectWordRight
		},
		NamedKey::ArrowLeft if mods.shift_key() => Key::SelectLeft,
		NamedKey::ArrowRight if mods.shift_key() => Key::SelectRight,
		NamedKey::ArrowLeft if mods.alt_key() || mods.control_key() => Key::WordLeft,
		NamedKey::ArrowRight if mods.alt_key() || mods.control_key() => Key::WordRight,
		NamedKey::ArrowLeft => Key::Left,
		NamedKey::ArrowRight => Key::Right,
		NamedKey::Tab if mods.shift_key() => Key::BackTab,
		NamedKey::Tab => Key::Tab,
		NamedKey::Enter if mods.shift_key() => Key::ShiftEnter,
		NamedKey::Enter => Key::Enter,
		NamedKey::Escape => Key::Esc,
		NamedKey::Backspace => Key::Backspace,
		NamedKey::Delete if mods.alt_key() => Key::WordDelete,
		NamedKey::Delete => Key::Delete,
		NamedKey::Insert => Key::Insert,
		NamedKey::Home if mods.shift_key() => Key::SelectHome,
		NamedKey::Home => Key::Home,
		NamedKey::End if mods.shift_key() => Key::SelectEnd,
		NamedKey::End => Key::End,
		NamedKey::PageUp => Key::PageUp,
		NamedKey::PageDown => Key::PageDown,
		NamedKey::F1 => Key::Function(1),
		NamedKey::F2 => Key::Function(2),
		NamedKey::F3 => Key::Function(3),
		NamedKey::F4 => Key::Function(4),
		NamedKey::F5 => Key::Function(5),
		NamedKey::F6 => Key::Function(6),
		NamedKey::F7 => Key::Function(7),
		NamedKey::F8 => Key::Function(8),
		NamedKey::F9 => Key::Function(9),
		NamedKey::F10 => Key::Function(10),
		NamedKey::F11 => Key::Function(11),
		NamedKey::F12 => Key::Function(12),
		NamedKey::Space => Key::Space,
		_ => return None,
	})
}

/// The QWERTY keycap character of a physical key, matching terminal chord
/// reports: letters, digits, and the `- = [ ] , .` symbol keys.
pub fn letter_of(physical: &PhysicalKey) -> Option<char> {
	let PhysicalKey::Code(code) = physical else {
		return None;
	};
	Some(match code {
		KeyCode::KeyA => 'a',
		KeyCode::KeyB => 'b',
		KeyCode::KeyC => 'c',
		KeyCode::KeyD => 'd',
		KeyCode::KeyE => 'e',
		KeyCode::KeyF => 'f',
		KeyCode::KeyG => 'g',
		KeyCode::KeyH => 'h',
		KeyCode::KeyI => 'i',
		KeyCode::KeyJ => 'j',
		KeyCode::KeyK => 'k',
		KeyCode::KeyL => 'l',
		KeyCode::KeyM => 'm',
		KeyCode::KeyN => 'n',
		KeyCode::KeyO => 'o',
		KeyCode::KeyP => 'p',
		KeyCode::KeyQ => 'q',
		KeyCode::KeyR => 'r',
		KeyCode::KeyS => 's',
		KeyCode::KeyT => 't',
		KeyCode::KeyU => 'u',
		KeyCode::KeyV => 'v',
		KeyCode::KeyW => 'w',
		KeyCode::KeyX => 'x',
		KeyCode::KeyY => 'y',
		KeyCode::KeyZ => 'z',
		KeyCode::Digit0 => '0',
		KeyCode::Digit1 => '1',
		KeyCode::Digit2 => '2',
		KeyCode::Digit3 => '3',
		KeyCode::Digit4 => '4',
		KeyCode::Digit5 => '5',
		KeyCode::Digit6 => '6',
		KeyCode::Digit7 => '7',
		KeyCode::Digit8 => '8',
		KeyCode::Digit9 => '9',
		KeyCode::Minus => '-',
		KeyCode::Equal => '=',
		KeyCode::BracketLeft => '[',
		KeyCode::BracketRight => ']',
		KeyCode::Comma => ',',
		KeyCode::Period => '.',
		_ => return None,
	})
}

/// winit button → the report's physical button vocabulary.
pub fn map_button(button: WinitButton) -> Option<MouseButton> {
	Some(match button {
		WinitButton::Left => MouseButton::Left,
		WinitButton::Right => MouseButton::Right,
		WinitButton::Middle => MouseButton::Middle,
		_ => return None,
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn shift_promotes_motions_to_selection_keys() {
		let shift = ModifiersState::SHIFT;
		let cases = [
			(NamedKey::ArrowLeft, Key::SelectLeft),
			(NamedKey::ArrowRight, Key::SelectRight),
			(NamedKey::ArrowUp, Key::SelectUp),
			(NamedKey::ArrowDown, Key::SelectDown),
			(NamedKey::Home, Key::SelectHome),
			(NamedKey::End, Key::SelectEnd),
		];
		for (named, expected) in cases {
			assert_eq!(map_named(named, shift), Some(expected));
		}
	}

	#[test]
	fn shifted_word_motions_extend_the_selection() {
		for word_mods in [ModifiersState::ALT, ModifiersState::CONTROL] {
			let mods = ModifiersState::SHIFT | word_mods;
			assert_eq!(map_named(NamedKey::ArrowLeft, mods), Some(Key::SelectWordLeft));
			assert_eq!(map_named(NamedKey::ArrowRight, mods), Some(Key::SelectWordRight));
			assert_eq!(map_named(NamedKey::ArrowLeft, word_mods), Some(Key::WordLeft));
			assert_eq!(map_named(NamedKey::ArrowRight, word_mods), Some(Key::WordRight));
		}
	}

	#[test]
	fn plain_motions_stay_unpromoted() {
		let none = ModifiersState::empty();
		assert_eq!(map_named(NamedKey::ArrowLeft, none), Some(Key::Left));
		assert_eq!(map_named(NamedKey::Home, none), Some(Key::Home));
		assert_eq!(map_named(NamedKey::Tab, ModifiersState::SHIFT), Some(Key::BackTab));
	}

	#[test]
	fn letter_of_covers_digits_and_symbol_row() {
		let cases = [
			(KeyCode::Digit7, '7'),
			(KeyCode::BracketRight, ']'),
			(KeyCode::Equal, '='),
			(KeyCode::Minus, '-'),
			(KeyCode::Comma, ','),
			(KeyCode::Period, '.'),
		];
		for (code, expected) in cases {
			assert_eq!(letter_of(&PhysicalKey::Code(code)), Some(expected));
		}
	}
}
