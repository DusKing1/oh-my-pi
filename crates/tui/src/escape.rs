/// Builds allocation-free terminal control strings from semantic commands.
///
/// Bare mode names enable the mode, `!mode` disables it, and `?mode` queries
/// modes that expose DECRQM. String literals append verbatim; every item is
/// concatenated at compile time.
macro_rules! esc {
	(@acc [$($output:tt)*]) => {
		concat!($($output)*)
	};
	(@acc [$($output:tt)*] , $($rest:tt)*) => {
		esc!(@acc [$($output)*] $($rest)*)
	};
	(@acc [$($output:tt)*] $literal:literal $($rest:tt)*) => {
		esc!(@acc [$($output)* $literal,] $($rest)*)
	};

	// Control introducers and terminators.
	(@acc [$($output:tt)*] escape $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b",] $($rest)*)
	};
	(@acc [$($output:tt)*] csi $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[",] $($rest)*)
	};
	(@acc [$($output:tt)*] osc $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b]",] $($rest)*)
	};
	(@acc [$($output:tt)*] dcs $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1bP",] $($rest)*)
	};
	(@acc [$($output:tt)*] apc $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b_",] $($rest)*)
	};
	(@acc [$($output:tt)*] st $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b\\",] $($rest)*)
	};
	(@acc [$($output:tt)*] bel $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x07",] $($rest)*)
	};

	// ANSI, DEC, and xterm modes: bare sets, `!` resets, and `?` queries.
	(@acc [$($output:tt)*] insert_mode $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[4h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! insert_mode $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[4l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? insert_mode $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[4$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] newline_mode $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[20h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! newline_mode $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[20l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? newline_mode $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[20$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] alt_screen $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1049h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! alt_screen $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1049l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? alt_screen $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1049$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] app_cursor_keys $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! app_cursor_keys $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? app_cursor_keys $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] app_keypad $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b=",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! app_keypad $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b>",] $($rest)*)
	};
	(@acc [$($output:tt)*] appearance_notifications $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2031h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! appearance_notifications $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2031l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? appearance_notifications $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2031$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] autowrap $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?7h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! autowrap $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?7l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? autowrap $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?7$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] bracketed_paste $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2004h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! bracketed_paste $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2004l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? bracketed_paste $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2004$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] cursor_visible $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?25h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! cursor_visible $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?25l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? cursor_visible $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?25$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] in_band_resize $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2048h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! in_band_resize $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2048l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? in_band_resize $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2048$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] modify_other_keys $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[>4;2m",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! modify_other_keys $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[>4;0m",] $($rest)*)
	};
	(@acc [$($output:tt)*] mouse_any_event $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1003h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! mouse_any_event $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1003l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? mouse_any_event $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1003$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] mouse_button_event $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1002h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! mouse_button_event $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1002l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? mouse_button_event $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1002$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] mouse_sgr $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1006h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! mouse_sgr $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1006l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? mouse_sgr $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1006$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] mouse_vt200 $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1000h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! mouse_vt200 $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1000l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? mouse_vt200 $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1000$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] origin $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?6h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! origin $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?6l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? origin $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?6$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] paste_events $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?5522h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! paste_events $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?5522l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? paste_events $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?5522$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] scroll_on_key_press $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1011h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! scroll_on_key_press $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1011l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? scroll_on_key_press $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1011$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] scroll_on_output $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1010h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! scroll_on_output $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1010l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? scroll_on_output $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?1010$p",] $($rest)*)
	};
	(@acc [$($output:tt)*] sync_output $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2026h",] $($rest)*)
	};
	(@acc [$($output:tt)*] ! sync_output $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2026l",] $($rest)*)
	};
	(@acc [$($output:tt)*] ? sync_output $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2026$p",] $($rest)*)
	};

	// Static capability and protocol queries.
	(@acc [$($output:tt)*] background_color_query $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b]11;?\x07",] $($rest)*)
	};
	(@acc [$($output:tt)*] cell_pixels_query $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[16t",] $($rest)*)
	};
	(@acc [$($output:tt)*] kitty_graphics_query $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\",] $($rest)*)
	};
	(@acc [$($output:tt)*] kitty_keyboard_query $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?u",] $($rest)*)
	};
	(@acc [$($output:tt)*] osc99_query $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b]99;i=omp-tui:p=?;\x1b\\",] $($rest)*)
	};
	(@acc [$($output:tt)*] primary_device_attributes_query $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[c",] $($rest)*)
	};
	(@acc [$($output:tt)*] sixel_color_registers_query $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[?2;1;0S",] $($rest)*)
	};

	// Stateless controls and format-string templates.
	(@acc [$($output:tt)*] cursor_home $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[H",] $($rest)*)
	};
	(@acc [$($output:tt)*] cursor_forward $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[{}C",] $($rest)*)
	};
	(@acc [$($output:tt)*] cursor_up $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[{}A",] $($rest)*)
	};
	(@acc [$($output:tt)*] cursor_down $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[{}B",] $($rest)*)
	};
	(@acc [$($output:tt)*] erase_display $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[2J",] $($rest)*)
	};
	(@acc [$($output:tt)*] erase_scrollback $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[3J",] $($rest)*)
	};
	(@acc [$($output:tt)*] margins_reset $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[r",] $($rest)*)
	};
	(@acc [$($output:tt)*] screen_to_scrollback $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[22J",] $($rest)*)
	};
	(@acc [$($output:tt)*] scroll_region $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[1;{}r",] $($rest)*)
	};
	(@acc [$($output:tt)*] style_prefix $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[0",] $($rest)*)
	};
	(@acc [$($output:tt)*] style_reset $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[0m",] $($rest)*)
	};
	(@acc [$($output:tt)*] viewport_bottom $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[65535B\r",] $($rest)*)
	};
	(@acc [$($output:tt)*] viewport_newline $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[999;1H\r\n",] $($rest)*)
	};

	// Terminal-owned stacks and stateful protocol operations.
	(@acc [$($output:tt)*] kitty_keyboard_pop $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[<u",] $($rest)*)
	};
	(@acc [$($output:tt)*] progress_clear $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b]9;4;0;\x07",] $($rest)*)
	};
	(@acc [$($output:tt)*] title_pop $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[23;0t",] $($rest)*)
	};
	(@acc [$($output:tt)*] title_push $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[22;0t",] $($rest)*)
	};

	// DECSCUSR cursor styles.
	(@acc [$($output:tt)*] cursor_style_default $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[0 q",] $($rest)*)
	};
	(@acc [$($output:tt)*] cursor_style_blinking_block $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[1 q",] $($rest)*)
	};
	(@acc [$($output:tt)*] cursor_style_steady_block $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[2 q",] $($rest)*)
	};
	(@acc [$($output:tt)*] cursor_style_blinking_underline $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[3 q",] $($rest)*)
	};
	(@acc [$($output:tt)*] cursor_style_steady_underline $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[4 q",] $($rest)*)
	};
	(@acc [$($output:tt)*] cursor_style_blinking_bar $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[5 q",] $($rest)*)
	};
	(@acc [$($output:tt)*] cursor_style_steady_bar $($rest:tt)*) => {
		esc!(@acc [$($output)* "\x1b[6 q",] $($rest)*)
	};

	() => {
		""
	};
	($($part:tt)+) => {
		esc!(@acc [] $($part)+)
	};
}

pub(crate) use esc;

#[cfg(test)]
mod tests {
	#[test]
	fn mode_polarity_uses_bare_set_and_bang_reset() {
		assert_eq!(
			esc!(
				app_cursor_keys,
				!app_cursor_keys,
				?app_cursor_keys,
				app_keypad,
				!app_keypad,
				modify_other_keys,
				!modify_other_keys,
			),
			"\x1b[?1h\x1b[?1l\x1b[?1$p\x1b=\x1b>\x1b[>4;2m\x1b[>4;0m"
		);
	}

	#[test]
	fn introducers_compose_with_literal_payloads() {
		assert_eq!(esc!(osc, "11;?", bel), "\x1b]11;?\x07");
		assert_eq!(
			esc!(dcs, "tmux;", escape, apc, "G", escape, st, st),
			"\x1bPtmux;\x1b\x1b_G\x1b\x1b\\\x1b\\"
		);
	}

	#[test]
	fn protocol_queries_read_as_capabilities_not_wire_syntax() {
		assert_eq!(
			esc!(
				kitty_graphics_query,
				sixel_color_registers_query,
				cell_pixels_query,
				background_color_query,
				osc99_query,
				?insert_mode,
				?newline_mode,
				?scroll_on_output,
				?scroll_on_key_press,
				?sync_output,
				?appearance_notifications,
				?in_band_resize,
				kitty_keyboard_query,
				primary_device_attributes_query,
			),
			concat!(
				"\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\",
				"\x1b[?2;1;0S\x1b[16t\x1b]11;?\x07",
				"\x1b]99;i=omp-tui:p=?;\x1b\\",
				"\x1b[4$p\x1b[20$p\x1b[?1010$p\x1b[?1011$p\x1b[?2026$p",
				"\x1b[?2031$p\x1b[?2048$p",
				"\x1b[?u\x1b[c",
			)
		);
	}

	#[test]
	fn stateful_modes_pair_set_and_reset_byte_exactly() {
		assert_eq!(
			esc!(
				insert_mode,
				!insert_mode,
				newline_mode,
				!newline_mode,
				alt_screen,
				!alt_screen,
				cursor_visible,
				!cursor_visible,
				autowrap,
				!autowrap,
				origin,
				!origin,
				bracketed_paste,
				!bracketed_paste,
				sync_output,
				!sync_output,
				in_band_resize,
				!in_band_resize,
				appearance_notifications,
				!appearance_notifications,
				paste_events,
				!paste_events,
				scroll_on_output,
				!scroll_on_output,
				scroll_on_key_press,
				!scroll_on_key_press,
				mouse_vt200,
				!mouse_vt200,
				mouse_button_event,
				!mouse_button_event,
				mouse_any_event,
				!mouse_any_event,
				mouse_sgr,
				!mouse_sgr,
			),
			concat!(
				"\x1b[4h\x1b[4l\x1b[20h\x1b[20l",
				"\x1b[?1049h\x1b[?1049l\x1b[?25h\x1b[?25l\x1b[?7h\x1b[?7l\x1b[?6h\x1b[?6l",
				"\x1b[?2004h\x1b[?2004l\x1b[?2026h\x1b[?2026l\x1b[?2048h\x1b[?2048l",
				"\x1b[?2031h\x1b[?2031l\x1b[?5522h\x1b[?5522l",
				"\x1b[?1010h\x1b[?1010l\x1b[?1011h\x1b[?1011l",
				"\x1b[?1000h\x1b[?1000l\x1b[?1002h\x1b[?1002l\x1b[?1003h\x1b[?1003l",
				"\x1b[?1006h\x1b[?1006l",
			)
		);
	}

	#[test]
	fn stateless_controls_and_stacks_are_byte_exact() {
		assert_eq!(
			esc!(
				cursor_home,
				cursor_up,
				cursor_down,
				cursor_forward,
				erase_display,
				erase_scrollback,
				screen_to_scrollback,
				margins_reset,
				scroll_region,
				style_prefix,
				style_reset,
				viewport_bottom,
				viewport_newline,
				title_push,
				title_pop,
				kitty_keyboard_pop,
				progress_clear,
			),
			concat!(
				"\x1b[H\x1b[{}A\x1b[{}B\x1b[{}C\x1b[2J\x1b[3J\x1b[22J\x1b[r\x1b[1;{}r",
				"\x1b[0\x1b[0m\x1b[65535B\r\x1b[999;1H\r\n",
				"\x1b[22;0t\x1b[23;0t\x1b[<u\x1b]9;4;0;\x07",
			)
		);
	}

	#[test]
	fn cursor_styles_are_byte_exact() {
		assert_eq!(
			esc!(
				cursor_style_default,
				cursor_style_blinking_block,
				cursor_style_steady_block,
				cursor_style_blinking_underline,
				cursor_style_steady_underline,
				cursor_style_blinking_bar,
				cursor_style_steady_bar,
			),
			"\x1b[0 q\x1b[1 q\x1b[2 q\x1b[3 q\x1b[4 q\x1b[5 q\x1b[6 q"
		);
	}
}
