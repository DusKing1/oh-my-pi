//! Hex encode/decode contract tests.
use std::{fmt, io::Cursor};

use bytes::BytesMut;
use omp_core::hex::*;
use proptest::prelude::*;

#[test]
fn test_decode_odd_length() {
	let result = decode(b"f48656c6c6f").into_vec().unwrap();
	assert_eq!(result, &[0x0f, 0x48, 0x65, 0x6c, 0x6c, 0x6f]);
}

#[test]
fn test_roundtrip() {
	let original = b"The quick brown fox jumps over the lazy dog";
	let encoded = encode(original).into_vec();
	let decoded = decode(&encoded).into_vec().unwrap();
	assert_eq!(&decoded, original);
}

#[test]
fn test_decode_const_odd_ascii() {
	let s = decode_n(b"f48656c6c6f").unwrap();
	assert_eq!(s.as_bytes(), b"\x0fHello");
}

fn assert_fmt<I: fmt::Display + fmt::LowerHex + fmt::UpperHex>(s: &I) {
	assert_eq!(format!("{s}"), "48656c6c6f20576f726c6421");
	assert_eq!(format!("{s:x}"), "48656c6c6f20576f726c6421");
	assert_eq!(format!("{s:X}"), "48656C6C6F20576F726C6421");
	assert_eq!(format!("{s:.8}"), "4865…6421");
	assert_eq!(format!("{s:>.8}"), "…726c6421");
	assert_eq!(format!("{s:<.8}"), "48656c6c…");
	assert_eq!(format!("{s:^.8}"), "4865…6421");
	assert_eq!(format!("{s:<.8X}"), "48656C6C…");
}

#[test]
fn test_precision_truncation() {
	let data = b"Hello World!";
	let enc = encode(data);
	assert_fmt(&enc);
	let enc = encode_n(data);
	assert_fmt(&enc);
}

#[test]
fn test_bidirectional_decode() {
	let hex = b"48656c6c6f";
	let mut decoder = Decoder::new(hex.as_slice());

	assert_eq!(decoder.next().unwrap().unwrap(), 0x48); // 'H'
	assert_eq!(decoder.next_back().unwrap().unwrap(), 0x6f); // 'o'
	assert_eq!(decoder.next().unwrap().unwrap(), 0x65); // 'e'
	assert_eq!(decoder.next_back().unwrap().unwrap(), 0x6c); // 'l'
	assert_eq!(decoder.next().unwrap().unwrap(), 0x6c); // 'l'
	assert!(decoder.next().is_none());
}

#[test]
fn test_bidirectional_encode() {
	let data = b"ABC";
	let mut encoder = Encoder::new(data);

	assert_eq!(encoder.next().unwrap(), b'4'); // High nibble of 'A'
	assert_eq!(encoder.next_back().unwrap(), b'3'); // Low nibble of 'C'
	assert_eq!(encoder.next().unwrap(), b'1'); // Low nibble of 'A'
	assert_eq!(encoder.next_back().unwrap(), b'4'); // High nibble of 'C'
	assert_eq!(encoder.next().unwrap(), b'4'); // High nibble of 'B'
	assert_eq!(encoder.next_back().unwrap(), b'2'); // Low nibble of 'B'
	assert!(encoder.next().is_none());
}

#[test]
fn test_decode_mut_odd() {
	let mut dst = [0u8; 10];
	let n = decode_mut(b"f48656c6c6f", &mut dst).unwrap();
	assert_eq!(n, 6);
	assert_eq!(&dst[..n], b"\x0fHello");
}

#[test]
fn test_skip_0x_slice() {
	// With 0x prefix
	assert_eq!(skip_0x(b"0x48656c6c6f"), b"48656c6c6f");
	assert_eq!(skip_0x(b"0X48656c6c6f"), b"48656c6c6f");

	// Without prefix
	assert_eq!(skip_0x(b"48656c6c6f"), b"48656c6c6f");

	// Just prefix
	assert_eq!(skip_0x(b"0x"), b"");

	// Only '0'
	assert_eq!(skip_0x(b"0"), b"0");

	// Empty
	assert_eq!(skip_0x(b""), b"");
}

#[test]
fn test_skip_0x_decoder() {
	// With 0x prefix
	let hex = b"0x48656c6c6f";
	let result = Decoder::new(hex).skip_0x().into_vec().unwrap();
	assert_eq!(result, b"Hello");

	// With 0X prefix
	let hex = b"0X48656c6c6f";
	let result = Decoder::new(hex).skip_0x().into_vec().unwrap();
	assert_eq!(result, b"Hello");

	// Without prefix
	let hex = b"48656c6c6f";
	let result = Decoder::new(hex).skip_0x().into_vec().unwrap();
	assert_eq!(result, b"Hello");

	// Odd length with 0x prefix
	let hex = b"0x148656c6c6f";
	let result = Decoder::new(hex).skip_0x().into_vec().unwrap();
	assert_eq!(result, b"\x01Hello");
}

#[test]
fn test_skip_leading_zeros_slice() {
	// With leading zeros
	assert_eq!(skip_leading_zeros(b"000048656c6c6f"), b"48656c6c6f");

	// All zeros
	assert_eq!(skip_leading_zeros(b"0000"), b"");

	// No leading zeros
	assert_eq!(skip_leading_zeros(b"48656c6c6f"), b"48656c6c6f");

	// Single zero
	assert_eq!(skip_leading_zeros(b"0"), b"");

	// Empty
	assert_eq!(skip_leading_zeros(b""), b"");
}

#[test]
fn test_skip_leading_zeros_decoder() {
	// With leading zeros (pairs)
	let hex = b"000048656c6c6f";
	let result = Decoder::new(hex).skip_leading_zeros().into_vec().unwrap();
	assert_eq!(result, b"Hello");

	// With one pair of zeros
	let hex = b"0048656c6c6f";
	let result = Decoder::new(hex).skip_leading_zeros().into_vec().unwrap();
	assert_eq!(result, b"Hello");

	// No leading zeros
	let hex = b"48656c6c6f";
	let result = Decoder::new(hex).skip_leading_zeros().into_vec().unwrap();
	assert_eq!(result, b"Hello");

	// Odd-length with single leading zero
	let hex = b"048656c6c6f";
	let result = Decoder::new(hex).skip_leading_zeros().into_vec().unwrap();
	assert_eq!(result, b"Hello");
}

#[test]
fn test_combined_prefix_skip() {
	// 0x prefix with leading zeros
	let hex = b"0x000048656c6c6f";
	let result = Decoder::new(hex)
		.skip_0x()
		.skip_leading_zeros()
		.into_vec()
		.unwrap();
	assert_eq!(result, b"Hello");
}

#[test]
fn test_decoder_write_into() {
	let hex = b"48656c6c6f";
	let mut writer = Cursor::new(Vec::new());
	let n = Decoder::new(hex).write_into(&mut writer).unwrap();
	assert_eq!(n, 5);
	assert_eq!(writer.into_inner(), b"Hello");

	// Large input (test buffering - exactly 514 bytes to test buffer boundary)
	let expected = vec![0x42u8; 514]; // 514 bytes
	let hex_vec: Vec<u8> = Encoder::new(&expected).collect();

	let mut writer = Cursor::new(Vec::new());
	let n = Decoder::new(hex_vec.as_slice())
		.write_into(&mut writer)
		.unwrap();
	assert_eq!(n, 514);
	assert_eq!(writer.into_inner(), expected);
}

// ============================================================================
// DECODER ERROR PATHS & EDGE CASES
// ============================================================================

#[test]
fn test_decode_invalid_char_at_start() {
	let result = decode(b"g48656c6c6f").into_vec();
	assert!(matches!(result, Err(DecodeError::InvalidCharacter(b'g'))));
}

#[test]
fn test_decode_invalid_char_at_end() {
	let result = decode(b"48656c6c6g").into_vec();
	assert!(matches!(result, Err(DecodeError::InvalidCharacter(b'g'))));
}

#[test]
fn test_decode_invalid_char_in_middle() {
	let result = decode(b"4865*c6c6f").into_vec();
	assert!(matches!(result, Err(DecodeError::InvalidCharacter(b'*'))));
}

#[test]
fn test_decode_all_invalid_chars() {
	for &ch in b"ghijklmnopqrstuvwxyzGHIJKLMNOPQRSTUVWXYZ@#$%^&*()[]{}!~`" {
		let input = [ch, ch];
		let result = decode(&input[..]).into_vec();
		assert!(
			matches!(result, Err(DecodeError::InvalidCharacter(c)) if c == ch),
			"Expected InvalidCharacter({ch:?}) but got {result:?}"
		);
	}
}

#[test]
fn test_decode_empty_input() {
	let result = decode(b"").into_vec().unwrap();
	assert_eq!(result, b"");
}

#[test]
fn test_decode_single_nibble() {
	let result = decode(b"f").into_vec().unwrap();
	assert_eq!(result, b"\x0f");
}

#[test]
fn test_decode_max_size_input() {
	// 10000 bytes -> 20000 hex chars
	let data = vec![0xabu8; 10000];
	let encoded = encode(&data).into_vec();
	let decoded = decode(&encoded).into_vec().unwrap();
	assert_eq!(decoded, data);
}

#[test]
fn test_decode_into_array_wrong_size() {
	let hex = b"48656c6c6f"; // 5 bytes
	let result: Result<[u8; 10]> = Decoder::new(hex).into_array();
	assert!(matches!(result, Err(DecodeError::InputTooShort)));
}

#[test]
fn test_decode_into_array_exact_size() {
	let hex = b"48656c6c6f"; // 5 bytes
	let result: Result<[u8; 5]> = Decoder::new(hex).into_array();
	assert_eq!(result.unwrap(), *b"Hello");
}

#[test]
fn test_decode_into_slice_partial() {
	let hex = b"48656c6c6f776f726c64"; // "Helloworld"
	let mut buf = [0u8; 5];
	let n = Decoder::new(hex).into_slice(&mut buf).unwrap();
	assert_eq!(n, 5);
	assert_eq!(&buf, b"Hello");
}

#[test]
fn test_decode_into_buf() {
	let hex = b"48656c6c6f";
	let buf = BytesMut::with_capacity(10);
	let result = Decoder::new(hex).into_buf(buf).unwrap();
	assert_eq!(&result[..], b"Hello");
}

#[test]
fn test_decode_extend_into() {
	let hex = b"48656c6c6f";
	let mut vec = Vec::new();
	let n = Decoder::new(hex).extend_into(&mut vec).unwrap();
	assert_eq!(n, 5);
	assert_eq!(vec, b"Hello");
}

#[test]
fn test_decode_invalid_char_in_odd_position() {
	let result = decode(b"4g656c6c6f").into_vec();
	assert!(matches!(result, Err(DecodeError::InvalidCharacter(b'g'))));
}

#[test]
fn test_decode_invalid_char_in_even_position() {
	let result = decode(b"48g56c6c6f").into_vec();
	assert!(matches!(result, Err(DecodeError::InvalidCharacter(b'g'))));
}

#[test]
fn test_decode_case_insensitive() {
	let lower = decode(b"abcdef").into_vec().unwrap();
	let upper = decode(b"ABCDEF").into_vec().unwrap();
	let mixed = decode(b"AbCdEf").into_vec().unwrap();
	assert_eq!(lower, upper);
	assert_eq!(lower, mixed);
}

// ============================================================================
// ENCODER CHARSET & FORMATTING
// ============================================================================

#[test]
fn test_encode_lowercase_default() {
	let result = encode(b"\xab\xcd\xef").into_string();
	assert_eq!(result, "abcdef");
	assert!(!result.contains(char::is_uppercase));
}

#[test]
fn test_encode_uppercase_explicit() {
	let result = Encoder::new(b"\xab\xcd\xef").upper().into_string();
	assert_eq!(result, "ABCDEF");
	assert!(!result.contains(char::is_lowercase));
}

#[test]
fn test_encode_charset_switching() {
	let data = b"\xab\xcd\xef";
	let lower = Encoder::new(data).lower().into_string();
	let upper = Encoder::new(data).upper().into_string();
	assert_eq!(lower.to_uppercase(), upper);
	assert_eq!(upper.to_lowercase(), lower);
}

#[test]
fn test_encode_all_bytes() {
	let data: Vec<u8> = (0..=255).collect();
	let encoded = encode(&data).into_string();
	assert_eq!(encoded.len(), 512); // 256 bytes * 2 hex chars

	// Verify all hex chars are lowercase
	for ch in encoded.chars() {
		assert!(ch.is_ascii_hexdigit());
		if ch.is_alphabetic() {
			assert!(ch.is_lowercase());
		}
	}
}

#[test]
fn test_encode_upper_all_bytes() {
	let data: Vec<u8> = (0..=255).collect();
	let encoded = Encoder::new(&data).upper().into_string();
	assert_eq!(encoded.len(), 512);

	// Verify all alpha chars are uppercase
	for ch in encoded.chars() {
		assert!(ch.is_ascii_hexdigit());
		if ch.is_alphabetic() {
			assert!(ch.is_uppercase());
		}
	}
}

#[test]
fn test_encode_nibble_boundaries() {
	// Test nibbles 0-15
	for i in 0u8..16 {
		let lower = LOWER.encode_nibble(i);
		let upper = UPPER.encode_nibble(i);

		if i < 10 {
			assert_eq!(lower, b'0' + i);
			assert_eq!(upper, b'0' + i);
		} else {
			assert_eq!(lower, b'a' + (i - 10));
			assert_eq!(upper, b'A' + (i - 10));
		}
	}
}

#[test]
fn test_encode_byte_consistency() {
	for byte in 0u8..=255 {
		let lower_pair = LOWER.encode_byte(byte);
		let upper_pair = UPPER.encode_byte(byte);

		// High nibble
		let high = byte >> 4;
		assert_eq!(lower_pair[0], LOWER.encode_nibble(high));
		assert_eq!(upper_pair[0], UPPER.encode_nibble(high));

		// Low nibble
		let low = byte & 0x0f;
		assert_eq!(lower_pair[1], LOWER.encode_nibble(low));
		assert_eq!(upper_pair[1], UPPER.encode_nibble(low));
	}
}

#[test]
fn test_encoder_into_bytes() {
	let data = b"Hello";
	let bytes = Encoder::new(data).into_bytes();
	assert_eq!(&bytes[..], b"48656c6c6f");
}

#[test]
fn test_encoder_into_buf() {
	let data = b"Hello";
	let buf = BytesMut::with_capacity(20);
	let result = Encoder::new(data).into_buf(buf);
	assert_eq!(&result[..], b"48656c6c6f");
}

#[test]
fn test_encoder_extend_into() {
	let data = b"Hello";
	let mut vec = Vec::new();
	Encoder::new(data).extend_into(&mut vec);
	assert_eq!(vec, b"48656c6c6f");
}

#[test]
fn test_encoder_format_into() {
	let data = b"Hello";
	let mut s = String::new();
	Encoder::new(data).format_into(&mut s).unwrap();
	assert_eq!(s, "48656c6c6f");
}

#[test]
fn test_encoder_write_into_upper() {
	let data = b"Hello";
	let mut writer = Cursor::new(Vec::new());
	let n = Encoder::new(data).upper().write_into(&mut writer).unwrap();
	assert_eq!(n, 10);
	assert_eq!(writer.into_inner(), b"48656C6C6F");
}

#[test]
fn test_encoder_bidirectional_exhaustion() {
	let data = b"AB";
	let mut enc = Encoder::new(data);

	// Consume from both ends until exhausted
	assert_eq!(enc.next().unwrap(), b'4'); // A high
	assert_eq!(enc.next_back().unwrap(), b'2'); // B low
	assert_eq!(enc.next().unwrap(), b'1'); // A low
	assert_eq!(enc.next_back().unwrap(), b'4'); // B high
	assert_eq!(enc.next(), None);
	assert_eq!(enc.next_back(), None);

	// FusedIterator: stays None
	assert_eq!(enc.next(), None);
	assert_eq!(enc.next_back(), None);
}

#[test]
fn test_decoder_bidirectional_odd_length() {
	let hex = b"f48"; // 3 chars -> [0x0f, 0x48]
	let mut dec = Decoder::new(hex);

	assert_eq!(dec.next().unwrap().unwrap(), 0x0f); // Single nibble
	assert_eq!(dec.next_back().unwrap().unwrap(), 0x48); // Pair
	assert_eq!(dec.next(), None);
}

// ============================================================================
// PARSE HELPERS
// ============================================================================

#[test]
fn test_parse_nibble_all_valid() {
	// '0'-'9'
	for (i, ch) in (b'0'..=b'9').enumerate() {
		assert_eq!(parse_nibble(ch), Some(i as u8));
	}
	// 'a'-'f'
	for (i, ch) in (b'a'..=b'f').enumerate() {
		assert_eq!(parse_nibble(ch), Some(10 + i as u8));
	}
	// 'A'-'F'
	for (i, ch) in (b'A'..=b'F').enumerate() {
		assert_eq!(parse_nibble(ch), Some(10 + i as u8));
	}
}

#[test]
fn test_parse_nibble_invalid() {
	for &ch in b"ghijklmnopqrstuvwxyzGHIJKLMNOPQRSTUVWXYZ@#$%^&*()" {
		assert_eq!(parse_nibble(ch), None, "Expected None for {ch:?}");
	}
}

#[test]
fn test_parse_byte_all_combinations() {
	// Test a sample of byte values
	for high in [0, 5, 9, 10, 15] {
		for low in [0, 5, 9, 10, 15] {
			let expected = (high << 4) | low;
			let h_ch = if high < 10 {
				b'0' + high
			} else {
				b'a' + (high - 10)
			};
			let l_ch = if low < 10 {
				b'0' + low
			} else {
				b'a' + (low - 10)
			};
			let result = parse_byte([h_ch, l_ch]).unwrap();
			assert_eq!(result, expected);
		}
	}
}

#[test]
fn test_parse_byte_invalid_high() {
	let result = parse_byte(*b"g0");
	assert!(matches!(result, Err(DecodeError::InvalidCharacter(b'g'))));
}

#[test]
fn test_parse_byte_invalid_low() {
	let result = parse_byte(*b"0g");
	assert!(matches!(result, Err(DecodeError::InvalidCharacter(b'g'))));
}

#[test]
fn test_parse_byte_both_invalid() {
	// Should return first invalid char
	let result = parse_byte(*b"zy");
	assert!(matches!(result, Err(DecodeError::InvalidCharacter(b'z'))));
}

// ============================================================================
// BUFFER OPERATIONS
// ============================================================================

#[test]
fn test_encode_mut_small_dst() {
	let src = b"Hello";
	let mut dst = [0u8; 5]; // Too small (need 10)
	let n = encode_mut(src, &mut dst);
	assert_eq!(n, 4); // Only 2 bytes encoded (4 hex chars)
	assert_eq!(&dst[..4], b"4865");
}

#[test]
fn test_encode_mut_exact_dst() {
	let src = b"Hello";
	let mut dst = [0u8; 10];
	let n = encode_mut(src, &mut dst);
	assert_eq!(n, 10);
	assert_eq!(&dst, b"48656c6c6f");
}

#[test]
fn test_encode_mut_large_dst() {
	let src = b"Hello";
	let mut dst = [0u8; 20];
	let n = encode_mut(src, &mut dst);
	assert_eq!(n, 10);
	assert_eq!(&dst[..10], b"48656c6c6f");
	// Rest should be untouched (zeros)
	assert_eq!(&dst[10..], &[0u8; 10]);
}

#[test]
fn test_decode_mut_small_dst() {
	let src = b"48656c6c6f"; // "Hello"
	let mut dst = [0u8; 3]; // Too small
	let n = decode_mut(src, &mut dst).unwrap();
	assert_eq!(n, 3);
	assert_eq!(&dst, b"Hel");
}

#[test]
fn test_decode_mut_exact_dst() {
	let src = b"48656c6c6f";
	let mut dst = [0u8; 5];
	let n = decode_mut(src, &mut dst).unwrap();
	assert_eq!(n, 5);
	assert_eq!(&dst, b"Hello");
}

#[test]
fn test_decode_mut_large_dst() {
	let src = b"48656c6c6f";
	let mut dst = [0u8; 10];
	let n = decode_mut(src, &mut dst).unwrap();
	assert_eq!(n, 5);
	assert_eq!(&dst[..5], b"Hello");
	assert_eq!(&dst[5..], &[0u8; 5]);
}

// ============================================================================
// EXACTSIZEITERATOR & FUSEDITERATOR
// ============================================================================

#[test]
fn test_encoder_exact_size() {
	let data = b"Hello";
	let enc = Encoder::new(data);
	assert_eq!(enc.len(), 10);
	assert_eq!(enc.size_hint(), (10, Some(10)));

	assert_eq!(enc.count(), 10);
}

#[test]
fn test_decoder_exact_size() {
	let hex = b"48656c6c6f";
	let dec = Decoder::new(hex);
	assert_eq!(dec.len(), 5);
	assert_eq!(dec.size_hint(), (5, Some(5)));

	let collected: Result<Vec<u8>> = dec.collect();
	assert_eq!(collected.unwrap().len(), 5);
}

#[test]
fn test_decoder_exact_size_odd() {
	let hex = b"f48656c6c6f"; // Odd length
	let dec = Decoder::new(hex);
	assert_eq!(dec.len(), 6);
}

#[test]
fn test_encoder_fused() {
	let data = b"A";
	let mut enc = Encoder::new(data);

	// Exhaust iterator
	assert_eq!(enc.next(), Some(b'4'));
	assert_eq!(enc.next(), Some(b'1'));
	assert_eq!(enc.next(), None);

	// FusedIterator: stays None
	for _ in 0..10 {
		assert_eq!(enc.next(), None);
	}
}

#[test]
fn test_decoder_fused() {
	let hex = b"41";
	let mut dec = Decoder::new(hex);

	assert_eq!(dec.next().unwrap().unwrap(), 0x41);
	assert_eq!(dec.next(), None);

	// FusedIterator: stays None
	for _ in 0..10 {
		assert_eq!(dec.next(), None);
	}
}

// ============================================================================
// COMPARISON TRAITS
// ============================================================================

#[test]
fn test_encoder_partial_eq_slice() {
	let data = b"Hello";
	let enc = Encoder::new(data);
	assert_eq!(enc, b"48656c6c6f"[..]);
	assert_ne!(enc, b"ffffff"[..]);
}

#[test]
fn test_encoder_partial_eq_str() {
	let data = b"Hello";
	let enc = Encoder::new(data);
	let enc_str = enc.clone().into_string();
	assert_eq!(enc_str, "48656c6c6f");
	let enc2 = Encoder::new(b"Hello");
	let other_str = Encoder::new(&[0xff, 0xff, 0xff]).into_string();
	assert_ne!(enc2.into_string(), other_str);
}

#[test]
fn test_decoder_partial_eq_slice() {
	let hex = b"48656c6c6f";
	let dec = Decoder::new(hex);
	assert_eq!(dec, b"Hello"[..]);
	assert_ne!(dec, b"World"[..]);
}

#[test]
fn test_encoder_ord() {
	let a = Encoder::new(b"A");
	let b = Encoder::new(b"B");
	use std::cmp::Ordering;
	// Compare using Ord trait impl
	let res = Ord::cmp(&a, &b);
	assert_eq!(res, Ordering::Less);
}

#[test]
fn test_decoder_ord() {
	let a = Decoder::new(b"41"); // 'A'
	let b = Decoder::new(b"42"); // 'B'
	// Decoders only have PartialOrd via comparison with Result items
	let a_vec = a.into_vec().unwrap();
	let b_vec = b.into_vec().unwrap();
	assert!(a_vec < b_vec);
}

// ============================================================================
// SERDE
// ============================================================================

#[test]
fn test_encoder_serde_json() {
	use serde_json as json;

	let data = b"Hello";
	let enc = Encoder::new(data);
	let json_str = json::to_string(&enc).unwrap();
	assert_eq!(json_str, r#""48656c6c6f""#);
}

// ============================================================================
// PROPTEST
// ============================================================================

proptest! {
	#[test]
	fn proptest_hex_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1000)) {
		let encoded = encode(&data).into_vec();
		let decoded = decode(&encoded).into_vec().unwrap();
		prop_assert_eq!(decoded, data);
	}

	#[test]
	fn proptest_hex_upper_lower_same_decode(data in prop::collection::vec(any::<u8>(), 0..1000)) {
		let lower = Encoder::new(&data).lower().into_vec();
		let upper = Encoder::new(&data).upper().into_vec();
		let dec_lower = decode(&lower).into_vec().unwrap();
		let dec_upper = decode(&upper).into_vec().unwrap();
		prop_assert_eq!(&dec_lower, &data);
		prop_assert_eq!(&dec_upper, &data);
		prop_assert_eq!(&dec_lower, &dec_upper);
	}

	#[test]
	fn proptest_encoder_exact_size_accurate(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let enc = Encoder::new(&data);
		let len = enc.len();
		prop_assert_eq!(len, encode_len(data.len()));
		prop_assert_eq!(enc.count(), len);
	}

	#[test]
	fn proptest_decoder_exact_size_accurate(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let encoded = encode(&data).into_vec();
		let dec = Decoder::new(&encoded);
		let len = dec.len();
		prop_assert_eq!(dec.count(), len);
	}

	#[test]
	fn proptest_encode_mut_consistency(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let mut dst1 = vec![0u8; encode_len(data.len())];
		let mut dst2 = vec![0u8; encode_len(data.len())];

		let n1 = LOWER.encode_mut(&data, &mut dst1);
		let n2 = encode_mut(&data, &mut dst2);

		prop_assert_eq!(n1, n2);
		prop_assert_eq!(&dst1[..n1], &dst2[..n2]);
	}

	#[test]
	fn proptest_decode_mut_consistency(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let encoded = encode(&data).into_vec();
		let mut dst = vec![0u8; decode_len(encoded.len())];
		let n = decode_mut(&encoded, &mut dst).unwrap();
		prop_assert_eq!(&dst[..n], data.as_slice());
	}
}

// ===== Partial-iteration collector regressions =====
// The Encoder collectors must stay correct (and sound) when `next`/`next_back`
// have consumed nibbles, leaving pending `low`/`high` state.

#[test]
fn test_encoder_into_vec_after_partial_front_iteration() {
	let mut enc = encode(&[0xab, 0xcd, 0xef]);
	assert_eq!(enc.next(), Some(b'a'));
	assert_eq!(enc.into_vec(), b"bcdef");
}

#[test]
fn test_encoder_into_vec_after_partial_back_iteration() {
	// Pending high nibble + non-empty src: the pair block must be published
	// before the high nibble is appended.
	let mut enc = encode(&[0xab, 0xcd, 0xef]);
	assert_eq!(enc.next_back(), Some(b'f'));
	assert_eq!(enc.into_vec(), b"abcde");
}

#[test]
fn test_encoder_into_vec_after_both_end_iteration() {
	let mut enc = encode(&[0xab, 0xcd]);
	assert_eq!(enc.next(), Some(b'a'));
	assert_eq!(enc.next_back(), Some(b'd'));
	assert_eq!(enc.into_vec(), b"bc");
}

#[test]
fn test_encoder_write_into_after_partial_back_iteration() {
	// The pending high nibble lands mid-batch and must be counted in the
	// flushed slice, not silently dropped.
	let mut enc = encode(&[0xab, 0xcd, 0xef]);
	assert_eq!(enc.next_back(), Some(b'f'));
	let mut out = Vec::new();
	let n = enc.write_into(&mut out).unwrap();
	assert_eq!(out, b"abcde");
	assert_eq!(n, 5);
}

#[test]
fn test_encoder_write_into_after_partial_back_iteration_large() {
	// Cross the 64-pair internal batch boundary with a pending high nibble.
	let src: Vec<u8> = (0..=200u8).collect();
	let full = encode(&src).into_vec();
	let mut enc = encode(&src);
	assert_eq!(enc.next_back(), Some(*full.last().unwrap()));
	let mut out = Vec::new();
	let n = enc.write_into(&mut out).unwrap();
	assert_eq!(out, full[..full.len() - 1]);
	assert_eq!(n, full.len() - 1);
}
