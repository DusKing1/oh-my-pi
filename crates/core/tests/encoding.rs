//! Round-trip and edge-case tests for the base16/base32/base64 encoders.
use omp_core::encoding::*;
use proptest::{prelude::*, proptest};

// ============================================================================
// BASE64 TESTS
// ============================================================================

#[test]
fn test_base64_all_chars() {
	// Test that all 64 characters can be encoded/decoded
	let data = b"\x00\x10 0@P`p\x83\x10\xa4\x38\xc7\x1c\xeb\x39\xff";
	let encoded = base64::encode(data).into_string();
	let decoded = base64::decode(encoded.as_bytes()).into_vec().unwrap();
	assert_eq!(&decoded, data);
}

#[test]
fn test_base64_invalid_char() {
	let result = base64::decode(b"SGVs*G8=").into_vec();
	assert!(matches!(result, Err(DecodeError::InvalidCharacter(b'*'))));
}

#[test]
fn test_base64_invalid_length() {
	// But with padding char in wrong position
	let result = base64::decode(b"SG=sbG8=").into_vec();
	assert!(result.is_err());
}

#[test]
fn test_base32_invalid_char() {
	let result = base32::decode(b"JBSWY*DP").into_vec();
	assert!(matches!(result, Err(DecodeError::InvalidCharacter(b'*'))));
}

#[test]
fn test_base32_invalid_padding() {
	// Invalid padding count (2, 5, 7, 8 are invalid)
	let result = base32::decode(b"IE==").into_vec();
	assert!(matches!(result, Err(DecodeError::InvalidLength)));
}

// ============================================================================
// ENCODER/DECODER TESTS
// ============================================================================

#[test]
fn test_encoder_write_into() {
	use std::io::Cursor;

	let data = b"Hello World";
	let mut writer = Cursor::new(Vec::new());
	let n = base64::encode(data).write_into(&mut writer).unwrap();
	assert_eq!(n, 16); // "SGVsbG8gV29ybGQ=" is 16 bytes
	assert_eq!(writer.into_inner(), b"SGVsbG8gV29ybGQ=");
}

#[test]
fn test_decoder_write_into() {
	use std::io::Cursor;

	let encoded = b"SGVsbG8gV29ybGQ=";
	let mut writer = Cursor::new(Vec::new());
	let n = base64::decode(encoded).write_into(&mut writer).unwrap();
	assert_eq!(n, 11);
	assert_eq!(writer.into_inner(), b"Hello World");
}

#[test]
fn test_encoder_write_into_large() {
	use std::io::Cursor;

	// Test buffering with data larger than internal buffer (>512 bytes)
	let data = vec![0x42u8; 600];
	let mut writer = Cursor::new(Vec::new());
	let n = base64::encode(&data).write_into(&mut writer).unwrap();

	let expected_len = base64::encode_len(600);
	assert_eq!(n, expected_len);

	let result = writer.into_inner();
	let decoded = base64::decode(&result).into_vec().unwrap();
	assert_eq!(decoded, data);
}

// ============================================================================
// EDGE CASES AND ERROR HANDLING
// ============================================================================

#[test]
fn test_large_input() {
	let data = vec![0x42u8; 10000];
	let encoded = base64::encode(&data).into_string();
	let decoded = base64::decode(encoded.as_bytes()).into_vec().unwrap();
	assert_eq!(decoded, data);
}

#[test]
fn test_all_bytes_base64() {
	let data: Vec<u8> = (0..=255).collect();
	let encoded = base64::encode(&data).into_string();
	let decoded = base64::decode(encoded.as_bytes()).into_vec().unwrap();
	assert_eq!(decoded, data);
}

#[test]
fn test_all_bytes_base32() {
	let data: Vec<u8> = (0..=255).collect();
	let encoded = base32::encode(&data).into_string();
	let decoded = base32::decode(encoded.as_bytes()).into_vec().unwrap();
	assert_eq!(decoded, data);
}

#[test]
fn test_base64_padding_edge_cases() {
	// Test all padding scenarios
	assert_eq!(base64::encode(b"").into_string(), "");
	assert_eq!(base64::encode(b"f").into_string(), "Zg==");
	assert_eq!(base64::encode(b"fo").into_string(), "Zm8=");
	assert_eq!(base64::encode(b"foo").into_string(), "Zm9v");
	assert_eq!(base64::encode(b"foob").into_string(), "Zm9vYg==");
	assert_eq!(base64::encode(b"fooba").into_string(), "Zm9vYmE=");
	assert_eq!(base64::encode(b"foobar").into_string(), "Zm9vYmFy");
}

#[test]
fn test_base32_padding_edge_cases() {
	assert_eq!(base32::encode(b"").into_string(), "");
	assert_eq!(base32::encode(b"f").into_string(), "MY======");
	assert_eq!(base32::encode(b"fo").into_string(), "MZXQ====");
	assert_eq!(base32::encode(b"foo").into_string(), "MZXW6===");
	assert_eq!(base32::encode(b"foob").into_string(), "MZXW6YQ=");
	assert_eq!(base32::encode(b"fooba").into_string(), "MZXW6YTB");
	assert_eq!(base32::encode(b"foobar").into_string(), "MZXW6YTBOI======");
}

#[test]
fn test_base64_decode_error_propagation() {
	let result: Result<Vec<u8>> = base64::decode(b"!@#$").collect();
	assert!(result.is_err());
}

#[test]
fn test_base32_decode_error_propagation() {
	let result: Result<Vec<u8>> = base32::decode(b"!@#$====").collect();
	assert!(result.is_err());
}

// ============================================================================
// DICTIONARY TESTS
// ============================================================================

#[test]
fn test_base64_url_special_chars() {
	// Standard base64 uses + and /
	// URL-safe uses - and _
	let data = b"\xfb\xff"; // Should produce +/ in standard, -_ in URL-safe

	let std_encoded = base64::encode(data).into_string();
	let url_encoded = base64_url::encode(data).into_string();

	assert!(std_encoded.contains('+') || std_encoded.contains('/'));
	assert!(url_encoded.contains('-') || url_encoded.contains('_'));
	assert!(!url_encoded.contains('+'));
	assert!(!url_encoded.contains('/'));
}

#[test]
fn test_base32_hex_variant() {
	let data = b"Hello";
	let encoded = base32_hex::encode(data).into_string();
	assert_eq!(encoded, "91IMOR3F");
	let decoded = base32_hex::decode(encoded.as_bytes()).into_vec().unwrap();
	assert_eq!(decoded, b"Hello");
}

// ============================================================================
// PROPTEST ROUNDTRIPS
// ============================================================================

proptest! {
	fn proptest_base64_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1000)) {
		let encoded = base64::encode(&data).into_string();
		let decoded = base64::decode(encoded.as_bytes()).into_vec().unwrap();
		prop_assert_eq!(decoded, data);
	}

	fn proptest_base64_raw_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1000)) {
		let encoded = base64::encode_raw(&data).into_string();
		let decoded = base64::decode_raw(encoded.as_bytes()).into_vec().unwrap();
		prop_assert_eq!(decoded, data);
	}

	fn proptest_base64_url_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1000)) {
		let encoded = base64_url::encode(&data).into_string();
		let decoded = base64_url::decode(encoded.as_bytes()).into_vec().unwrap();
		prop_assert_eq!(decoded, data);
	}

	fn proptest_base64_url_raw_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1000)) {
		let encoded = base64_url::encode_raw(&data).into_string();
		let decoded = base64_url::decode_raw(encoded.as_bytes()).into_vec().unwrap();
		prop_assert_eq!(decoded, data);
	}

	fn proptest_base32_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1000)) {
		let encoded = base32::encode(&data).into_string();
		let decoded = base32::decode(encoded.as_bytes()).into_vec().unwrap();
		prop_assert_eq!(decoded, data);
	}

	fn proptest_base32_raw_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1000)) {
		let encoded = base32::encode_raw(&data).into_string();
		let decoded = base32::decode_raw(encoded.as_bytes()).into_vec().unwrap();
		prop_assert_eq!(decoded, data);
	}

	fn proptest_base32_hex_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1000)) {
		let encoded = base32_hex::encode(&data).into_string();
		let decoded = base32_hex::decode(encoded.as_bytes()).into_vec().unwrap();
		prop_assert_eq!(decoded, data);
	}

	fn proptest_base32_hex_raw_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1000)) {
		let encoded = base32_hex::encode_raw(&data).into_string();
		let decoded = base32_hex::decode_raw(encoded.as_bytes()).into_vec().unwrap();
		prop_assert_eq!(decoded, data);
	}

	fn proptest_base64_encode_opt_matches_const(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let mut dst1 = vec![0u8; base64::encode_len(data.len())];
		let mut dst2 = vec![0u8; base64::encode_len(data.len())];

		let n1 = Encoding::<64>::encode_opt(&base64::STD, &data, &mut dst1);
		let n2 = Encoding::<64>::encode_const(&base64::STD, &data, &mut dst2);

		prop_assert_eq!(n1, n2);
		prop_assert_eq!(&dst1[..n1], &dst2[..n2]);
	}

	fn proptest_base64_decode_opt_matches_const(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let encoded = base64::encode(&data).into_vec();

		let mut dst1 = vec![0u8; data.len()];
		let mut dst2 = vec![0u8; data.len()];

		let n1 = Encoding::<64>::decode_opt(&base64::STD, &encoded, &mut dst1).unwrap();
		let n2 = Encoding::<64>::decode_const(&base64::STD, &encoded, &mut dst2).unwrap();

		prop_assert_eq!(n1, n2);
		prop_assert_eq!(&dst1[..n1], &dst2[..n2]);
		prop_assert_eq!(&dst1[..n1], data.as_slice());
	}

	fn proptest_base32_encode_opt_matches_const(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let mut dst1 = vec![0u8; base32::encode_len(data.len())];
		let mut dst2 = vec![0u8; base32::encode_len(data.len())];

		let n1 = Encoding::<32>::encode_opt(&base32::STD, &data, &mut dst1);
		let n2 = Encoding::<32>::encode_const(&base32::STD, &data, &mut dst2);

		prop_assert_eq!(n1, n2);
		prop_assert_eq!(&dst1[..n1], &dst2[..n2]);
	}

	fn proptest_base32_decode_opt_matches_const(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let encoded = base32::encode(&data).into_vec();

		let mut dst1 = vec![0u8; data.len()];
		let mut dst2 = vec![0u8; data.len()];

		let n1 = Encoding::<32>::decode_opt(&base32::STD, &encoded, &mut dst1).unwrap();
		let n2 = Encoding::<32>::decode_const(&base32::STD, &encoded, &mut dst2).unwrap();

		prop_assert_eq!(n1, n2);
		prop_assert_eq!(&dst1[..n1], &dst2[..n2]);
		prop_assert_eq!(&dst1[..n1], data.as_slice());
	}

	fn proptest_base64_encode_raw_opt_matches_const(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let mut dst1 = vec![0u8; base64::encode_raw_len(data.len())];
		let mut dst2 = vec![0u8; base64::encode_raw_len(data.len())];

		let n1 = Encoding::<64>::encode_opt(&base64::RAW, &data, &mut dst1);
		let n2 = Encoding::<64>::encode_const(&base64::RAW, &data, &mut dst2);

		prop_assert_eq!(n1, n2);
		prop_assert_eq!(&dst1[..n1], &dst2[..n2]);
	}

	fn proptest_base64_decode_raw_opt_matches_const(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let encoded = base64::encode_raw(&data).into_vec();

		let mut dst1 = vec![0u8; data.len()];
		let mut dst2 = vec![0u8; data.len()];

		let n1 = Encoding::<64>::decode_opt(&base64::RAW, &encoded, &mut dst1).unwrap();
		let n2 = Encoding::<64>::decode_const(&base64::RAW, &encoded, &mut dst2).unwrap();

		prop_assert_eq!(n1, n2);
		prop_assert_eq!(&dst1[..n1], &dst2[..n2]);
		prop_assert_eq!(&dst1[..n1], data.as_slice());
	}

	fn proptest_base32_encode_raw_opt_matches_const(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let mut dst1 = vec![0u8; base32::encode_raw_len(data.len())];
		let mut dst2 = vec![0u8; base32::encode_raw_len(data.len())];

		let n1 = Encoding::<32>::encode_opt(&base32::RAW, &data, &mut dst1);
		let n2 = Encoding::<32>::encode_const(&base32::RAW, &data, &mut dst2);

		prop_assert_eq!(n1, n2);
		prop_assert_eq!(&dst1[..n1], &dst2[..n2]);
	}

	fn proptest_base32_decode_raw_opt_matches_const(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let encoded = base32::encode_raw(&data).into_vec();

		let mut dst1 = vec![0u8; data.len()];
		let mut dst2 = vec![0u8; data.len()];

		let n1 = Encoding::<32>::decode_opt(&base32::RAW, &encoded, &mut dst1).unwrap();
		let n2 = Encoding::<32>::decode_const(&base32::RAW, &encoded, &mut dst2).unwrap();

		prop_assert_eq!(n1, n2);
		prop_assert_eq!(&dst1[..n1], &dst2[..n2]);
		prop_assert_eq!(&dst1[..n1], data.as_slice());
	}
}

// ============================================================================
// ENCODING/DECODING WRITER TESTS
// ============================================================================

#[test]
fn test_encode_writer_base64_multiple_writes() {
	use std::io::Write;

	let mut output = Vec::new();
	{
		let mut writer = base64::encode_writer(&mut output);
		writer.write_all(b"Hello").unwrap();
		writer.write_all(b", ").unwrap();
		writer.write_all(b"World!").unwrap();
		writer.flush().unwrap();
	}

	let expected = base64::encode(b"Hello, World!").into_vec();
	assert_eq!(output, expected);
}

#[test]
fn test_decode_writer_base64_multiple_writes() {
	use std::io::Write;

	let encoded = base64::encode(b"Hello, World!").into_vec();

	let mut output = Vec::new();
	{
		let mut writer = base64::decode_writer(&mut output);

		// Write in chunks
		let mid = encoded.len() / 2;
		writer.write_all(&encoded[..mid]).unwrap();
		writer.write_all(&encoded[mid..]).unwrap();
		writer.flush().unwrap();
	}

	assert_eq!(output, b"Hello, World!");
}

#[test]
fn test_encode_writer_base64_large_data() {
	use std::io::Write;

	let data = vec![0x42u8; 10000];

	let mut output = Vec::new();
	{
		let mut writer = base64::encode_writer(&mut output);
		writer.write_all(&data).unwrap();
		writer.flush().unwrap();
	}

	let expected = base64::encode(&data).into_vec();
	assert_eq!(output, expected);
}

#[test]
fn test_decode_writer_base64_large_data() {
	use std::io::Write;

	let data = vec![0x42u8; 10000];
	let encoded = base64::encode(&data).into_vec();

	let mut output = Vec::new();
	{
		let mut writer = base64::decode_writer(&mut output);
		writer.write_all(&encoded).unwrap();
		writer.flush().unwrap();
	}

	assert_eq!(output, data);
}

#[test]
fn test_encode_writer_base64_auto_flush_on_drop() {
	use std::io::Write;

	let mut output = Vec::new();
	{
		let mut writer = base64::encode_writer(&mut output);
		writer.write_all(b"Drop test").unwrap();
		// Writer is dropped here, should auto-flush
	}

	let expected = base64::encode(b"Drop test").into_vec();
	assert_eq!(output, expected);
}

#[test]
fn test_encode_writer_base32_padding() {
	use std::io::Write;

	// Test various lengths to ensure padding is handled correctly
	for input in [&b"A"[..], b"AB", b"ABC", b"ABCD", b"ABCDE"] {
		let mut output = Vec::new();
		{
			let mut writer = base32::encode_writer(&mut output);
			writer.write_all(input).unwrap();
			writer.flush().unwrap();
		}

		let expected = base32::encode(input).into_vec();
		assert_eq!(output, expected, "Failed for input: {input:?}");
	}
}

#[test]
fn test_decode_writer_base32_incomplete_groups() {
	use std::io::Write;

	let mut output = Vec::new();
	{
		let mut writer = base32::decode_writer(&mut output);

		// Write incomplete groups across multiple calls
		writer.write_all(b"JBSW").unwrap(); // Incomplete group (4 chars)
		writer.write_all(b"Y3DP").unwrap(); // Complete the group

		writer.flush().unwrap();
	}
	assert_eq!(output, b"Hello");
}

#[test]
fn test_decode_writer_base64_invalid_data() {
	use std::io::Write;

	let mut output = Vec::new();
	let mut writer = base64::decode_writer(&mut output);

	// Write invalid base64 characters
	let result = writer.write_all(b"****");
	assert!(result.is_ok()); // Write succeeds, buffering the data

	// Flush should fail due to invalid characters
	let result = writer.flush();
	assert!(result.is_err());
}

#[test]
fn test_encode_writer_base64_roundtrip() {
	use std::io::Write;

	let original = b"The quick brown fox jumps over the lazy dog";

	// Encode
	let mut encoded = Vec::new();
	{
		let mut enc_writer = base64::encode_writer(&mut encoded);
		enc_writer.write_all(original).unwrap();
		enc_writer.flush().unwrap();
	}

	// Decode
	let mut decoded = Vec::new();
	{
		let mut dec_writer = base64::decode_writer(&mut decoded);
		dec_writer.write_all(&encoded).unwrap();
		dec_writer.flush().unwrap();
	}

	assert_eq!(decoded, original);
}

#[test]
fn test_encode_writer_base32_roundtrip() {
	use std::io::Write;

	let original = b"Base32 roundtrip test";

	// Encode
	let mut encoded = Vec::new();
	{
		let mut enc_writer = base32::encode_writer(&mut encoded);
		enc_writer.write_all(original).unwrap();
		enc_writer.flush().unwrap();
	}

	// Decode
	let mut decoded = Vec::new();
	{
		let mut dec_writer = base32::decode_writer(&mut decoded);
		dec_writer.write_all(&encoded).unwrap();
		dec_writer.flush().unwrap();
	}

	assert_eq!(decoded, original);
}

#[test]
fn test_encode_writer_base64_streaming() {
	use std::io::Write;

	let mut output = Vec::new();
	{
		let mut writer = base64::encode_writer(&mut output);

		// Write in very small chunks to test buffering
		for &byte in b"Streaming test data" {
			writer.write_all(&[byte]).unwrap();
		}
		writer.flush().unwrap();
	}

	let expected = base64::encode(b"Streaming test data").into_vec();
	assert_eq!(output, expected);
}

#[test]
fn test_decode_writer_base64_streaming() {
	use std::io::Write;

	let encoded = base64::encode(b"Streaming test data").into_vec();

	let mut output = Vec::new();
	{
		let mut writer = base64::decode_writer(&mut output);

		// Write in very small chunks
		for &byte in &encoded {
			writer.write_all(&[byte]).unwrap();
		}
		writer.flush().unwrap();
	}

	assert_eq!(output, b"Streaming test data");
}
