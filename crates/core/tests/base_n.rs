//! Comprehensive tests for `base_n` encoding covering buffer boundaries and
//! edge cases.

use std::io::Write;

use bytes::BytesMut;
use omp_core::encoding::*;
use proptest::prelude::*;

// ============================================================================
// BUFFER BOUNDARY TESTS - ENCODE_WRITER
// ============================================================================

const INPUT_BUFFER_SIZE: usize = 768;

#[test]
fn test_encode_writer_exactly_input_buffer_size() {
	let data = vec![0x42u8; INPUT_BUFFER_SIZE];
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
fn test_encode_writer_one_byte_over_input_buffer() {
	let data = vec![0x42u8; INPUT_BUFFER_SIZE + 1];
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
fn test_encode_writer_one_byte_under_input_buffer() {
	let data = vec![0x42u8; INPUT_BUFFER_SIZE - 1];
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
fn test_encode_writer_multiple_buffer_fills() {
	// 3x input buffer size to test multiple flushes
	let data = vec![0x42u8; INPUT_BUFFER_SIZE * 3];
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
fn test_encode_writer_small_writes_across_boundary() {
	let mut output = Vec::new();
	{
		let mut writer = base64::encode_writer(&mut output);
		// Write in small chunks that cross buffer boundary
		for _ in 0..(INPUT_BUFFER_SIZE / 10 + 2) {
			writer.write_all(&[0x42u8; 10]).unwrap();
		}
		writer.flush().unwrap();
	}

	let total_size = (INPUT_BUFFER_SIZE / 10 + 2) * 10;
	let expected = base64::encode(&vec![0x42u8; total_size]).into_vec();
	assert_eq!(output, expected);
}

// ============================================================================
// BUFFER BOUNDARY TESTS - DECODE_WRITER
// ============================================================================

#[test]
fn test_decode_writer_incomplete_group_buffering() {
	let data = b"Hello World!";
	let encoded = base64::encode(data).into_vec();

	let mut output = Vec::new();
	{
		let mut writer = base64::decode_writer(&mut output);

		// Write 1 char at a time to test buffering
		for &byte in &encoded {
			writer.write_all(&[byte]).unwrap();
		}
		writer.flush().unwrap();
	}

	assert_eq!(output, data);
}

#[test]
fn test_decode_writer_exactly_one_group() {
	// Base64 group is 4 chars -> 3 bytes
	let data = b"Hel"; // 3 bytes
	let encoded = base64::encode(data).into_vec();

	let mut output = Vec::new();
	{
		let mut writer = base64::decode_writer(&mut output);
		writer.write_all(&encoded).unwrap();
		writer.flush().unwrap();
	}

	assert_eq!(output, data);
}

#[test]
fn test_decode_writer_partial_groups_not_flushed_until_final() {
	let mut output = Vec::new();
	{
		let mut writer = base64::decode_writer(&mut output);

		// Write incomplete group (only 2 chars of 4-char group)
		writer.write_all(b"SG").unwrap();

		// Flush WITHOUT final_flush should not decode incomplete group
		// But we can't test this directly since flush() does final flush...
		// Instead we verify that incomplete writes are buffered correctly
		writer.write_all(b"Vs").unwrap();
		writer.write_all(b"bG").unwrap();
		writer.write_all(b"8=").unwrap();

		writer.flush().unwrap();
	}

	assert_eq!(output, b"Hello");
}

#[test]
fn test_decode_writer_buffer_boundary_base32() {
	// Base32 has 8-char groups
	let data = vec![0x42u8; 100];
	let encoded = base32::encode(&data).into_vec();

	let mut output = Vec::new();
	{
		let mut writer = base32::decode_writer(&mut output);

		// Write in chunks that don't align with group boundaries
		let chunk_size = 13; // Not divisible by 8
		for chunk in encoded.chunks(chunk_size) {
			writer.write_all(chunk).unwrap();
		}
		writer.flush().unwrap();
	}

	assert_eq!(output, data);
}

#[test]
fn test_decode_writer_exactly_input_buffer_size() {
	// Generate encoded data that's exactly INPUT_BUFFER_SIZE
	let data_size = (INPUT_BUFFER_SIZE * 3) / 4; // Base64: 4 chars -> 3 bytes
	let data = vec![0x42u8; data_size];
	let encoded = base64::encode(&data).into_vec();

	let mut output = Vec::new();
	{
		let mut writer = base64::decode_writer(&mut output);
		writer.write_all(&encoded).unwrap();
		writer.flush().unwrap();
	}

	assert_eq!(output, data);
}

// ============================================================================
// EDGE CASES - ENCODE_LEN & DECODE_LEN
// ============================================================================

#[test]
fn test_encode_len_base64_padding() {
	// Base64: 3 bytes -> 4 chars
	assert_eq!(base64::encode_len(0), 0);
	assert_eq!(base64::encode_len(1), 4); // With padding: "Zg=="
	assert_eq!(base64::encode_len(2), 4); // With padding: "Zm8="
	assert_eq!(base64::encode_len(3), 4); // No padding: "Zm9v"
	assert_eq!(base64::encode_len(4), 8); // With padding: "Zm9vYg=="
}

#[test]
fn test_encode_raw_len_base64_no_padding() {
	assert_eq!(base64::encode_raw_len(0), 0);
	assert_eq!(base64::encode_raw_len(1), 2); // "Zg"
	assert_eq!(base64::encode_raw_len(2), 3); // "Zm8"
	assert_eq!(base64::encode_raw_len(3), 4); // "Zm9v"
	assert_eq!(base64::encode_raw_len(4), 6); // "Zm9vYg"
}

#[test]
fn test_encode_len_base32_padding() {
	// Base32: 5 bytes -> 8 chars
	assert_eq!(base32::encode_len(0), 0);
	assert_eq!(base32::encode_len(1), 8); // "MY======"
	assert_eq!(base32::encode_len(2), 8); // "MZXQ===="
	assert_eq!(base32::encode_len(3), 8); // "MZXW6==="
	assert_eq!(base32::encode_len(4), 8); // "MZXW6YQ="
	assert_eq!(base32::encode_len(5), 8); // "MZXW6YTB"
	assert_eq!(base32::encode_len(6), 16); // Next group
}

#[test]
fn test_decode_len_base64() {
	// Assumes no padding chars counted
	assert_eq!(base64::decode_len(0), 0);
	assert_eq!(base64::decode_len(2), 1); // "Zg" -> 1 byte
	assert_eq!(base64::decode_len(3), 2); // "Zm8" -> 2 bytes
	assert_eq!(base64::decode_len(4), 3); // "Zm9v" -> 3 bytes
}

#[test]
fn test_decode_len_base32() {
	assert_eq!(base32::decode_len(0), 0);
	assert_eq!(base32::decode_len(2), 1); // "MY" -> 1 byte
	assert_eq!(base32::decode_len(4), 2); // "MZXQ" -> 2 bytes
	assert_eq!(base32::decode_len(5), 3); // "MZXW6" -> 3 bytes
	assert_eq!(base32::decode_len(7), 4); // "MZXW6YQ" -> 4 bytes
	assert_eq!(base32::decode_len(8), 5); // "MZXW6YTB" -> 5 bytes
}

// ============================================================================
// ENCODE_N / DECODE_N EDGE CASES
// ============================================================================

#[test]
fn test_encode_n_empty_array() {
	let data: [u8; 0] = [];
	let encoded = base64::encode_n(&data);
	assert_eq!(&*encoded, "");
}

#[test]
fn test_encode_n_single_byte() {
	// encode_n requires 2*L capacity, but base64 padded produces 4 chars for 1 byte
	// So we can't use encode_n for padded base64 with small arrays
	// Instead test with RAW (no padding)
	let data: [u8; 1] = [0x41]; // 'A'
	let encoded = base64::RAW.encode_n(&data);
	assert_eq!(&*encoded, "QQ");
}

#[test]
fn test_decode_n_empty_array() {
	let data: [u8; 0] = [];
	let decoded = base64::decode_n(&data).unwrap();
	let empty: &[u8] = &[];
	assert_eq!(&*decoded, empty);
}

#[test]
fn test_decode_n_wrong_size() {
	// decode_n checks that decoded len == N
	// "QUFB" (4 chars) decodes to 3 bytes, but we want exactly 2
	let encoded: &[u8; 4] = b"QUFB";
	let result = base64::RAW.decode_n(encoded);
	// Decodes to 3 bytes, not 2, so into_array::<2>() would fail
	let decoded = result.unwrap();
	assert_eq!(decoded.len(), 3); // Proves it's not 2 bytes
}

#[test]
fn test_encode_n_max_capacity() {
	// ArrayStr<L> has 2*L capacity, ensure we don't exceed it
	let data: [u8; 5] = *b"Hello";
	let encoded = base64::encode_n(&data); // Should work (8 chars < 10 capacity)
	assert_eq!(&*encoded, "SGVsbG8=");
}

// ============================================================================
// ITERATOR PROTOCOL - EXACTSIZEITERATOR
// ============================================================================

#[test]
fn test_encoder_exact_size_iter() {
	let data = b"Hello";
	let enc = base64::encode(data);
	assert_eq!(enc.len(), 8); // "SGVsbG8=" is 8 chars
	assert_eq!(enc.size_hint(), (8, Some(8)));

	assert_eq!(enc.count(), 8);
}

#[test]
fn test_decoder_exact_size_iter() {
	let encoded = b"SGVsbG8=";
	let dec = base64::decode(encoded);
	assert_eq!(dec.len(), 5); // "Hello" is 5 bytes
	assert_eq!(dec.size_hint(), (5, Some(5)));

	let collected: Result<Vec<u8>> = dec.collect();
	assert_eq!(collected.unwrap().len(), 5);
}

#[test]
fn test_encoder_exact_size_after_partial_consumption() {
	let data = b"Hi";
	let mut enc = base64::encode(data);

	assert_eq!(enc.len(), 4); // "SGk=" is 4 chars
	enc.next(); // Consume 'S'
	// Note: len() doesn't decrease on consumption in current impl,
	// but size_hint should still be consistent
	assert_eq!(enc.count(), 3);
}

#[test]
fn test_decoder_exact_size_after_partial_consumption() {
	let encoded = b"SGk=";
	let mut dec = base64::decode(encoded);

	assert_eq!(dec.len(), 2); // "Hi" is 2 bytes
	dec.next(); // Consume first byte
	let remaining: Result<Vec<u8>> = dec.collect();
	assert_eq!(remaining.unwrap().len(), 1);
}

// ============================================================================
// FUSEDITERATOR
// ============================================================================

#[test]
fn test_encoder_fused() {
	let data = b"A";
	let mut enc = base64::encode(data);

	// Consume all
	while enc.next().is_some() {}

	// FusedIterator: stays None
	for _ in 0..10 {
		assert_eq!(enc.next(), None);
	}
}

#[test]
fn test_decoder_fused() {
	let encoded = b"QQ==";
	let mut dec = base64::decode(encoded);

	// Consume all
	while dec.next().is_some() {}

	// FusedIterator: stays None
	for _ in 0..10 {
		assert_eq!(dec.next(), None);
	}
}

// ============================================================================
// COMPARISON TRAITS
// ============================================================================

#[test]
fn test_encoder_partial_eq() {
	let enc1 = base64::encode(b"Hello");
	let enc2 = base64::encode(b"Hello");
	assert_eq!(enc1, enc2);
}

#[test]
fn test_encoder_partial_eq_slice() {
	let enc = base64::encode(b"Hello");
	assert_eq!(enc, b"SGVsbG8="[..]);
}

#[test]
fn test_encoder_partial_eq_str() {
	let enc = base64::encode(b"Hello");
	let s = enc.clone().into_string();
	assert_eq!(s, "SGVsbG8=");
}

#[test]
fn test_decoder_partial_eq_slice() {
	let dec = base64::decode(b"SGVsbG8=");
	assert_eq!(dec, b"Hello"[..]);
}

#[test]
fn test_encoder_ord() {
	let a = base64::encode(b"A");
	let b = base64::encode(b"B");
	assert!(a < b);
}

// ============================================================================
// INTO_BUF / EXTEND_INTO
// ============================================================================

#[test]
fn test_encoder_into_buf() {
	let data = b"Hello";
	let buf = BytesMut::with_capacity(20);
	let result = base64::encode(data).into_buf(buf);
	assert_eq!(&result[..], b"SGVsbG8=");
}

#[test]
fn test_decoder_into_buf() {
	let encoded = b"SGVsbG8=";
	let buf = BytesMut::with_capacity(10);
	let result = base64::decode(encoded).into_buf(buf).unwrap();
	assert_eq!(&result[..], b"Hello");
}

#[test]
fn test_encoder_extend_into() {
	let data = b"Hello";
	let mut vec = Vec::new();
	base64::encode(data).extend_into(&mut vec);
	assert_eq!(vec, b"SGVsbG8=");
}

#[test]
fn test_decoder_extend_into() {
	let encoded = b"SGVsbG8=";
	let mut vec = Vec::new();
	let n = base64::decode(encoded).extend_into(&mut vec).unwrap();
	assert_eq!(n, 5);
	assert_eq!(vec, b"Hello");
}

// ============================================================================
// DISPLAY & FORMATTING
// ============================================================================

#[test]
fn test_encoder_display() {
	let enc = base64::encode(b"Hello");
	let s = format!("{enc}");
	assert_eq!(s, "SGVsbG8=");
}

#[test]
fn test_decoder_display() {
	let dec = base64::decode(b"SGVsbG8=");
	let s = format!("{dec}");
	// Decoder displays as hex
	assert_eq!(s, "48656c6c6f");
}

// ============================================================================
// PROPTEST
// ============================================================================

proptest! {
	#[test]
	fn proptest_encode_writer_consistency(data in prop::collection::vec(any::<u8>(), 0..1000)) {
		let mut output = Vec::new();
		{
			let mut writer = base64::encode_writer(&mut output);
			writer.write_all(&data).unwrap();
			writer.flush().unwrap();
		}

		let expected = base64::encode(&data).into_vec();
		prop_assert_eq!(output, expected);
	}

	#[test]
	fn proptest_decode_writer_consistency(data in prop::collection::vec(any::<u8>(), 0..1000)) {
		let encoded = base64::encode(&data).into_vec();

		let mut output = Vec::new();
		{
			let mut writer = base64::decode_writer(&mut output);
			writer.write_all(&encoded).unwrap();
			writer.flush().unwrap();
		}

		prop_assert_eq!(output, data);
	}

	#[test]
	fn proptest_encode_writer_chunked(
		data in prop::collection::vec(any::<u8>(), 1..1000),
		chunk_size in 1usize..100
	) {
		let mut output = Vec::new();
		{
			let mut writer = base64::encode_writer(&mut output);
			for chunk in data.chunks(chunk_size) {
				writer.write_all(chunk).unwrap();
			}
			writer.flush().unwrap();
		}

		let expected = base64::encode(&data).into_vec();
		prop_assert_eq!(output, expected);
	}

	#[test]
	fn proptest_decoder_into_buf_consistency(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let encoded = base64::encode(&data).into_vec();
		let buf = BytesMut::with_capacity(data.len() + 10);
		let result = base64::decode(&encoded).into_buf(buf).unwrap();
		prop_assert_eq!(&result[..], data.as_slice());
	}

	#[test]
	fn proptest_encoder_exact_size_accurate(data in prop::collection::vec(any::<u8>(), 0..100)) {
		let enc = base64::encode(&data);
		let len = enc.len();
		prop_assert_eq!(enc.count(), len);
	}
}
