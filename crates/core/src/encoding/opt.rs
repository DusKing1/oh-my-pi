//! Optimized scalar implementations for Base64 and Base32 encoding/decoding.
//!
//! Provides hand-tuned, SIMD-ready encoders and decoders with loop unrolling
//! and minimal branching. These implementations process data in large blocks
//! (24 bytes for Base64, 10 bytes for Base32) before handling remainder bytes.
//!
//! The optimized paths are automatically selected at runtime by the parent
//! `Encoding` type when `N == 64` or `N == 32`.

#![allow(
	clippy::many_single_char_names,
	reason = "single char names are clearer for encoding/decoding bit manipulation"
)]

use std::hint;

use super::{
	base_n::{DTable, ETable},
	error::{DecodeError, Result},
};

/// ===============================
/// Base64 encode (optimized scalar)
/// ===============================
#[inline(always)]
pub fn enc64(src: &[u8], sym256: &ETable, dst: &mut [u8], pad: u8) -> usize {
	let mut si = 0usize;
	let mut di = 0usize;

	// SAFETY: We access src and dst through raw pointers with bounds checking.
	// All indices are validated: si+24/si+3 <= src.len() ensures valid reads,
	// and di is incremented proportionally to maintain alignment with dst capacity.
	unsafe {
		let sp = src.as_ptr();
		let dp = dst.as_mut_ptr();

		// Process 24 bytes -> 32 chars per iteration (8 triplets unrolled)
		while si + 24 <= src.len() {
			macro_rules! enc_triplet {
				($ofs:expr, $outofs:expr) => {{
					let p = sp.add(si + $ofs);
					let x = ((*p as u32) << 16) | ((*p.add(1) as u32) << 8) | (*p.add(2) as u32);
					let q = dp.add(di + $outofs);
					q.add(0)
						.write(*sym256.get_unchecked((x >> 18) as u8 as usize));
					q.add(1)
						.write(*sym256.get_unchecked((x >> 12) as u8 as usize));
					q.add(2)
						.write(*sym256.get_unchecked((x >> 6) as u8 as usize));
					q.add(3).write(*sym256.get_unchecked(x as u8 as usize));
				}};
			}

			enc_triplet!(0, 0);
			enc_triplet!(3, 4);
			enc_triplet!(6, 8);
			enc_triplet!(9, 12);
			enc_triplet!(12, 16);
			enc_triplet!(15, 20);
			enc_triplet!(18, 24);
			enc_triplet!(21, 28);

			si += 24;
			di += 32;
		}

		// Remaining full 3-byte groups
		while si + 3 <= src.len() {
			let p = sp.add(si);
			let x = ((*p as u32) << 16) | ((*p.add(1) as u32) << 8) | (*p.add(2) as u32);
			let q = dp.add(di);
			q.add(0)
				.write(*sym256.get_unchecked((x >> 18) as u8 as usize));
			q.add(1)
				.write(*sym256.get_unchecked((x >> 12) as u8 as usize));
			q.add(2)
				.write(*sym256.get_unchecked((x >> 6) as u8 as usize));
			q.add(3).write(*sym256.get_unchecked(x as u8 as usize));
			si += 3;
			di += 4;
		}
	}

	// Tail (1 or 2 bytes)
	match src.len() - si {
		0 => {},
		1 => {
			let v = (src[si] as u32) << 16;
			dst[di] = sym256[(v >> 18) as u8 as usize];
			dst[di + 1] = sym256[(v >> 12) as u8 as usize];
			di += 2;
			if pad != 0 {
				dst[di] = pad;
				dst[di + 1] = pad;
				di += 2;
			}
		},
		2 => {
			let v = ((src[si] as u32) << 16) | ((src[si + 1] as u32) << 8);
			dst[di] = sym256[(v >> 18) as u8 as usize];
			dst[di + 1] = sym256[(v >> 12) as u8 as usize];
			dst[di + 2] = sym256[(v >> 6) as u8 as usize];
			di += 3;
			if pad != 0 {
				dst[di] = pad;
				di += 1;
			}
		},
		_ => unreachable!(),
	}

	di
}

/// ===============================
/// Base64 decode (optimized scalar)
/// ===============================
#[inline(always)]
pub fn dec64(src: &[u8], val: &DTable, dst: &mut [u8], pad: u8) -> Result<usize> {
	let len = src.len();
	let mut full_quads = len / 4;

	// If padded, reserve the last quad for special handling when '=' present.
	if pad != 0 && len >= 4 {
		let last = src[len - 1];
		let prev = src[len - 2];
		if last == pad || prev == pad {
			full_quads = full_quads.saturating_sub(1);
		}
	}

	let mut si = 0usize;
	let mut di = 0usize;

	// SAFETY: We access src and dst through raw pointers with bounds checking.
	// Indices si+32/si+4 are validated against full_quads*4, and di is incremented
	// proportionally. All character values are validated before use.
	unsafe {
		let sp = src.as_ptr();
		let dp = dst.as_mut_ptr();

		// Process 8 quads (32 chars) -> 24 bytes per iteration
		while si + 32 <= full_quads * 4 {
			macro_rules! dec_quad {
				($ofs:expr, $o:expr) => {{
					let a = *sp.add(si + $ofs);
					let b = *sp.add(si + $ofs + 1);
					let c = *sp.add(si + $ofs + 2);
					let d = *sp.add(si + $ofs + 3);
					let va = *val.get_unchecked(a as usize);
					let vb = *val.get_unchecked(b as usize);
					let vc = *val.get_unchecked(c as usize);
					let vd = *val.get_unchecked(d as usize);
					if (va | vb | vc | vd) >= 64 {
						hint::cold_path();
						if va >= 64 {
							return Err(DecodeError::InvalidCharacter(a));
						} else if vb >= 64 {
							return Err(DecodeError::InvalidCharacter(b));
						} else if vc >= 64 {
							return Err(DecodeError::InvalidCharacter(c));
						}
						return Err(DecodeError::InvalidCharacter(d));
					}
					let v = ((va as u32) << 18) | ((vb as u32) << 12) | ((vc as u32) << 6) | (vd as u32);
					let q = dp.add(di + $o);
					q.add(0).write((v >> 16) as u8);
					q.add(1).write((v >> 8) as u8);
					q.add(2).write(v as u8);
				}};
			}

			dec_quad!(0, 0);
			dec_quad!(4, 3);
			dec_quad!(8, 6);
			dec_quad!(12, 9);
			dec_quad!(16, 12);
			dec_quad!(20, 15);
			dec_quad!(24, 18);
			dec_quad!(28, 21);

			si += 32;
			di += 24;
		}

		// Remaining full quads (0..3)
		while si + 4 <= full_quads * 4 {
			let a = *sp.add(si);
			let b = *sp.add(si + 1);
			let c = *sp.add(si + 2);
			let d = *sp.add(si + 3);
			let va = *val.get_unchecked(a as usize);
			let vb = *val.get_unchecked(b as usize);
			let vc = *val.get_unchecked(c as usize);
			let vd = *val.get_unchecked(d as usize);
			if (va | vb | vc | vd) >= 64 {
				hint::cold_path();
				if va >= 64 {
					return Err(DecodeError::InvalidCharacter(a));
				} else if vb >= 64 {
					return Err(DecodeError::InvalidCharacter(b));
				} else if vc >= 64 {
					return Err(DecodeError::InvalidCharacter(c));
				}
				return Err(DecodeError::InvalidCharacter(d));
			}
			let v = ((va as u32) << 18) | ((vb as u32) << 12) | ((vc as u32) << 6) | (vd as u32);
			let q = dp.add(di);
			q.add(0).write((v >> 16) as u8);
			q.add(1).write((v >> 8) as u8);
			q.add(2).write(v as u8);
			si += 4;
			di += 3;
		}
	}

	// Tail
	let rem = &src[si..];
	if pad != 0 {
		match rem {
			[] => Ok(di),
			&[a, b, c, d] => {
				let va = val[a as usize];
				let vb = val[b as usize];
				if (va | vb) >= 64 {
					hint::cold_path();
					return Err(DecodeError::InvalidCharacter(if va >= 64 { a } else { b }));
				}

				match (c == pad, d == pad) {
					(true, true) => {
						// 'xx==': 1 output byte
						let v = ((va as u32) << 18) | ((vb as u32) << 12);
						dst[di] = (v >> 16) as u8;
						di += 1;
					},
					(false, true) => {
						// 'xxx=': 2 output bytes
						let vc = val[c as usize];
						if vc >= 64 {
							hint::cold_path();
							return Err(DecodeError::InvalidCharacter(c));
						}
						let v = ((va as u32) << 18) | ((vb as u32) << 12) | ((vc as u32) << 6);
						dst[di] = (v >> 16) as u8;
						dst[di + 1] = (v >> 8) as u8;
						di += 2;
					},
					_ => {
						// 'xxxx': 3 output bytes
						let vc = val[c as usize];
						let vd = val[d as usize];
						if (vc | vd) >= 64 {
							hint::cold_path();
							return Err(DecodeError::InvalidCharacter(if vc >= 64 { c } else { d }));
						}
						let v =
							((va as u32) << 18) | ((vb as u32) << 12) | ((vc as u32) << 6) | (vd as u32);
						dst[di] = (v >> 16) as u8;
						dst[di + 1] = (v >> 8) as u8;
						dst[di + 2] = v as u8;
						di += 3;
					},
				}
				Ok(di)
			},
			_ => Err(DecodeError::InvalidLength),
		}
	} else {
		// No-pad mode: allow 0,2,3 trailing chars producing 1 or 2 bytes.
		match rem {
			[] => Ok(di),
			&[a, b] => {
				let va = val[a as usize];
				let vb = val[b as usize];
				if va >= 64 || vb >= 64 {
					hint::cold_path();
					return Err(DecodeError::InvalidCharacter(if va >= 64 { a } else { b }));
				}
				let v = ((va as u32) << 18) | ((vb as u32) << 12);
				dst[di] = (v >> 16) as u8;
				Ok(di + 1)
			},
			&[a, b, c] => {
				let va = val[a as usize];
				let vb = val[b as usize];
				let vc = val[c as usize];
				if (va | vb | vc) >= 64 {
					hint::cold_path();
					let bad = if va >= 64 {
						a
					} else if vb >= 64 {
						b
					} else {
						c
					};
					return Err(DecodeError::InvalidCharacter(bad));
				}
				let v = ((va as u32) << 18) | ((vb as u32) << 12) | ((vc as u32) << 6);
				dst[di] = (v >> 16) as u8;
				dst[di + 1] = (v >> 8) as u8;
				Ok(di + 2)
			},
			_ => Err(DecodeError::InvalidLength),
		}
	}
}

/// ===============================
/// Base32 encode (optimized scalar)
/// ===============================
#[inline(always)]
pub fn enc32(src: &[u8], sym256: &ETable, dst: &mut [u8], pad: u8) -> usize {
	let mut si = 0usize;
	let mut di = 0usize;

	// SAFETY: We access src and dst through raw pointers with bounds checking.
	// All indices are validated: si+10/si+5 <= src.len() ensures valid reads,
	// and di is incremented proportionally to maintain alignment with dst capacity.
	unsafe {
		let sp = src.as_ptr();
		let dp = dst.as_mut_ptr();

		// Process 10 bytes -> 16 chars per iteration (2×5B blocks unrolled)
		while si + 10 <= src.len() {
			macro_rules! enc_block5 {
				($ofs:expr, $o:expr) => {{
					let a = *sp.add(si + $ofs + 0) as u64;
					let b = *sp.add(si + $ofs + 1) as u64;
					let c = *sp.add(si + $ofs + 2) as u64;
					let d = *sp.add(si + $ofs + 3) as u64;
					let e = *sp.add(si + $ofs + 4) as u64;
					let v = (a << 32) | (b << 24) | (c << 16) | (d << 8) | e;
					let q = dp.add(di + $o);
					q.add(0)
						.write(*sym256.get_unchecked((v >> 35) as u8 as usize));
					q.add(1)
						.write(*sym256.get_unchecked((v >> 30) as u8 as usize));
					q.add(2)
						.write(*sym256.get_unchecked((v >> 25) as u8 as usize));
					q.add(3)
						.write(*sym256.get_unchecked((v >> 20) as u8 as usize));
					q.add(4)
						.write(*sym256.get_unchecked((v >> 15) as u8 as usize));
					q.add(5)
						.write(*sym256.get_unchecked((v >> 10) as u8 as usize));
					q.add(6)
						.write(*sym256.get_unchecked((v >> 5) as u8 as usize));
					q.add(7).write(*sym256.get_unchecked(v as u8 as usize));
				}};
			}

			enc_block5!(0, 0);
			enc_block5!(5, 8);

			si += 10;
			di += 16;
		}

		// Remaining full 5-byte groups
		while si + 5 <= src.len() {
			let a = src[si] as u64;
			let b = src[si + 1] as u64;
			let c = src[si + 2] as u64;
			let d = src[si + 3] as u64;
			let e = src[si + 4] as u64;
			let v = (a << 32) | (b << 24) | (c << 16) | (d << 8) | e;
			dst[di] = sym256[(v >> 35) as u8 as usize];
			dst[di + 1] = sym256[(v >> 30) as u8 as usize];
			dst[di + 2] = sym256[(v >> 25) as u8 as usize];
			dst[di + 3] = sym256[(v >> 20) as u8 as usize];
			dst[di + 4] = sym256[(v >> 15) as u8 as usize];
			dst[di + 5] = sym256[(v >> 10) as u8 as usize];
			dst[di + 6] = sym256[(v >> 5) as u8 as usize];
			dst[di + 7] = sym256[(v) as u8 as usize];
			si += 5;
			di += 8;
		}
	}

	// Tail (1..=4 bytes)
	let tail = src.len() - si;
	if tail > 0 {
		let mut buf = [0u8; 5];
		buf[..tail].copy_from_slice(&src[si..]);
		let v = ((buf[0] as u64) << 32)
			| ((buf[1] as u64) << 24)
			| ((buf[2] as u64) << 16)
			| ((buf[3] as u64) << 8)
			| (buf[4] as u64);

		let needed = (tail * 8).div_ceil(5); // ceil
		for i in 0..needed {
			let shift = 35 - i * 5;
			dst[di + i] = sym256[((v >> shift) as u8) as usize];
		}
		di += needed;

		if pad != 0 {
			// Pad to 8-char boundary
			let pad_count = (8 - needed) & 7;
			for _ in 0..pad_count {
				dst[di] = pad;
				di += 1;
			}
		}
	}

	di
}

/// ===============================
/// Base32 decode (optimized scalar)
/// ===============================
#[inline(always)]
pub fn dec32(src: &[u8], val: &DTable, dst: &mut [u8], pad: u8) -> Result<usize> {
	let len = src.len();
	let mut full_blocks = len / 8;

	// If padded, reserve last block when we see trailing '='.
	if pad != 0 && len >= 8 && src[len - 1] == pad {
		full_blocks = full_blocks.saturating_sub(1);
	}

	let mut si = 0usize;
	let mut di = 0usize;

	// SAFETY: We access src and dst through raw pointers with bounds checking.
	// Indices si+8 are validated against full_blocks*8, and di is incremented
	// proportionally. All character values are validated before use.
	unsafe {
		let sp = src.as_ptr();
		let dp = dst.as_mut_ptr();

		// Process 8 chars -> 5 bytes (unroll 1× here; two-at-once is also fine)
		while si + 8 <= full_blocks * 8 {
			let a = *sp.add(si);
			let b = *sp.add(si + 1);
			let c = *sp.add(si + 2);
			let d = *sp.add(si + 3);
			let e = *sp.add(si + 4);
			let f = *sp.add(si + 5);
			let g = *sp.add(si + 6);
			let h = *sp.add(si + 7);

			let va = *val.get_unchecked(a as usize);
			let vb = *val.get_unchecked(b as usize);
			let vc = *val.get_unchecked(c as usize);
			let vd = *val.get_unchecked(d as usize);
			let ve = *val.get_unchecked(e as usize);
			let vf = *val.get_unchecked(f as usize);
			let vg = *val.get_unchecked(g as usize);
			let vh = *val.get_unchecked(h as usize);

			// Valid base32 values are < 32. Any >=32 (including 0xFE pad marker) is invalid
			// here.
			if (va | vb | vc | vd | ve | vf | vg | vh) >= 32 {
				hint::cold_path();
				if va >= 32 {
					return Err(DecodeError::InvalidCharacter(a));
				} else if vb >= 32 {
					return Err(DecodeError::InvalidCharacter(b));
				} else if vc >= 32 {
					return Err(DecodeError::InvalidCharacter(c));
				} else if vd >= 32 {
					return Err(DecodeError::InvalidCharacter(d));
				} else if ve >= 32 {
					return Err(DecodeError::InvalidCharacter(e));
				} else if vf >= 32 {
					return Err(DecodeError::InvalidCharacter(f));
				} else if vg >= 32 {
					return Err(DecodeError::InvalidCharacter(g));
				}
				return Err(DecodeError::InvalidCharacter(h));
			}

			let v = ((va as u64) << 35)
				| ((vb as u64) << 30)
				| ((vc as u64) << 25)
				| ((vd as u64) << 20)
				| ((ve as u64) << 15)
				| ((vf as u64) << 10)
				| ((vg as u64) << 5)
				| (vh as u64);

			let q = dp.add(di);
			q.add(0).write((v >> 32) as u8);
			q.add(1).write((v >> 24) as u8);
			q.add(2).write((v >> 16) as u8);
			q.add(3).write((v >> 8) as u8);
			q.add(4).write(v as u8);

			si += 8;
			di += 5;
		}
	}

	let rem = len - si;
	if pad != 0 {
		// Padded tail (if any)
		if rem == 0 {
			return Ok(di);
		} else if rem != 8 {
			return Err(DecodeError::InvalidLength);
		}

		let block = &src[si..si + 8];

		// Count trailing '=' in the block.
		let mut p = 0usize;
		while p < 8 && block[7 - p] == pad {
			p += 1;
		}

		// Allowed pad counts per RFC 4648: 0, 1, 3, 4, 6
		let (valid, out_bytes) = match p {
			0 => (8, 5),
			1 => (7, 4),
			3 => (5, 3),
			4 => (4, 2),
			6 => (2, 1),
			_ => return Err(DecodeError::InvalidLength),
		};

		// Validate no PAD among the first `valid` chars and decode them.
		let mut v = 0u64;
		for (i, &ch) in block[..valid].iter().enumerate() {
			if ch == pad {
				hint::cold_path();
				return Err(DecodeError::InvalidLength);
			}
			let vv = val[ch as usize];
			if vv >= 32 {
				hint::cold_path();
				return Err(DecodeError::InvalidCharacter(ch));
			}
			v |= (vv as u64) << (35 - 5 * i);
		}

		// Emit bytes from the high end.
		for j in 0..out_bytes {
			dst[di + j] = ((v >> (32 - 8 * j)) & 0xff) as u8;
		}
		di += out_bytes;

		// Any non-PAD in the trailing area is invalid.
		for &ch in &block[valid..8] {
			if ch != pad {
				hint::cold_path();
				return Err(DecodeError::InvalidLength);
			}
		}

		Ok(di)
	} else {
		// No-pad tail: allowed lengths are 0, 2, 4, 5, 7 -> 1/2/3/4 bytes.
		match rem {
			0 => Ok(di),
			2 | 4 | 5 | 7 => {
				let mut v = 0u64;
				for i in 0..rem {
					let ch = src[si + i];
					let vv = val[ch as usize];
					if vv >= 32 {
						hint::cold_path();
						return Err(DecodeError::InvalidCharacter(ch));
					}
					v |= (vv as u64) << (35 - 5 * i);
				}
				let out_bytes = match rem {
					2 => 1, // 10 bits -> 1 byte
					4 => 2, // 20 bits -> 2 bytes
					5 => 3, // 25 bits -> 3 bytes
					7 => 4, // 35 bits -> 4 bytes
					_ => unreachable!(),
				};
				for j in 0..out_bytes {
					dst[di + j] = ((v >> (32 - 8 * j)) & 0xff) as u8;
				}
				Ok(di + out_bytes)
			},
			_ => Err(DecodeError::InvalidLength),
		}
	}
}
