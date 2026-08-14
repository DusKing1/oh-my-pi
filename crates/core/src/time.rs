//! RFC 3339 conversion for [`SystemTime`].

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Formats a [`SystemTime`] as a UTC RFC 3339 timestamp with second precision.
///
/// Times before [`UNIX_EPOCH`] are clamped to the epoch, preserving the
/// formatter's historical behavior.
#[must_use]
pub fn format_rfc3339(time: SystemTime) -> String {
	let seconds = time
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_secs();
	let days = i64::try_from(seconds / 86_400).unwrap_or(i64::MAX);
	let day_seconds = seconds % 86_400;
	let (year, month, day) = civil_from_days(days);
	let hour = day_seconds / 3_600;
	let minute = day_seconds % 3_600 / 60;
	let second = day_seconds % 60;
	format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Parses an RFC 3339 timestamp into a [`SystemTime`].
///
/// The timestamp must use a four-digit year and include seconds. `T` and `Z`
/// are accepted case-insensitively, the fractional second may contain one to
/// nine digits, and the time zone may be `Z` or a numeric `±HH:MM` offset.
/// Returns [`None`] for invalid dates, times, offsets, or syntax.
#[must_use]
pub fn parse_rfc3339(value: &str) -> Option<SystemTime> {
	if value.len() < 20
		|| value.as_bytes().get(4) != Some(&b'-')
		|| value.as_bytes().get(7) != Some(&b'-')
		|| !matches!(value.as_bytes().get(10), Some(b'T' | b't'))
		|| value.as_bytes().get(13) != Some(&b':')
		|| value.as_bytes().get(16) != Some(&b':')
	{
		return None;
	}
	let year = parse_digits(value, 0, 4)? as i32;
	let month = parse_digits(value, 5, 2)? as u32;
	let day = parse_digits(value, 8, 2)? as u32;
	let hour = parse_digits(value, 11, 2)? as u32;
	let minute = parse_digits(value, 14, 2)? as u32;
	let second = parse_digits(value, 17, 2)? as u32;
	if !(1..=12).contains(&month)
		|| day == 0
		|| day > days_in_month(year, month)
		|| hour > 23
		|| minute > 59
		|| second > 59
	{
		return None;
	}
	let mut cursor = 19;
	let mut nanos = 0_u32;
	if value.as_bytes().get(cursor) == Some(&b'.') {
		cursor += 1;
		let start = cursor;
		while value.as_bytes().get(cursor).is_some_and(u8::is_ascii_digit) {
			cursor += 1;
		}
		let digits = cursor.checked_sub(start)?;
		if digits == 0 || digits > 9 {
			return None;
		}
		nanos = parse_digits(value, start, digits)? as u32;
		for _ in digits..9 {
			nanos *= 10;
		}
	}
	let offset = match value.as_bytes().get(cursor) {
		Some(b'Z' | b'z') if cursor + 1 == value.len() => 0_i64,
		Some(sign @ (b'+' | b'-'))
			if cursor + 6 == value.len() && value.as_bytes().get(cursor + 3) == Some(&b':') =>
		{
			let hours = parse_digits(value, cursor + 1, 2)? as i64;
			let minutes = parse_digits(value, cursor + 4, 2)? as i64;
			if hours > 23 || minutes > 59 {
				return None;
			}
			let seconds = hours * 3_600 + minutes * 60;
			if *sign == b'+' { seconds } else { -seconds }
		},
		_ => return None,
	};
	let local = days_from_civil(year, month, day)
		.checked_mul(86_400)?
		.checked_add(i64::from(hour * 3_600 + minute * 60 + second))?;
	let unix = local.checked_sub(offset)?;
	if unix >= 0 {
		UNIX_EPOCH.checked_add(Duration::new(unix as u64, nanos))
	} else if nanos == 0 {
		UNIX_EPOCH.checked_sub(Duration::from_secs(unix.unsigned_abs()))
	} else {
		UNIX_EPOCH.checked_sub(Duration::new(unix.unsigned_abs() - 1, 1_000_000_000 - nanos))
	}
}

fn parse_digits(value: &str, start: usize, count: usize) -> Option<u64> {
	value
		.get(start..start.checked_add(count)?)?
		.bytes()
		.try_fold(0_u64, |number, byte| {
			if byte.is_ascii_digit() {
				Some(number * 10 + u64::from(byte - b'0'))
			} else {
				None
			}
		})
}

const fn days_in_month(year: i32, month: u32) -> u32 {
	match month {
		1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
		4 | 6 | 9 | 11 => 30,
		2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
		2 => 28,
		_ => 0,
	}
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
	let year = i64::from(year) - i64::from(month <= 2);
	let era = if year >= 0 { year } else { year - 399 } / 400;
	let year_of_era = year - era * 400;
	let month = i64::from(month);
	let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + i64::from(day) - 1;
	let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
	era * 146_097 + day_of_era - 719_468
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
	let days = days + 719_468;
	let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
	let day_of_era = days - era * 146_097;
	let year_of_era =
		(day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
	let mut year = year_of_era + era * 400;
	let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
	let month_prime = (5 * day_of_year + 2) / 153;
	let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
	let month = month_prime + if month_prime < 10 { 3 } else { -9 };
	year += i64::from(month <= 2);
	(year, month, day)
}

#[cfg(test)]
mod tests {
	use std::time::{Duration, UNIX_EPOCH};

	use super::{format_rfc3339, parse_rfc3339};

	#[test]
	fn formats_utc_seconds() {
		assert_eq!(format_rfc3339(UNIX_EPOCH), "1970-01-01T00:00:00Z");
		assert_eq!(
			format_rfc3339(UNIX_EPOCH + Duration::from_secs(1_709_208_245)),
			"2024-02-29T12:04:05Z"
		);
		assert_eq!(format_rfc3339(UNIX_EPOCH - Duration::from_secs(1)), "1970-01-01T00:00:00Z");
	}

	#[test]
	fn parses_fractional_seconds_and_offsets() {
		let expected = UNIX_EPOCH + Duration::new(1_700_098_592, 547_123_456);
		assert_eq!(parse_rfc3339("2023-11-16T01:36:32.547123456Z"), Some(expected));
		assert_eq!(parse_rfc3339("2023-11-16t04:06:32.547123456+02:30"), Some(expected));
		assert_eq!(parse_rfc3339("2023-11-15T19:36:32.547123456-06:00"), Some(expected));
		assert_eq!(
			parse_rfc3339("2023-11-16T01:36:32.5z"),
			Some(UNIX_EPOCH + Duration::new(1_700_098_592, 500_000_000))
		);
	}

	#[test]
	fn parses_pre_epoch_values() {
		assert_eq!(parse_rfc3339("1969-12-31T23:59:59Z"), Some(UNIX_EPOCH - Duration::from_secs(1)));
		assert_eq!(
			parse_rfc3339("1969-12-31T23:59:59.5Z"),
			Some(UNIX_EPOCH - Duration::from_millis(500))
		);
	}

	#[test]
	fn rejects_malformed_values() {
		for value in [
			"",
			"2023-11-16T01:36:32",
			"2023/11/16T01:36:32Z",
			"2023-13-16T01:36:32Z",
			"2023-02-29T01:36:32Z",
			"2024-02-30T01:36:32Z",
			"2023-11-16T24:00:00Z",
			"2023-11-16T01:60:00Z",
			"2023-11-16T01:36:60Z",
			"2023-11-16T01:36:32.Z",
			"2023-11-16T01:36:32.1234567890Z",
			"2023-11-16T01:36:32+24:00",
			"2023-11-16T01:36:32+00:60",
			"2023-11-16T01:36:32Ztrailing",
		] {
			assert_eq!(parse_rfc3339(value), None, "accepted {value:?}");
		}
	}
}
