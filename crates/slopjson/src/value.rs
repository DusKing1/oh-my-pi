//! JSON value tree: [`Value`], [`Number`], and the insertion-ordered
//! [`Object`].

use std::{
	fmt::{self, Write as _},
	ops,
};

use omp_core::SmolStr;

/// A parsed JSON value. `Display` serializes back to compact JSON.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Value {
	/// JSON `null` (also recovered from Python `None`).
	#[default]
	Null,
	/// JSON `true` / `false` (also recovered from Python `True` / `False`).
	Bool(bool),
	/// A finite number; see [`Number`].
	Number(Number),
	/// A string; inline-allocated up to 23 bytes via [`SmolStr`].
	String(SmolStr),
	/// An ordered array of values.
	Array(Vec<Self>),
	/// An insertion-ordered object; see [`Object`].
	Object(Object),
}

/// Shared fallback for [`Value`] indexing misses.
static NULL: Value = Value::Null;

impl Value {
	/// Member lookup; `None` unless this is an object containing `key`.
	pub fn get(&self, key: &str) -> Option<&Self> {
		match self {
			Self::Object(object) => object.get(key),
			_ => None,
		}
	}

	/// Whether this is `Null`.
	pub const fn is_null(&self) -> bool {
		matches!(self, Self::Null)
	}

	/// Whether this is an array.
	pub const fn is_array(&self) -> bool {
		matches!(self, Self::Array(_))
	}

	/// Whether this is an object.
	pub const fn is_object(&self) -> bool {
		matches!(self, Self::Object(_))
	}

	/// Boolean value; `None` for non-booleans.
	pub const fn as_bool(&self) -> Option<bool> {
		match self {
			Self::Bool(b) => Some(*b),
			_ => None,
		}
	}

	/// String contents; `None` for non-strings.
	pub fn as_str(&self) -> Option<&str> {
		match self {
			Self::String(s) => Some(s),
			_ => None,
		}
	}

	/// Numeric value; `None` for non-numbers.
	pub const fn as_number(&self) -> Option<Number> {
		match self {
			Self::Number(n) => Some(*n),
			_ => None,
		}
	}

	/// Integer value when this is a number that fits in `i64`.
	pub fn as_i64(&self) -> Option<i64> {
		self.as_number().and_then(Number::as_i64)
	}

	/// Integer value when this is a non-negative integer number.
	pub fn as_u64(&self) -> Option<u64> {
		self.as_number().and_then(Number::as_u64)
	}

	/// Numeric value as `f64` (lossy for large integers).
	pub fn as_f64(&self) -> Option<f64> {
		self.as_number().map(Number::as_f64)
	}

	/// Array elements; `None` for non-arrays.
	pub fn as_array(&self) -> Option<&[Self]> {
		match self {
			Self::Array(items) => Some(items),
			_ => None,
		}
	}

	/// Object members; `None` for non-objects.
	pub const fn as_object(&self) -> Option<&Object> {
		match self {
			Self::Object(object) => Some(object),
			_ => None,
		}
	}
}

/// Object member access; missing keys and non-objects yield `Null`.
impl ops::Index<&str> for Value {
	type Output = Self;

	fn index(&self, key: &str) -> &Self {
		self.get(key).unwrap_or(&NULL)
	}
}

/// Array element access; out-of-bounds and non-arrays yield `Null`.
impl ops::Index<usize> for Value {
	type Output = Self;

	fn index(&self, index: usize) -> &Self {
		match self {
			Self::Array(items) => items.get(index).unwrap_or(&NULL),
			_ => &NULL,
		}
	}
}

impl fmt::Display for Value {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Null => f.write_str("null"),
			Self::Bool(true) => f.write_str("true"),
			Self::Bool(false) => f.write_str("false"),
			Self::Number(n) => n.fmt(f),
			Self::String(s) => write_escaped(f, s),
			Self::Array(items) => {
				f.write_char('[')?;
				for (i, item) in items.iter().enumerate() {
					if i > 0 {
						f.write_char(',')?;
					}
					item.fmt(f)?;
				}
				f.write_char(']')
			},
			Self::Object(object) => object.fmt(f),
		}
	}
}

/// Write `s` as a JSON string literal with the minimal required escapes.
fn write_escaped(f: &mut fmt::Formatter<'_>, s: &str) -> fmt::Result {
	f.write_char('"')?;
	let mut start = 0;
	for (i, b) in s.bytes().enumerate() {
		let short: Option<&str> = match b {
			b'"' => Some("\\\""),
			b'\\' => Some("\\\\"),
			0x08 => Some("\\b"),
			0x09 => Some("\\t"),
			0x0a => Some("\\n"),
			0x0c => Some("\\f"),
			0x0d => Some("\\r"),
			b if b < 0x20 => None, // \uXXXX below
			_ => continue,
		};
		f.write_str(&s[start..i])?;
		match short {
			Some(esc) => f.write_str(esc)?,
			None => write!(f, "\\u{b:04x}")?,
		}
		start = i + 1;
	}
	f.write_str(&s[start..])?;
	f.write_char('"')
}

// ── Number ───────────────────────────────────────────────────────────────────

/// A JSON number: an exact integer when it fits, `f64` otherwise. Never
/// non-finite — construction rejects NaN and infinities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Number(N);

#[derive(Debug, Clone, Copy, PartialEq)]
enum N {
	/// Integer in `0..=u64::MAX`.
	PosInt(u64),
	/// Integer in `i64::MIN..0`.
	NegInt(i64),
	/// Always finite.
	Float(f64),
}

impl Number {
	/// A finite float; `None` for NaN and infinities.
	pub fn from_f64(value: f64) -> Option<Self> {
		value.is_finite().then_some(Self(N::Float(value)))
	}

	/// Exact integer value when it fits in `i64`; `None` for floats.
	pub fn as_i64(self) -> Option<i64> {
		match self.0 {
			N::PosInt(v) => i64::try_from(v).ok(),
			N::NegInt(v) => Some(v),
			N::Float(_) => None,
		}
	}

	/// Exact integer value when non-negative; `None` for floats.
	pub const fn as_u64(self) -> Option<u64> {
		match self.0 {
			N::PosInt(v) => Some(v),
			_ => None,
		}
	}

	/// Numeric value as `f64` (lossy above 2^53).
	pub const fn as_f64(self) -> f64 {
		match self.0 {
			N::PosInt(v) => v as f64,
			N::NegInt(v) => v as f64,
			N::Float(v) => v,
		}
	}

	/// Whether this number is stored as a float (has a fractional or
	/// exponent form, or overflowed the integer range).
	pub const fn is_f64(self) -> bool {
		matches!(self.0, N::Float(_))
	}
}

impl fmt::Display for Number {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self.0 {
			N::PosInt(v) => v.fmt(f),
			N::NegInt(v) => v.fmt(f),
			// Keep integral floats recognizably float ("1.0", not "1") so
			// serialization does not silently change the number's type.
			N::Float(v) if v.fract() == 0.0 && v.abs() < 1e16 => write!(f, "{v:.1}"),
			N::Float(v) => v.fmt(f),
		}
	}
}

impl From<u64> for Number {
	fn from(value: u64) -> Self {
		Self(N::PosInt(value))
	}
}

impl From<i64> for Number {
	fn from(value: i64) -> Self {
		if value < 0 {
			Self(N::NegInt(value))
		} else {
			Self(N::PosInt(value as u64))
		}
	}
}

// ── Object ───────────────────────────────────────────────────────────────────

/// Insertion-ordered JSON object. Duplicate inserts overwrite in place (last
/// value wins); equality is order-insensitive like JSON object semantics.
#[derive(Debug, Clone, Default)]
pub struct Object(Vec<(SmolStr, Value)>);

/// Borrowed iterator over an [`Object`]'s members in insertion order.
pub type ObjectIter<'a> = impl DoubleEndedIterator<Item = (&'a SmolStr, &'a Value)>
	+ ExactSizeIterator
	+ std::iter::FusedIterator
	+ Clone;

/// Mutable iterator over an [`Object`]'s members in insertion order.
pub type ObjectIterMut<'a> = impl DoubleEndedIterator<Item = (&'a SmolStr, &'a mut Value)>
	+ ExactSizeIterator
	+ std::iter::FusedIterator;

impl Object {
	/// An empty object.
	pub const fn new() -> Self {
		Self(Vec::new())
	}

	/// An empty object with room for `capacity` members before reallocating.
	pub fn with_capacity(capacity: usize) -> Self {
		Self(Vec::with_capacity(capacity))
	}

	/// Number of members.
	pub const fn len(&self) -> usize {
		self.0.len()
	}

	/// Whether the object has no members.
	pub const fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	/// Value for `key`; `None` when absent.
	pub fn get(&self, key: &str) -> Option<&Value> {
		self.0.iter().find_map(|(k, v)| (&**k == key).then_some(v))
	}

	/// Insert or overwrite `key`, returning the previous value if any.
	pub fn insert(&mut self, key: SmolStr, value: Value) -> Option<Value> {
		if let Some(slot) = self.0.iter_mut().find(|(k, _)| *k == key) {
			Some(std::mem::replace(&mut slot.1, value))
		} else {
			self.0.push((key, value));
			None
		}
	}

	/// Members in insertion order.
	#[define_opaque(ObjectIter)]
	pub fn iter(&self) -> ObjectIter<'_> {
		self.0.iter().map(|(key, value)| (key, value))
	}

	/// Members in insertion order, with mutable values.
	#[define_opaque(ObjectIterMut)]
	pub fn iter_mut(&mut self) -> ObjectIterMut<'_> {
		self.0.iter_mut().map(|(key, value)| (&*key, value))
	}
}

impl PartialEq for Object {
	fn eq(&self, other: &Self) -> bool {
		self.0.len() == other.0.len()
			&& self
				.0
				.iter()
				.all(|(key, value)| other.get(key) == Some(value))
	}
}

impl fmt::Display for Object {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_char('{')?;
		for (i, (key, value)) in self.0.iter().enumerate() {
			if i > 0 {
				f.write_char(',')?;
			}
			write_escaped(f, key)?;
			f.write_char(':')?;
			value.fmt(f)?;
		}
		f.write_char('}')
	}
}

impl<'a> IntoIterator for &'a Object {
	type IntoIter = ObjectIter<'a>;
	type Item = (&'a SmolStr, &'a Value);

	fn into_iter(self) -> Self::IntoIter {
		self.iter()
	}
}

impl<'a> IntoIterator for &'a mut Object {
	type IntoIter = ObjectIterMut<'a>;
	type Item = (&'a SmolStr, &'a mut Value);

	fn into_iter(self) -> Self::IntoIter {
		self.iter_mut()
	}
}

impl IntoIterator for Object {
	type IntoIter = std::vec::IntoIter<(SmolStr, Value)>;
	type Item = (SmolStr, Value);

	fn into_iter(self) -> Self::IntoIter {
		self.0.into_iter()
	}
}

impl FromIterator<(SmolStr, Value)> for Object {
	fn from_iter<I: IntoIterator<Item = (SmolStr, Value)>>(iter: I) -> Self {
		let mut object = Self::new();
		for (key, value) in iter {
			object.insert(key, value);
		}
		object
	}
}

// ── Conversions into Value ───────────────────────────────────────────────────

impl From<bool> for Value {
	fn from(value: bool) -> Self {
		Self::Bool(value)
	}
}

impl From<&str> for Value {
	fn from(value: &str) -> Self {
		Self::String(SmolStr::from(value))
	}
}

impl From<String> for Value {
	fn from(value: String) -> Self {
		Self::String(SmolStr::from(value))
	}
}

impl From<SmolStr> for Value {
	fn from(value: SmolStr) -> Self {
		Self::String(value)
	}
}

impl From<Number> for Value {
	fn from(value: Number) -> Self {
		Self::Number(value)
	}
}

impl From<Object> for Value {
	fn from(value: Object) -> Self {
		Self::Object(value)
	}
}

/// Non-finite floats become `Null`, mirroring JSON's lack of NaN/Infinity.
impl From<f64> for Value {
	fn from(value: f64) -> Self {
		Number::from_f64(value).map_or(Self::Null, Self::Number)
	}
}

impl From<f32> for Value {
	fn from(value: f32) -> Self {
		Self::from(f64::from(value))
	}
}

macro_rules! impl_from_unsigned {
	($($ty:ty),*) => {$(
		impl From<$ty> for Value {
			fn from(value: $ty) -> Self {
				Self::Number(Number::from(value as u64))
			}
		}
	)*};
}

macro_rules! impl_from_signed {
	($($ty:ty),*) => {$(
		impl From<$ty> for Value {
			fn from(value: $ty) -> Self {
				Self::Number(Number::from(value as i64))
			}
		}
	)*};
}

impl_from_unsigned!(u8, u16, u32, u64, usize);
impl_from_signed!(i8, i16, i32, i64, isize);

impl<T: Into<Self>> From<Vec<T>> for Value {
	fn from(values: Vec<T>) -> Self {
		Self::Array(values.into_iter().map(Into::into).collect())
	}
}

impl<T: Into<Self> + Clone> From<&[T]> for Value {
	fn from(values: &[T]) -> Self {
		Self::Array(values.iter().cloned().map(Into::into).collect())
	}
}

impl<T: Into<Self>> From<Option<T>> for Value {
	fn from(value: Option<T>) -> Self {
		value.map_or(Self::Null, Into::into)
	}
}

impl From<()> for Value {
	fn from((): ()) -> Self {
		Self::Null
	}
}

impl<T: Into<Self>> FromIterator<T> for Value {
	fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
		Self::Array(iter.into_iter().map(Into::into).collect())
	}
}

// ── serde integration ────────────────────────────────────────────────────────

impl serde::Serialize for Number {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		if let Some(unsigned) = self.as_u64() {
			serializer.serialize_u64(unsigned)
		} else if let Some(signed) = self.as_i64() {
			serializer.serialize_i64(signed)
		} else {
			serializer.serialize_f64(self.as_f64())
		}
	}
}

impl serde::Serialize for Value {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		match self {
			Self::Null => serializer.serialize_unit(),
			Self::Bool(b) => serializer.serialize_bool(*b),
			Self::Number(n) => n.serialize(serializer),
			Self::String(s) => serializer.serialize_str(s),
			Self::Array(items) => serializer.collect_seq(items),
			Self::Object(object) => serializer.collect_map(object.iter()),
		}
	}
}

/// `Value` is an ordinary visitor over any self-describing deserializer;
/// [`parse`](crate::parse) is exactly `from_str::<Value>`.
impl<'de> serde::Deserialize<'de> for Value {
	fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
		struct ValueVisitor;

		impl<'de> serde::de::Visitor<'de> for ValueVisitor {
			type Value = Value;

			fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
				f.write_str("any JSON value")
			}

			fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
				Ok(Value::Bool(v))
			}

			fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
				Ok(Value::Number(Number::from(v)))
			}

			fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
				Ok(Value::Number(Number::from(v)))
			}

			fn visit_f64<E>(self, v: f64) -> Result<Value, E> {
				Ok(Value::from(v))
			}

			fn visit_str<E>(self, v: &str) -> Result<Value, E> {
				Ok(Value::String(SmolStr::from(v)))
			}

			fn visit_string<E>(self, v: String) -> Result<Value, E> {
				Ok(Value::String(SmolStr::from(v)))
			}

			fn visit_none<E>(self) -> Result<Value, E> {
				Ok(Value::Null)
			}

			fn visit_unit<E>(self) -> Result<Value, E> {
				Ok(Value::Null)
			}

			fn visit_some<D: serde::Deserializer<'de>>(
				self,
				deserializer: D,
			) -> Result<Value, D::Error> {
				serde::Deserialize::deserialize(deserializer)
			}

			fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
				let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(64));
				while let Some(item) = seq.next_element()? {
					items.push(item);
				}
				Ok(Value::Array(items))
			}

			fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
				let mut object = Object::with_capacity(map.size_hint().unwrap_or(0).min(64));
				while let Some((key, value)) = map.next_entry::<SmolStr, Value>()? {
					object.insert(key, value);
				}
				Ok(Value::Object(object))
			}
		}

		deserializer.deserialize_any(ValueVisitor)
	}
}
