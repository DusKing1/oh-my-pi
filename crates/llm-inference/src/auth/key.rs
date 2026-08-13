//! Encryption-key sources for persistent credentials.

#[cfg(target_os = "macos")]
use std::str;
use std::{fmt, sync::Arc};

use omp_core::Str;
#[cfg(target_os = "macos")]
use omp_core::hex;
use parking_lot::RwLock;
#[cfg(target_os = "macos")]
use ring::rand::{SecureRandom, SystemRandom};
use zeroize::Zeroizing;

const KEY_BYTES: usize = 32;

/// Stable, non-secret identifier for an encryption key.
#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KeyId(Str);

impl KeyId {
	/// Creates a key identifier from stored text.
	#[must_use]
	pub fn new(value: impl Into<Str>) -> Self {
		Self(value.into())
	}

	/// Borrows the identifier as text.
	#[must_use]
	pub fn as_str(&self) -> &str {
		self.0.as_str()
	}
}

impl fmt::Debug for KeyId {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.debug_tuple("KeyId").field(&self.0).finish()
	}
}

impl fmt::Display for KeyId {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(self.as_str())
	}
}

/// A zeroizing 256-bit authenticated-encryption key.
pub struct EncryptionKey {
	id:    KeyId,
	bytes: Zeroizing<[u8; KEY_BYTES]>,
}

impl EncryptionKey {
	/// Constructs key material from an explicit 256-bit value.
	#[must_use]
	pub fn new(id: KeyId, bytes: [u8; KEY_BYTES]) -> Self {
		Self { id, bytes: Zeroizing::new(bytes) }
	}

	/// Returns the non-secret key identifier.
	#[must_use]
	pub fn id(&self) -> &KeyId {
		&self.id
	}

	pub(crate) fn bytes(&self) -> &[u8; KEY_BYTES] {
		&self.bytes
	}
}

impl Clone for EncryptionKey {
	fn clone(&self) -> Self {
		Self::new(self.id.clone(), *self.bytes)
	}
}

impl fmt::Debug for EncryptionKey {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("EncryptionKey")
			.field("id", &self.id)
			.field("material", &"[REDACTED]")
			.finish()
	}
}

/// Failure to obtain persistent encryption key material.
#[derive(Clone, Eq, PartialEq, thiserror::Error)]
pub enum KeyError {
	/// The selected key source is not available in this environment.
	#[error("credential encryption key source is unavailable")]
	Unavailable,
	/// The requested historical key is unavailable.
	#[error("credential encryption key {0} is unavailable")]
	NotFound(KeyId),
	/// A key identifier was reused for different material.
	#[error("credential encryption key identifier {0} is already installed")]
	IdentifierInUse(KeyId),
	/// Stored key material has an invalid length.
	#[error("credential encryption key has an invalid length")]
	InvalidLength,
	/// The operating-system credential facility rejected the operation.
	#[error("operating-system credential facility rejected the key operation")]
	OsCredential,
	/// Secure random generation failed.
	#[error("secure random generation failed")]
	Random,
}

impl fmt::Debug for KeyError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(match self {
			Self::Unavailable => "KeyError::Unavailable",
			Self::NotFound(_) => "KeyError::NotFound",
			Self::IdentifierInUse(_) => "KeyError::IdentifierInUse",
			Self::InvalidLength => "KeyError::InvalidLength",
			Self::OsCredential => "KeyError::OsCredential",
			Self::Random => "KeyError::Random",
		})
	}
}

/// Supplies active and historical keys without exposing their origin to
/// persistence.
pub trait KeySource: Send + Sync {
	/// Loads the active key used for new writes.
	fn active_key(&self) -> Result<EncryptionKey, KeyError>;

	/// Loads a key by the identifier stored beside a ciphertext.
	fn key(&self, id: &KeyId) -> Result<EncryptionKey, KeyError>;
}

/// Explicit key source for headless deployments.
///
/// Callers are responsible for obtaining the bytes from a protected secret
/// injection mechanism. This type never reads an implicit environment value.
pub struct HeadlessKeySource {
	active: RwLock<KeyId>,
	keys:   RwLock<Vec<EncryptionKey>>,
}

impl HeadlessKeySource {
	/// Creates a source containing one active key.
	#[must_use]
	pub fn new(id: KeyId, bytes: [u8; KEY_BYTES]) -> Self {
		Self {
			active: RwLock::new(id.clone()),
			keys:   RwLock::new(vec![EncryptionKey::new(id, bytes)]),
		}
	}

	/// Adds a historical key that can decrypt records written before rotation.
	pub fn try_with_historical(self, id: KeyId, bytes: [u8; KEY_BYTES]) -> Result<Self, KeyError> {
		let mut keys = self.keys.write();
		if keys.iter().any(|key| key.id == id) {
			return Err(KeyError::IdentifierInUse(id));
		}
		keys.push(EncryptionKey::new(id, bytes));
		drop(keys);
		Ok(self)
	}

	/// Installs a new active key while retaining prior keys for atomic rotation.
	pub fn install_active(&self, id: KeyId, bytes: [u8; KEY_BYTES]) -> Result<(), KeyError> {
		let mut keys = self.keys.write();
		if keys.iter().any(|key| key.id == id) {
			return Err(KeyError::IdentifierInUse(id));
		}
		keys.push(EncryptionKey::new(id.clone(), bytes));
		*self.active.write() = id;
		Ok(())
	}
}

impl fmt::Debug for HeadlessKeySource {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		let active = self.active.read().clone();
		let key_count = self.keys.read().len();
		formatter
			.debug_struct("HeadlessKeySource")
			.field("active", &active)
			.field("key_count", &key_count)
			.finish()
	}
}

impl KeySource for HeadlessKeySource {
	fn active_key(&self) -> Result<EncryptionKey, KeyError> {
		let active = self.active.read().clone();
		self.key(&active)
	}

	fn key(&self, id: &KeyId) -> Result<EncryptionKey, KeyError> {
		self
			.keys
			.read()
			.iter()
			.find(|key| key.id == *id)
			.cloned()
			.ok_or_else(|| KeyError::NotFound(id.clone()))
	}
}

/// Key source that deterministically reports that no key is available.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnavailableKeySource;

impl KeySource for UnavailableKeySource {
	fn active_key(&self) -> Result<EncryptionKey, KeyError> {
		Err(KeyError::Unavailable)
	}

	fn key(&self, _id: &KeyId) -> Result<EncryptionKey, KeyError> {
		Err(KeyError::Unavailable)
	}
}

/// Explicitly opted-in encryption-key source backed by the OS credential
/// facility.
///
/// Constructing the source performs no I/O, but [`KeySource::active_key`],
/// [`KeySource::key`], and [`Self::rotate`] may ask the OS credential service
/// to authorize access. Applications must therefore select this source
/// explicitly; it is never a default or fallback. Tests and unattended
/// deployments should use [`HeadlessKeySource`] with injected key bytes
/// instead.
///
/// The implementation is available on macOS, where keys are stored as generic
/// passwords in the user's Keychain. Unsupported targets return
/// [`KeyError::Unavailable`] and never fall back to plaintext or a local file.
#[derive(Clone)]
pub struct OsCredentialKeySource {
	service: Arc<str>,
	account: Arc<str>,
}

impl OsCredentialKeySource {
	/// Creates an explicitly opted-in service/account namespace without
	/// performing I/O.
	#[must_use]
	pub fn new(service: impl Into<Arc<str>>, account: impl Into<Arc<str>>) -> Self {
		Self { service: service.into(), account: account.into() }
	}

	/// Provisions a new active key while retaining historical keys in the OS
	/// facility.
	pub fn rotate(&self) -> Result<KeyId, KeyError> {
		#[cfg(target_os = "macos")]
		{
			use security_framework::passwords::set_generic_password;

			let random = SystemRandom::new();
			let mut id_bytes = [0_u8; 16];
			let mut key_bytes = Zeroizing::new([0_u8; KEY_BYTES]);
			random.fill(&mut id_bytes).map_err(|_| KeyError::Random)?;
			random
				.fill(key_bytes.as_mut())
				.map_err(|_| KeyError::Random)?;
			let encoded = hex::encode_n(&id_bytes);
			let id = KeyId::new(&*encoded);
			set_generic_password(&self.service, &self.key_account(&id), key_bytes.as_ref())
				.map_err(|_| KeyError::OsCredential)?;
			set_generic_password(&self.service, &self.active_account(), id.as_str().as_bytes())
				.map_err(|_| KeyError::OsCredential)?;
			Ok(id)
		}
		#[cfg(not(target_os = "macos"))]
		{
			Err(KeyError::Unavailable)
		}
	}

	fn active_account(&self) -> String {
		format!("{}:active", self.account)
	}

	fn key_account(&self, id: &KeyId) -> String {
		format!("{}:key:{}", self.account, id.as_str())
	}
}

impl fmt::Debug for OsCredentialKeySource {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("OsCredentialKeySource")
			.finish_non_exhaustive()
	}
}

impl KeySource for OsCredentialKeySource {
	fn active_key(&self) -> Result<EncryptionKey, KeyError> {
		#[cfg(target_os = "macos")]
		{
			use security_framework::passwords::get_generic_password;

			let raw = get_generic_password(&self.service, &self.active_account())
				.map_err(|_| KeyError::Unavailable)?;
			let id = str::from_utf8(&raw).map_err(|_| KeyError::InvalidLength)?;
			self.key(&KeyId::new(id))
		}
		#[cfg(not(target_os = "macos"))]
		{
			Err(KeyError::Unavailable)
		}
	}

	fn key(&self, id: &KeyId) -> Result<EncryptionKey, KeyError> {
		#[cfg(target_os = "macos")]
		{
			use security_framework::passwords::get_generic_password;

			let raw = Zeroizing::new(
				get_generic_password(&self.service, &self.key_account(id))
					.map_err(|_| KeyError::NotFound(id.clone()))?,
			);
			let bytes: [u8; KEY_BYTES] = raw
				.as_slice()
				.try_into()
				.map_err(|_| KeyError::InvalidLength)?;
			Ok(EncryptionKey::new(id.clone(), bytes))
		}
		#[cfg(not(target_os = "macos"))]
		{
			let _ = id;
			Err(KeyError::Unavailable)
		}
	}
}
