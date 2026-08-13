//! Platform-local OMP daemon endpoint.

use std::{
	fmt,
	path::{Path, PathBuf},
	str::FromStr,
};

use omp_core::Str;

/// Owner-local RPC endpoint represented by a Unix socket path or Windows pipe
/// name.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalEndpoint(PathBuf);

impl LocalEndpoint {
	/// Borrows the operating-system endpoint path.
	#[must_use]
	pub fn as_path(&self) -> &Path {
		&self.0
	}
}

impl From<PathBuf> for LocalEndpoint {
	fn from(path: PathBuf) -> Self {
		Self(path)
	}
}

impl From<LocalEndpoint> for PathBuf {
	fn from(endpoint: LocalEndpoint) -> Self {
		endpoint.0
	}
}

impl fmt::Display for LocalEndpoint {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.0.display().fmt(formatter)
	}
}

/// A local endpoint cannot be empty.
#[derive(Clone, Debug, thiserror::Error)]
#[error("local OMP endpoint cannot be empty")]
pub struct EndpointParseError;

impl FromStr for LocalEndpoint {
	type Err = EndpointParseError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		if value.is_empty() {
			return Err(EndpointParseError);
		}
		if let Some(name) = value.strip_prefix("npipe://./pipe/") {
			#[cfg(windows)]
			return Ok(Self(PathBuf::from(format!(r"\\.\pipe\{name}"))));
			#[cfg(not(windows))]
			return Ok(Self(PathBuf::from(name)));
		}
		Ok(Self(PathBuf::from(value)))
	}
}

impl From<&Path> for LocalEndpoint {
	fn from(path: &Path) -> Self {
		Self(path.to_owned())
	}
}

impl From<Str> for LocalEndpoint {
	fn from(value: Str) -> Self {
		Self(PathBuf::from(value.as_str()))
	}
}
