//! TLS configuration builders for remote TCP gateways.

use std::path::{Path, PathBuf};

use tonic::transport::{Certificate, ClientTlsConfig, Identity, ServerTlsConfig};

use crate::Error;

/// PEM files used by a gateway TLS server.
#[derive(Clone, Debug)]
pub struct TlsConfig {
	/// Server certificate chain in PEM format.
	pub cert:      PathBuf,
	/// Server private key in PEM format.
	pub key:       PathBuf,
	/// Optional client certificate-authority bundle; setting it enables
	/// mandatory mTLS.
	pub client_ca: Option<PathBuf>,
}

/// Load a server identity and, when configured, require certificates signed by
/// the client CA.
pub async fn server_tls(cfg: &TlsConfig) -> Result<ServerTlsConfig, Error> {
	let (cert, key) = tokio::try_join!(tokio::fs::read(&cfg.cert), tokio::fs::read(&cfg.key))?;
	let mut tls = ServerTlsConfig::new().identity(Identity::from_pem(cert, key));
	if let Some(client_ca) = &cfg.client_ca {
		let pem = tokio::fs::read(client_ca).await?;
		tls = tls
			.client_ca_root(Certificate::from_pem(pem))
			.client_auth_optional(false);
	}
	Ok(tls)
}

/// Load a server CA and optional client identity for a remote gateway
/// connection.
pub async fn client_tls(
	ca: &Path,
	domain: &str,
	identity: Option<(&Path, &Path)>,
) -> Result<ClientTlsConfig, Error> {
	let ca = Certificate::from_pem(tokio::fs::read(ca).await?);
	let mut tls = ClientTlsConfig::new()
		.ca_certificate(ca)
		.domain_name(domain);
	if let Some((cert, key)) = identity {
		let (cert, key) = tokio::try_join!(tokio::fs::read(cert), tokio::fs::read(key))?;
		tls = tls.identity(Identity::from_pem(cert, key));
	}
	Ok(tls)
}
