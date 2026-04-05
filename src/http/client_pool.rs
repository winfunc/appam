//! HTTP client registry for connection reuse.
//!
//! ## Purpose
//! Maintains a process-wide cache of pre-configured `reqwest::Client` instances
//! keyed by scheme/host/port. Reusing clients enables HTTP/2 multiplexing and
//! prevents spawning thousands of independent connection pools when the agents
//! fan out hundreds of requests per second.
//!
//! ## Design Notes
//! - Uses a `Mutex<HashMap<..>>` because client construction is infrequent and
//!   contention is negligible compared to network latency.
//! - Performs best-effort DNS resolution up front to populate the client's
//!   internal resolver cache. Failures fall back to the system resolver so
//!   degraded environments still function.
//! - Never logs credentials; only hostnames and counts are recorded for
//!   debugging.
//!
//! ## Security
//! All inputs are treated as untrusted. Hostnames originate from configuration
//! files or environment variables, so they are normalized and parsed through
//! `reqwest::Url` before use. DNS lookups use the OS resolver and we avoid
//! caching results beyond the lifetime of the process-wide client to honour DNS
//! TTL semantics.

use anyhow::{anyhow, Context, Result};
use once_cell::sync::Lazy;
use reqwest::{Client, Url};
use std::{
    collections::HashMap,
    net::{SocketAddr, ToSocketAddrs},
    sync::Mutex,
};
use tracing::{debug, warn};

/// Global cache of HTTP clients keyed by scheme/host/port.
///
/// The value is a fully configured `reqwest::Client`. Cloning a client is cheap
/// because it internally uses an `Arc`, so we store a single canonical instance
/// per key and hand out clones to callers.
static CLIENTS: Lazy<Mutex<HashMap<HttpClientKey, Client>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Describes a unique HTTP endpoint (scheme + host + port).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct HttpClientKey {
    scheme: String,
    host: String,
    port: u16,
}

impl HttpClientKey {
    /// Derive a key from a base URL.
    fn from_url(url: &str) -> Result<Self> {
        let parsed = Url::parse(url)
            .with_context(|| format!("Failed to parse base URL for HTTP client: {url}"))?;

        let host = parsed
            .host_str()
            .ok_or_else(|| anyhow!("Base URL is missing host: {url}"))?;

        let scheme = parsed.scheme();

        let port = parsed.port_or_known_default().ok_or_else(|| {
            anyhow!(
                "Unable to determine port for {}://{} – specify an explicit port",
                scheme,
                host
            )
        })?;

        Ok(Self {
            scheme: scheme.to_string(),
            host: host.to_string(),
            port,
        })
    }

    /// Hostname component.
    fn host(&self) -> &str {
        &self.host
    }
}

/// Context passed to the client builder closure.
///
/// Provides access to the resolved endpoint information and any DNS hint
/// gathered during pre-resolution.
pub struct HttpClientContext<'a> {
    key: &'a HttpClientKey,
    resolved_addrs: Option<Vec<SocketAddr>>,
}

impl<'a> HttpClientContext<'a> {
    /// Scheme (e.g., `https`).
    pub fn scheme(&self) -> &str {
        &self.key.scheme
    }

    /// Hostname portion of the URL.
    pub fn host(&self) -> &str {
        self.key.host()
    }

    /// Network port.
    pub fn port(&self) -> u16 {
        self.key.port
    }

    /// Optional slice of DNS-resolved socket addresses.
    pub fn resolved_addrs(&self) -> Option<&[SocketAddr]> {
        self.resolved_addrs.as_deref()
    }
}

/// Retrieve (or lazily build) a shared HTTP client for the given base URL.
///
/// # Parameters
/// - `base_url`: Provider base URL used to determine scheme/host/port.
/// - `builder`: Closure that receives [`HttpClientContext`] with optional DNS
///   hints and must return a fully configured `reqwest::Client`.
///
/// # Returns
/// A clone of the cached `reqwest::Client`.
///
/// # Errors
/// - Returns an error if the base URL is invalid.
/// - Returns an error if the builder closure fails to construct the client.
pub fn get_or_init_client<F>(base_url: &str, builder: F) -> Result<Client>
where
    F: FnOnce(&HttpClientContext) -> Result<Client>,
{
    let key = HttpClientKey::from_url(base_url)?;

    if let Some(existing) = CLIENTS.lock().unwrap().get(&key).cloned() {
        return Ok(existing);
    }

    let resolved = maybe_resolve_host(&key);
    let context = HttpClientContext {
        key: &key,
        resolved_addrs: resolved,
    };

    let client = builder(&context)?;

    let mut guard = CLIENTS.lock().unwrap();
    let entry = guard.entry(key).or_insert_with(|| client.clone());
    Ok(entry.clone())
}

/// Attempt to resolve the host upfront so we can seed reqwest's DNS cache.
///
/// The operation is best-effort; on failure we log a warning and fall back to
/// runtime resolution within hyper's connector.
fn maybe_resolve_host(key: &HttpClientKey) -> Option<Vec<SocketAddr>> {
    let target = format!("{}:{}", key.host(), key.port);
    match target.to_socket_addrs() {
        Ok(iter) => {
            let addrs: Vec<_> = iter.collect();
            if addrs.is_empty() {
                warn!(
                    host = key.host(),
                    scheme = key.scheme,
                    port = key.port,
                    "DNS lookup returned no addresses; falling back to runtime resolution"
                );
                None
            } else {
                debug!(
                    host = key.host(),
                    scheme = key.scheme,
                    port = key.port,
                    addr_count = addrs.len(),
                    "Cached DNS resolution succeeded"
                );
                Some(addrs)
            }
        }
        Err(err) => {
            warn!(
                host = key.host(),
                scheme = key.scheme,
                port = key.port,
                error = %err,
                "Failed to pre-resolve DNS; falling back to runtime resolution"
            );
            None
        }
    }
}
