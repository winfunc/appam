//! Authentication helpers for the OpenAI Codex subscription provider.
//!
//! The Codex backend accepts ChatGPT OAuth access tokens rather than Platform
//! API keys. Appam therefore stores refreshable ChatGPT credentials in a local
//! file cache and refreshes them under a file lock before expiry.
//!
//! # Security model
//!
//! - credentials are stored only on the local filesystem
//! - the cache file is created with `0600` permissions on Unix platforms
//! - refresh operations are serialized with an exclusive file lock
//! - access and refresh tokens are never logged
//!
//! This module is intended for trusted local developer environments. Do not
//! copy the cache file into public or multi-tenant environments.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use base64::engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// OAuth client ID used for ChatGPT Codex browser login.
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
/// Authorization endpoint for ChatGPT Codex OAuth.
const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
/// Token endpoint for ChatGPT Codex OAuth.
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
/// Fixed redirect URI used by the local callback helper.
const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
/// Scope set required for refreshable ChatGPT Codex credentials.
const SCOPE: &str = "openid profile email offline_access";
/// JWT claim path that carries the ChatGPT account identifier.
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
/// Provider key stored inside the auth cache.
const PROVIDER_KEY: &str = "openai-codex";
/// Pre-expiry refresh buffer to avoid racing near-expired tokens.
const TOKEN_REFRESH_SKEW_MS: u64 = 60_000;

/// Fully resolved ChatGPT OAuth credentials for the Codex provider.
///
/// The `account_id` field is derived from the access token and is required by
/// the Codex backend request headers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAICodexCredentials {
    /// Bearer access token used for Codex backend requests.
    pub access: String,
    /// Refresh token used to mint a new access token.
    pub refresh: String,
    /// Expiration timestamp in Unix milliseconds.
    pub expires: u64,
    /// ChatGPT account identifier extracted from the access-token JWT.
    pub account_id: String,
}

impl OpenAICodexCredentials {
    /// Returns whether the access token should be refreshed before use.
    pub fn is_expired(&self) -> bool {
        let threshold = self.expires.saturating_sub(TOKEN_REFRESH_SKEW_MS);
        current_time_millis() >= threshold
    }
}

/// Authentication source used for a resolved Codex access token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAICodexAuthSource {
    /// The token came from explicit configuration or an environment override.
    ConfiguredToken,
    /// The token came from Appam's local OAuth credential cache.
    CachedOAuth,
}

/// Resolved bearer token and account metadata for a Codex request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedOpenAICodexAuth {
    /// Bearer token to send in the `Authorization` header.
    pub access_token: String,
    /// ChatGPT account identifier to send in `chatgpt-account-id`.
    pub account_id: String,
    /// Where the token came from.
    pub source: OpenAICodexAuthSource,
}

/// Disk-backed storage for OpenAI Codex OAuth credentials.
#[derive(Debug, Clone)]
pub struct OpenAICodexAuthStorage {
    path: PathBuf,
}

impl OpenAICodexAuthStorage {
    /// Create a storage helper for the given auth-cache path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Return the path of the auth-cache file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns whether the auth cache contains an OpenAI Codex entry.
    ///
    /// This is intentionally shallow and does not validate or refresh the token.
    pub fn has_cached_entry(&self) -> Result<bool> {
        let mut file = self.open_locked_file()?;
        let data = read_auth_file(&mut file)?;
        Ok(data.contains_key(PROVIDER_KEY))
    }

    /// Load the raw cached credentials without refreshing them.
    pub fn load_credentials(&self) -> Result<Option<OpenAICodexCredentials>> {
        let mut file = self.open_locked_file()?;
        let data = read_auth_file(&mut file)?;
        Ok(data.get(PROVIDER_KEY).cloned().map(Into::into))
    }

    /// Store or replace cached Codex credentials.
    pub fn store_credentials(&self, credentials: &OpenAICodexCredentials) -> Result<()> {
        let mut file = self.open_locked_file()?;
        let mut data = read_auth_file(&mut file)?;
        data.insert(
            PROVIDER_KEY.to_string(),
            StoredCredential::from(credentials.clone()),
        );
        write_auth_file(&mut file, &data)
    }

    /// Load cached credentials and refresh them if needed.
    pub async fn resolve_credentials(&self) -> Result<Option<OpenAICodexCredentials>> {
        self.resolve_credentials_with_refresh(refresh_openai_codex_token)
            .await
    }

    async fn resolve_credentials_with_refresh<F, Fut>(
        &self,
        refresh_fn: F,
    ) -> Result<Option<OpenAICodexCredentials>>
    where
        F: Fn(String) -> Fut,
        Fut: std::future::Future<Output = Result<OpenAICodexCredentials>>,
    {
        let mut file = self.open_locked_file()?;
        let mut data = read_auth_file(&mut file)?;
        let Some(existing) = data
            .get(PROVIDER_KEY)
            .cloned()
            .map(OpenAICodexCredentials::from)
        else {
            return Ok(None);
        };

        if !existing.is_expired() {
            debug!(
                auth_path = %self.path.display(),
                "Using cached OpenAI Codex credentials without refresh"
            );
            return Ok(Some(existing));
        }

        info!(
            auth_path = %self.path.display(),
            "Refreshing expired OpenAI Codex credentials"
        );

        let refreshed = refresh_fn(existing.refresh.clone()).await?;
        data.insert(
            PROVIDER_KEY.to_string(),
            StoredCredential::from(refreshed.clone()),
        );
        write_auth_file(&mut file, &data)?;
        Ok(Some(refreshed))
    }

    fn open_locked_file(&self) -> Result<File> {
        ensure_parent_dir(&self.path)?;

        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        let file = options
            .open(&self.path)
            .with_context(|| format!("Failed to open auth file {}", self.path.display()))?;

        file.lock()
            .with_context(|| format!("Failed to lock auth file {}", self.path.display()))?;
        ensure_file_permissions(&file)?;
        Ok(file)
    }
}

/// Resolve Codex authentication for a request.
///
/// # Resolution order
///
/// 1. explicit `configured_access_token`
/// 2. `OPENAI_CODEX_ACCESS_TOKEN`
/// 3. cached OAuth credentials in `auth_file`
pub async fn resolve_openai_codex_auth(
    configured_access_token: Option<&str>,
    auth_file: &Path,
) -> Result<ResolvedOpenAICodexAuth> {
    if let Some(token) = configured_access_token.filter(|token| !token.trim().is_empty()) {
        let account_id = extract_account_id(token)
            .ok_or_else(|| anyhow!("Failed to extract ChatGPT account ID from configured token"))?;
        return Ok(ResolvedOpenAICodexAuth {
            access_token: token.to_string(),
            account_id,
            source: OpenAICodexAuthSource::ConfiguredToken,
        });
    }

    if let Ok(token) = std::env::var("OPENAI_CODEX_ACCESS_TOKEN") {
        if !token.trim().is_empty() {
            let account_id = extract_account_id(&token).ok_or_else(|| {
                anyhow!("Failed to extract ChatGPT account ID from OPENAI_CODEX_ACCESS_TOKEN")
            })?;
            return Ok(ResolvedOpenAICodexAuth {
                access_token: token,
                account_id,
                source: OpenAICodexAuthSource::ConfiguredToken,
            });
        }
    }

    let storage = OpenAICodexAuthStorage::new(auth_file.to_path_buf());
    if let Some(credentials) = storage.resolve_credentials().await? {
        return Ok(ResolvedOpenAICodexAuth {
            access_token: credentials.access,
            account_id: credentials.account_id,
            source: OpenAICodexAuthSource::CachedOAuth,
        });
    }

    bail!(
        "Missing OpenAI Codex credentials. Set OPENAI_CODEX_ACCESS_TOKEN or authenticate into {}",
        auth_file.display()
    );
}

/// Run an interactive browser login for OpenAI Codex and persist the result.
///
/// This helper is intended for local developer workflows such as the example
/// binary shipped with Appam. It attempts to open the authorization URL in the
/// system browser, waits for the localhost callback, and falls back to a manual
/// paste prompt when needed.
///
/// # Security considerations
///
/// - The returned credentials are persisted to the provided `storage`.
/// - The user is prompted only for the authorization code or redirect URL.
/// - Access and refresh tokens are never echoed to stdout.
pub async fn login_openai_codex_interactive(
    storage: &OpenAICodexAuthStorage,
    originator: &str,
) -> Result<OpenAICodexCredentials> {
    let request = create_authorization_request(originator)?;
    let callback_server = LocalCallbackServer::start(&request.state).await?;

    println!("OpenAI Codex login required.");
    println!("Auth file: {}", storage.path().display());
    println!("Opening browser for ChatGPT authentication...");
    if !try_open_browser(&request.url) {
        println!("Browser auto-open failed. Open this URL manually:");
    }
    println!("{}", request.url);
    println!(
        "If the browser does not return to this terminal automatically, paste the full redirect URL or authorization code when prompted."
    );

    let callback_code = if callback_server.callback_available {
        match timeout(Duration::from_secs(120), callback_server.wait_for_code()).await {
            Ok(result) => result?,
            Err(_) => {
                warn!("Timed out waiting for OpenAI Codex localhost callback");
                None
            }
        }
    } else {
        None
    };

    let code = if let Some(code) = callback_code {
        code
    } else {
        prompt_for_authorization_code(&request.state)?
    };

    let credentials = exchange_authorization_code(&code, &request.verifier).await?;
    storage.store_credentials(&credentials)?;
    println!("OpenAI Codex authentication completed.");
    Ok(credentials)
}

/// Refresh OpenAI Codex OAuth credentials using the refresh token.
pub async fn refresh_openai_codex_token(refresh_token: String) -> Result<OpenAICodexCredentials> {
    let refresh_token_value = refresh_token.clone();
    let params = [
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token_value.as_str()),
        ("client_id", CLIENT_ID),
    ];

    exchange_token_form(&params, Some(refresh_token)).await
}

/// Return Appam's default auth-cache path.
pub fn default_auth_file_path() -> PathBuf {
    default_appam_dir().join("auth.json")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredCredential {
    #[serde(rename = "type")]
    type_field: String,
    access: String,
    refresh: String,
    expires: u64,
    account_id: String,
}

impl From<OpenAICodexCredentials> for StoredCredential {
    fn from(value: OpenAICodexCredentials) -> Self {
        Self {
            type_field: "oauth".to_string(),
            access: value.access,
            refresh: value.refresh,
            expires: value.expires,
            account_id: value.account_id,
        }
    }
}

impl From<StoredCredential> for OpenAICodexCredentials {
    fn from(value: StoredCredential) -> Self {
        Self {
            access: value.access,
            refresh: value.refresh,
            expires: value.expires,
            account_id: value.account_id,
        }
    }
}

#[derive(Debug)]
struct AuthorizationRequest {
    url: String,
    verifier: String,
    state: String,
}

#[derive(Debug)]
struct LocalCallbackServer {
    callback_available: bool,
    join_handle: Option<tokio::task::JoinHandle<Result<Option<String>>>>,
}

impl LocalCallbackServer {
    async fn start(expected_state: &str) -> Result<Self> {
        let listener = match tokio::net::TcpListener::bind(("127.0.0.1", 1455)).await {
            Ok(listener) => listener,
            Err(error) => {
                warn!(
                    error = %error,
                    "Failed to bind OpenAI Codex localhost callback listener; falling back to manual code input"
                );
                return Ok(Self {
                    callback_available: false,
                    join_handle: None,
                });
            }
        };

        let expected_state = expected_state.to_string();
        let join_handle = tokio::spawn(async move {
            let (mut stream, _) = listener
                .accept()
                .await
                .context("Failed to accept OpenAI Codex callback connection")?;

            let mut buffer = vec![0_u8; 8192];
            let bytes_read = tokio::io::AsyncReadExt::read(&mut stream, &mut buffer)
                .await
                .context("Failed to read OpenAI Codex callback request")?;
            let request = String::from_utf8_lossy(&buffer[..bytes_read]).to_string();
            let path = request
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .ok_or_else(|| anyhow!("Malformed OpenAI Codex callback request"))?;

            let url = format!("http://127.0.0.1{path}");
            let callback_url =
                url::Url::parse(&url).with_context(|| format!("Invalid callback URL {}", url))?;
            let code = callback_url
                .query_pairs()
                .find_map(|(key, value)| (key == "code").then(|| value.into_owned()));
            let state = callback_url
                .query_pairs()
                .find_map(|(key, value)| (key == "state").then(|| value.into_owned()));

            let (status_line, body) = match (state, code) {
                (Some(state), Some(_code)) if state == expected_state => (
                    "HTTP/1.1 200 OK",
                    "<html><body><h1>OpenAI authentication completed.</h1><p>You can close this window.</p></body></html>",
                ),
                (Some(_), _) => (
                    "HTTP/1.1 400 Bad Request",
                    "<html><body><h1>State mismatch.</h1></body></html>",
                ),
                _ => (
                    "HTTP/1.1 400 Bad Request",
                    "<html><body><h1>Missing authorization code.</h1></body></html>",
                ),
            };

            let response = format!(
                "{status_line}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            tokio::io::AsyncWriteExt::write_all(&mut stream, response.as_bytes())
                .await
                .context("Failed to write OpenAI Codex callback response")?;
            tokio::io::AsyncWriteExt::flush(&mut stream)
                .await
                .context("Failed to flush OpenAI Codex callback response")?;

            Ok(callback_url
                .query_pairs()
                .find_map(|(key, value)| (key == "code").then(|| value.into_owned())))
        });

        Ok(Self {
            callback_available: true,
            join_handle: Some(join_handle),
        })
    }

    async fn wait_for_code(self) -> Result<Option<String>> {
        let Some(join_handle) = self.join_handle else {
            return Ok(None);
        };

        join_handle
            .await
            .context("OpenAI Codex callback task failed to join")?
    }
}

fn create_authorization_request(originator: &str) -> Result<AuthorizationRequest> {
    let verifier = generate_pkce_verifier();
    let challenge = generate_pkce_challenge(&verifier);
    let state = generate_state();

    let mut url = url::Url::parse(AUTHORIZE_URL).context("Failed to parse OpenAI authorize URL")?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPE)
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", &state)
        .append_pair("id_token_add_organizations", "true")
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("originator", originator);

    Ok(AuthorizationRequest {
        url: url.to_string(),
        verifier,
        state,
    })
}

async fn exchange_authorization_code(code: &str, verifier: &str) -> Result<OpenAICodexCredentials> {
    let params = [
        ("grant_type", "authorization_code"),
        ("client_id", CLIENT_ID),
        ("code", code),
        ("code_verifier", verifier),
        ("redirect_uri", REDIRECT_URI),
    ];

    exchange_token_form(&params, None).await
}

async fn exchange_token_form(
    params: &[(&str, &str)],
    fallback_refresh_token: Option<String>,
) -> Result<OpenAICodexCredentials> {
    let encoded_body = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(params.iter().copied())
        .finish();

    let response = reqwest::Client::new()
        .post(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(encoded_body)
        .send()
        .await
        .context("OpenAI Codex token exchange request failed")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("OpenAI Codex token exchange failed ({}): {}", status, body);
    }

    let payload: TokenResponse = response
        .json()
        .await
        .context("Failed to parse OpenAI Codex token response")?;

    let access = payload
        .access_token
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("OpenAI Codex token response missing access_token"))?;
    let refresh = payload
        .refresh_token
        .filter(|value| !value.is_empty())
        .or(fallback_refresh_token)
        .ok_or_else(|| anyhow!("OpenAI Codex token response missing refresh_token"))?;
    let expires_in = payload
        .expires_in
        .ok_or_else(|| anyhow!("OpenAI Codex token response missing expires_in"))?;
    let account_id = extract_account_id(&access)
        .ok_or_else(|| anyhow!("Failed to extract ChatGPT account ID from access token"))?;

    Ok(OpenAICodexCredentials {
        access,
        refresh,
        expires: current_time_millis().saturating_add((expires_in as u64) * 1000),
        account_id,
    })
}

fn prompt_for_authorization_code(expected_state: &str) -> Result<String> {
    print!("Paste the OpenAI Codex redirect URL or authorization code: ");
    std::io::stdout()
        .flush()
        .context("Failed to flush authorization prompt")?;

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("Failed to read authorization code input")?;

    let parsed = parse_authorization_input(&input);
    if let Some(state) = parsed.state.as_deref() {
        if state != expected_state {
            bail!("Authorization state mismatch");
        }
    }
    if let Some(code) = parsed.code {
        return Ok(code);
    }

    bail!("No authorization code was provided")
}

fn parse_authorization_input(input: &str) -> ParsedAuthorizationInput {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return ParsedAuthorizationInput::default();
    }

    if let Ok(url) = url::Url::parse(trimmed) {
        return ParsedAuthorizationInput {
            code: url
                .query_pairs()
                .find_map(|(key, value)| (key == "code").then(|| value.into_owned())),
            state: url
                .query_pairs()
                .find_map(|(key, value)| (key == "state").then(|| value.into_owned())),
        };
    }

    if trimmed.contains("code=") {
        let params = url::form_urlencoded::parse(trimmed.as_bytes())
            .into_owned()
            .collect::<Vec<_>>();
        return ParsedAuthorizationInput {
            code: params
                .iter()
                .find_map(|(key, value)| (key == "code").then(|| value.clone())),
            state: params
                .iter()
                .find_map(|(key, value)| (key == "state").then(|| value.clone())),
        };
    }

    if let Some((code, state)) = trimmed.split_once('#') {
        return ParsedAuthorizationInput {
            code: (!code.trim().is_empty()).then(|| code.trim().to_string()),
            state: (!state.trim().is_empty()).then(|| state.trim().to_string()),
        };
    }

    ParsedAuthorizationInput {
        code: Some(trimmed.to_string()),
        state: None,
    }
}

fn generate_pkce_verifier() -> String {
    let bytes: [u8; 32] = rand::random();
    URL_SAFE_NO_PAD.encode(bytes)
}

fn generate_pkce_challenge(verifier: &str) -> String {
    let hash = Sha256::digest(verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(hash)
}

fn generate_state() -> String {
    let bytes: [u8; 16] = rand::random();
    hex::encode(bytes)
}

fn try_open_browser(url: &str) -> bool {
    let attempts: &[(&str, &[&str])] = if cfg!(target_os = "macos") {
        &[("open", &[])]
    } else if cfg!(target_os = "windows") {
        &[("cmd", &["/C", "start", "", url])]
    } else {
        &[("xdg-open", &[]), ("gio", &["open"])]
    };

    attempts.iter().any(|(program, fixed_args)| {
        let mut command = Command::new(program);
        command.args(*fixed_args);
        if !cfg!(target_os = "windows") {
            command.arg(url);
        }
        command
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    })
}

fn current_time_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn default_appam_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .or_else(
            || match (std::env::var_os("HOMEDRIVE"), std::env::var_os("HOMEPATH")) {
                (Some(drive), Some(path)) => {
                    let mut value = PathBuf::from(drive);
                    value.push(path);
                    Some(value.into_os_string())
                }
                _ => None,
            },
        )
        .unwrap_or_else(|| ".".into());

    PathBuf::from(home).join(".appam")
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("Auth file path {} has no parent directory", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create auth directory {}", parent.display()))?;
    ensure_directory_permissions(parent)?;
    Ok(())
}

fn ensure_directory_permissions(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let permissions = fs::Permissions::from_mode(0o700);
        fs::set_permissions(path, permissions).with_context(|| {
            format!(
                "Failed to set permissions on auth directory {}",
                path.display()
            )
        })?;
    }

    #[cfg(not(unix))]
    {
        let _ = path;
    }

    Ok(())
}

fn ensure_file_permissions(file: &File) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        file.set_permissions(fs::Permissions::from_mode(0o600))
            .context("Failed to set auth file permissions to 0600")?;
    }

    #[cfg(not(unix))]
    {
        let _ = file;
    }

    Ok(())
}

type AuthFileData = std::collections::BTreeMap<String, StoredCredential>;

fn read_auth_file(file: &mut File) -> Result<AuthFileData> {
    file.seek(SeekFrom::Start(0))
        .context("Failed to rewind auth file before read")?;

    let mut content = String::new();
    file.read_to_string(&mut content)
        .context("Failed to read auth file")?;
    if content.trim().is_empty() {
        return Ok(AuthFileData::new());
    }

    serde_json::from_str(&content).context("Invalid JSON in OpenAI Codex auth file")
}

fn write_auth_file(file: &mut File, data: &AuthFileData) -> Result<()> {
    let serialized =
        serde_json::to_string_pretty(data).context("Failed to serialize auth file contents")?;
    file.set_len(0).context("Failed to truncate auth file")?;
    file.seek(SeekFrom::Start(0))
        .context("Failed to rewind auth file before write")?;
    file.write_all(serialized.as_bytes())
        .context("Failed to write auth file")?;
    file.flush().context("Failed to flush auth file")?;
    ensure_file_permissions(file)?;
    Ok(())
}

/// Extract the ChatGPT account identifier from a JWT access token.
pub fn extract_account_id(access_token: &str) -> Option<String> {
    let payload = decode_jwt_payload(access_token)?;
    payload
        .get(JWT_CLAIM_PATH)?
        .get("chatgpt_account_id")?
        .as_str()
        .map(ToString::to_string)
}

fn decode_jwt_payload(token: &str) -> Option<Value> {
    let payload_segment = token.split('.').nth(1)?;
    let decoded = URL_SAFE_NO_PAD
        .decode(payload_segment)
        .or_else(|_| URL_SAFE.decode(payload_segment))
        .ok()?;
    serde_json::from_slice(&decoded).ok()
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_in: Option<u32>,
}

#[derive(Debug, Default)]
struct ParsedAuthorizationInput {
    code: Option<String>,
    state: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn mock_token(account_id: &str) -> String {
        let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"none"}"#);
        let payload = URL_SAFE_NO_PAD.encode(
            format!(r#"{{"{JWT_CLAIM_PATH}":{{"chatgpt_account_id":"{account_id}"}}}}"#).as_bytes(),
        );
        format!("{header}.{payload}.signature")
    }

    #[test]
    fn test_extract_account_id() {
        assert_eq!(
            extract_account_id(&mock_token("acc_test")).as_deref(),
            Some("acc_test")
        );
        assert!(extract_account_id("not-a-jwt").is_none());
    }

    #[test]
    fn test_storage_round_trip() {
        let temp_dir = tempdir().unwrap();
        let storage = OpenAICodexAuthStorage::new(temp_dir.path().join("auth.json"));
        let credentials = OpenAICodexCredentials {
            access: mock_token("acc_roundtrip"),
            refresh: "refresh_token".to_string(),
            expires: current_time_millis() + 60_000,
            account_id: "acc_roundtrip".to_string(),
        };

        storage.store_credentials(&credentials).unwrap();
        let loaded = storage.load_credentials().unwrap().unwrap();
        assert_eq!(loaded, credentials);
    }

    #[test]
    fn test_storage_reports_cached_entry() {
        let temp_dir = tempdir().unwrap();
        let storage = OpenAICodexAuthStorage::new(temp_dir.path().join("auth.json"));
        assert!(!storage.has_cached_entry().unwrap());

        storage
            .store_credentials(&OpenAICodexCredentials {
                access: mock_token("acc_exists"),
                refresh: "refresh".to_string(),
                expires: current_time_millis() + 60_000,
                account_id: "acc_exists".to_string(),
            })
            .unwrap();
        assert!(storage.has_cached_entry().unwrap());
    }

    #[tokio::test]
    async fn test_storage_refreshes_expired_credentials() {
        let temp_dir = tempdir().unwrap();
        let storage = OpenAICodexAuthStorage::new(temp_dir.path().join("auth.json"));
        storage
            .store_credentials(&OpenAICodexCredentials {
                access: mock_token("acc_old"),
                refresh: "refresh_old".to_string(),
                expires: current_time_millis().saturating_sub(1),
                account_id: "acc_old".to_string(),
            })
            .unwrap();

        let refreshed = storage
            .resolve_credentials_with_refresh(|refresh| async move {
                assert_eq!(refresh, "refresh_old");
                Ok(OpenAICodexCredentials {
                    access: mock_token("acc_new"),
                    refresh: "refresh_new".to_string(),
                    expires: current_time_millis() + 60_000,
                    account_id: "acc_new".to_string(),
                })
            })
            .await
            .unwrap()
            .unwrap();

        assert_eq!(refreshed.account_id, "acc_new");
        let stored = storage.load_credentials().unwrap().unwrap();
        assert_eq!(stored.account_id, "acc_new");
        assert_eq!(stored.refresh, "refresh_new");
    }

    #[test]
    fn test_resolve_openai_codex_auth_prefers_configured_token() {
        let temp_dir = tempdir().unwrap();
        let token = mock_token("acc_config");
        let resolved = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(resolve_openai_codex_auth(
                Some(&token),
                &temp_dir.path().join("auth.json"),
            ));

        let resolved = resolved.unwrap();
        assert_eq!(resolved.account_id, "acc_config");
        assert_eq!(resolved.source, OpenAICodexAuthSource::ConfiguredToken);
    }

    #[test]
    fn test_parse_authorization_input_accepts_redirect_url() {
        let parsed =
            parse_authorization_input("http://127.0.0.1:1455/auth/callback?code=abc&state=xyz");
        assert_eq!(parsed.code.as_deref(), Some("abc"));
        assert_eq!(parsed.state.as_deref(), Some("xyz"));
    }

    #[test]
    fn test_auth_file_default_path_ends_in_appam_auth_json() {
        assert!(default_auth_file_path()
            .to_string_lossy()
            .ends_with(".appam/auth.json"));
    }
}
