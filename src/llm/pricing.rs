//! LLM pricing calculation, models.dev synchronization, and model cost tracking.
//!
//! Appam prices requests by loading a checked-in models.dev seed snapshot,
//! optionally overlaying a persisted cache from disk, and then attempting a
//! short blocking refresh from `https://models.dev/api.json` during pricing
//! initialization. This keeps startup deterministic for offline workflows while
//! still letting pricing converge to the latest upstream data when the network
//! is available.
//!
//! # Source Priority
//!
//! Pricing data is resolved in the following order:
//!
//! 1. Remote models.dev sync, when the endpoint is reachable within the timeout
//! 2. Persisted cache at `data/pricing/models.dev.json`
//! 3. Embedded seed snapshot committed with the crate
//! 4. Hardcoded default fallback pricing for unknown models
//!
//! # Compatibility
//!
//! The public pricing API remains synchronous and reference-based, so callers do
//! not need to manage async initialization. To preserve backwards
//! compatibility, Appam normalizes several legacy model identifiers, including
//! `openai/<model>` direct-model inputs, AWS Bedrock Anthropic model IDs, and
//! legacy Appam aliases such as `claude-haiku-4-20250514`.
//!
//! # Current Limitations
//!
//! **OpenRouter Usage Tracking**: OpenRouter's Chat Completions and Responses APIs
//! do not currently provide token usage data in their streaming responses. As a
//! result, OpenRouter requests can still report zero live tokens/cost in
//! real-time displays even though the pricing tables themselves are synced and
//! accurate.

use super::UnifiedUsage;
use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use once_cell::sync::Lazy;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    thread,
    time::Duration,
};
use tracing::{debug, info, warn};

const MODELS_DEV_API_URL: &str = "https://models.dev/api.json";
const PRICING_CACHE_PATH: &str = "data/pricing/models.dev.json";
const PRICING_SNAPSHOT_VERSION: u32 = 1;
const TIER_THRESHOLD_TOKENS: u32 = 200_000;

/// Global pricing database loaded from the embedded seed, optional cache, and
/// an optional models.dev refresh.
static PRICING: Lazy<PricingStore> = Lazy::new(PricingStore::initialize);

/// Pricing configuration for a single model.
///
/// Stores per-million-token rates in USD. The structure intentionally preserves
/// Appam's existing public surface even though the upstream models.dev schema
/// represents extended-context pricing under `cost.context_over_200k`.
///
/// # Design Notes
///
/// Appam keeps flat fields (`input`, `output`, `cache_*`, `reasoning`) for
/// simple models and maps models.dev's extended-context pricing into the
/// existing `*_base` / `*_extended` fields. This avoids a breaking API change
/// while still letting the runtime represent tiered pricing from upstream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Human-readable model name.
    pub name: String,

    /// Input token price per million tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<f64>,

    /// Output token price per million tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<f64>,

    /// Cache write (creation) price per million tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,

    /// Cache read price per million tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,

    /// Reasoning token price per million tokens.
    ///
    /// When this field is absent, Appam bills reasoning tokens at the selected
    /// output rate. This matches the historic behaviour of the pricing module
    /// and is required because models.dev omits `cost.reasoning` for many
    /// reasoning-capable models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<f64>,

    /// Base input price for prompts at or below `threshold_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_base: Option<f64>,

    /// Extended input price for prompts above `threshold_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_extended: Option<f64>,

    /// Base output price for prompts at or below `threshold_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_base: Option<f64>,

    /// Extended output price for prompts above `threshold_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_extended: Option<f64>,

    /// Base cache write price for prompts at or below `threshold_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_base: Option<f64>,

    /// Extended cache write price for prompts above `threshold_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_extended: Option<f64>,

    /// Base cache read price for prompts at or below `threshold_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_base: Option<f64>,

    /// Extended cache read price for prompts above `threshold_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_extended: Option<f64>,

    /// Token threshold that switches pricing from base to extended rates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct PricingProviders {
    #[serde(default)]
    anthropic: HashMap<String, ModelPricing>,
    #[serde(default)]
    openai: HashMap<String, ModelPricing>,
    #[serde(default)]
    openrouter: HashMap<String, ModelPricing>,
    #[serde(default)]
    vertex: HashMap<String, ModelPricing>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PersistedPricingSnapshot {
    version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    synced_at: Option<String>,
    #[serde(default)]
    providers: PricingProviders,
    #[serde(default = "default_pricing")]
    default: ModelPricing,
}

impl Default for PersistedPricingSnapshot {
    fn default() -> Self {
        Self {
            version: PRICING_SNAPSHOT_VERSION,
            synced_at: None,
            providers: PricingProviders::default(),
            default: default_pricing(),
        }
    }
}

impl PersistedPricingSnapshot {
    /// Load the checked-in seed snapshot used for offline and first-run pricing.
    fn embedded_seed() -> Result<Self> {
        let snapshot: Self = serde_json::from_str(include_str!("pricing_seed.json"))
            .context("Failed to parse embedded pricing seed snapshot")?;
        snapshot.validate()?;
        Ok(snapshot)
    }

    /// Validate snapshot compatibility before it is used as pricing input.
    ///
    /// The version guard prevents old persisted cache files from silently
    /// shadowing newly generated seed data after the cache envelope changes.
    fn validate(&self) -> Result<()> {
        if self.version != PRICING_SNAPSHOT_VERSION {
            anyhow::bail!(
                "Unsupported pricing snapshot version {} (expected {})",
                self.version,
                PRICING_SNAPSHOT_VERSION
            );
        }

        Ok(())
    }

    /// Persist the snapshot atomically to the configured cache path.
    ///
    /// The write strategy uses `create_dir_all`, writes into a sibling temp file,
    /// flushes the file descriptor, and then renames the temp file into place so
    /// readers never observe a partially written JSON document.
    fn persist(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create pricing cache directory: {}",
                    parent.display()
                )
            })?;
        }

        let serialized =
            serde_json::to_vec_pretty(self).context("Failed to serialize pricing snapshot")?;
        let tmp_path = path.with_extension(format!("json.tmp.{}", std::process::id()));

        {
            let mut file = fs::File::create(&tmp_path).with_context(|| {
                format!(
                    "Failed to create temporary pricing cache: {}",
                    tmp_path.display()
                )
            })?;
            file.write_all(&serialized).with_context(|| {
                format!(
                    "Failed to write temporary pricing cache: {}",
                    tmp_path.display()
                )
            })?;
            file.sync_all().with_context(|| {
                format!(
                    "Failed to flush temporary pricing cache: {}",
                    tmp_path.display()
                )
            })?;
        }

        fs::rename(&tmp_path, path).with_context(|| {
            format!(
                "Failed to replace pricing cache {} with {}",
                path.display(),
                tmp_path.display()
            )
        })?;

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct PricingStore {
    providers: PricingProviders,
    default: ModelPricing,
}

impl PricingStore {
    /// Build the process-wide pricing store.
    ///
    /// This never panics on cache or network failures. If the embedded seed is
    /// unexpectedly invalid, Appam falls back to the hardcoded default table so
    /// price lookups continue to work for unknown models rather than crashing
    /// process initialization.
    fn initialize() -> Self {
        let runtime = PricingRuntimeConfig::default();
        let seed_snapshot = match PersistedPricingSnapshot::embedded_seed() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                warn!(error = %error, "Failed to load embedded pricing seed, using hardcoded defaults");
                PersistedPricingSnapshot::default()
            }
        };

        Self::bootstrap(&seed_snapshot, &runtime)
    }

    /// Bootstrap a pricing store from the embedded seed and runtime settings.
    ///
    /// Tests call this directly with a mocked endpoint and temporary cache path
    /// so they can validate sync and persistence without mutating the global
    /// pricing singleton or reaching the live network.
    fn bootstrap(seed_snapshot: &PersistedPricingSnapshot, runtime: &PricingRuntimeConfig) -> Self {
        let seed_store = Self::from_snapshot(seed_snapshot);
        let mut active_store = seed_store.clone();
        let mut active_source = "embedded seed";

        if runtime.cache_enabled {
            match Self::load_snapshot(runtime.cache_path.as_path()) {
                Ok(Some(snapshot)) => {
                    active_store.overlay_snapshot(&snapshot);
                    active_source = "persisted cache";
                    info!(
                        path = %runtime.cache_path.display(),
                        "Loaded pricing cache from disk"
                    );
                }
                Ok(None) => {
                    debug!(
                        path = %runtime.cache_path.display(),
                        "No pricing cache file found, using embedded seed"
                    );
                }
                Err(error) => {
                    warn!(
                        error = %error,
                        path = %runtime.cache_path.display(),
                        "Ignoring invalid pricing cache and using embedded seed"
                    );
                }
            }
        }

        if runtime.sync_enabled {
            match Self::fetch_remote_snapshot(runtime) {
                Ok(snapshot) => {
                    let remote_store = Self::from_snapshot(&snapshot);

                    if runtime.cache_enabled {
                        if let Err(error) = snapshot.persist(runtime.cache_path.as_path()) {
                            warn!(
                                error = %error,
                                path = %runtime.cache_path.display(),
                                "Failed to persist refreshed pricing cache"
                            );
                        } else {
                            info!(
                                path = %runtime.cache_path.display(),
                                "Persisted refreshed pricing cache"
                            );
                        }
                    }

                    info!(
                        endpoint = %runtime.api_url,
                        "Synchronized pricing data from models.dev"
                    );
                    return remote_store;
                }
                Err(error) => {
                    warn!(
                        error = %error,
                        endpoint = %runtime.api_url,
                        fallback = active_source,
                        "Pricing sync unavailable, using fallback source"
                    );
                }
            }
        }

        active_store
    }

    /// Rebuild the in-memory store from a normalized snapshot.
    fn from_snapshot(snapshot: &PersistedPricingSnapshot) -> Self {
        Self {
            providers: snapshot.providers.clone(),
            default: snapshot.default.clone(),
        }
    }

    /// Merge a normalized snapshot on top of the current store.
    ///
    /// Cache overlay intentionally overwrites matching entries while keeping seed
    /// entries that might be absent from older cache files.
    fn overlay_snapshot(&mut self, snapshot: &PersistedPricingSnapshot) {
        self.providers
            .anthropic
            .extend(snapshot.providers.anthropic.clone());
        self.providers
            .openai
            .extend(snapshot.providers.openai.clone());
        self.providers
            .openrouter
            .extend(snapshot.providers.openrouter.clone());
        self.providers
            .vertex
            .extend(snapshot.providers.vertex.clone());
        self.default = snapshot.default.clone();
    }

    /// Load a previously persisted cache snapshot from disk.
    fn load_snapshot(path: &Path) -> Result<Option<PersistedPricingSnapshot>> {
        if !path.exists() {
            return Ok(None);
        }

        let raw = fs::read_to_string(path)
            .with_context(|| format!("Failed to read pricing cache {}", path.display()))?;
        let snapshot: PersistedPricingSnapshot = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse pricing cache {}", path.display()))?;
        snapshot.validate()?;
        Ok(Some(snapshot))
    }

    /// Fetch the live models.dev payload and normalize it into Appam's snapshot
    /// format.
    ///
    /// This function keeps the public pricing API synchronous, but it still
    /// supports callers that initialize pricing from async contexts. When a
    /// Tokio runtime is already active, the blocking network work is moved onto
    /// a dedicated OS thread so the current runtime worker is not used for the
    /// outbound request itself.
    fn fetch_remote_snapshot(runtime: &PricingRuntimeConfig) -> Result<PersistedPricingSnapshot> {
        if tokio::runtime::Handle::try_current().is_ok() {
            let runtime = runtime.clone();
            let handle = thread::Builder::new()
                .name("appam-pricing-sync".to_string())
                .spawn(move || Self::fetch_remote_snapshot_blocking(&runtime))
                .context("Failed to spawn pricing sync thread")?;

            return handle
                .join()
                .map_err(|_| anyhow!("Pricing sync thread panicked"))?;
        }

        Self::fetch_remote_snapshot_blocking(runtime)
    }

    /// Perform the actual blocking models.dev fetch.
    ///
    /// Separating the blocking transport from the runtime detection keeps the
    /// call path explicit and lets async callers offload the network I/O onto a
    /// dedicated thread without duplicating request logic.
    fn fetch_remote_snapshot_blocking(
        runtime: &PricingRuntimeConfig,
    ) -> Result<PersistedPricingSnapshot> {
        let client = Client::builder()
            .connect_timeout(runtime.connect_timeout)
            .timeout(runtime.request_timeout)
            .build()
            .context("Failed to construct blocking pricing sync client")?;

        let payload = client
            .get(&runtime.api_url)
            .send()
            .with_context(|| format!("Failed to fetch pricing data from {}", runtime.api_url))?
            .error_for_status()
            .with_context(|| {
                format!(
                    "Pricing endpoint returned error status: {}",
                    runtime.api_url
                )
            })?
            .json::<ModelsDevPayload>()
            .with_context(|| {
                format!("Failed to decode pricing payload from {}", runtime.api_url)
            })?;

        Self::snapshot_from_models_dev(payload)
    }

    /// Convert the raw models.dev provider payload into Appam's normalized
    /// snapshot representation.
    ///
    /// Only the providers that Appam currently prices are imported:
    /// `anthropic`, `openai`, `openrouter`, and `google-vertex`.
    fn snapshot_from_models_dev(payload: ModelsDevPayload) -> Result<PersistedPricingSnapshot> {
        let anthropic = Self::normalize_models_dev_provider(payload.get("anthropic"))
            .context("Missing anthropic pricing provider in models.dev payload")?;
        let openai = Self::normalize_models_dev_provider(payload.get("openai"))
            .context("Missing openai pricing provider in models.dev payload")?;
        let openrouter = Self::normalize_models_dev_provider(payload.get("openrouter"))
            .context("Missing openrouter pricing provider in models.dev payload")?;
        let vertex = Self::normalize_models_dev_provider(payload.get("google-vertex"))
            .context("Missing google-vertex pricing provider in models.dev payload")?;

        Ok(PersistedPricingSnapshot {
            version: PRICING_SNAPSHOT_VERSION,
            synced_at: Some(Utc::now().to_rfc3339()),
            providers: PricingProviders {
                anthropic,
                openai,
                openrouter,
                vertex,
            },
            default: default_pricing(),
        })
    }

    /// Normalize a single provider section from the raw models.dev schema.
    fn normalize_models_dev_provider(
        provider: Option<&ModelsDevProviderPayload>,
    ) -> Result<HashMap<String, ModelPricing>> {
        let provider = provider.context("Provider payload missing")?;
        let mut normalized = HashMap::with_capacity(provider.models.len());

        for (model_id, model) in &provider.models {
            normalized.insert(model_id.clone(), Self::normalize_models_dev_model(model));
        }

        Ok(normalized)
    }

    /// Convert a raw models.dev model record into Appam's pricing shape.
    fn normalize_models_dev_model(model: &ModelsDevModelPayload) -> ModelPricing {
        let mut pricing = ModelPricing {
            name: model.name.clone(),
            input: model.cost.as_ref().and_then(|cost| cost.input),
            output: model.cost.as_ref().and_then(|cost| cost.output),
            cache_write: model.cost.as_ref().and_then(|cost| cost.cache_write),
            cache_read: model.cost.as_ref().and_then(|cost| cost.cache_read),
            reasoning: model.cost.as_ref().and_then(|cost| cost.reasoning),
            input_base: None,
            input_extended: None,
            output_base: None,
            output_extended: None,
            cache_write_base: None,
            cache_write_extended: None,
            cache_read_base: None,
            cache_read_extended: None,
            threshold_tokens: None,
        };

        if let Some(tiered) = model
            .cost
            .as_ref()
            .and_then(|cost| cost.context_over_200k.as_ref())
        {
            pricing.input_base = pricing.input;
            pricing.input_extended = tiered.input;
            pricing.output_base = pricing.output;
            pricing.output_extended = tiered.output;
            pricing.cache_write_base = pricing.cache_write;
            pricing.cache_write_extended = tiered.cache_write;
            pricing.cache_read_base = pricing.cache_read;
            pricing.cache_read_extended = tiered.cache_read;
            pricing.threshold_tokens = Some(TIER_THRESHOLD_TOKENS);
        }

        pricing
    }

    /// Return the provider map used for lookups.
    fn provider_map(&self, provider: &str) -> Option<&HashMap<String, ModelPricing>> {
        match provider {
            "anthropic" => Some(&self.providers.anthropic),
            "openai" => Some(&self.providers.openai),
            "openrouter" => Some(&self.providers.openrouter),
            "vertex" => Some(&self.providers.vertex),
            _ => None,
        }
    }

    /// Find pricing for a provider/model pair after applying Appam's legacy
    /// normalization rules.
    fn lookup(&self, provider: &str, model: &str) -> Option<&ModelPricing> {
        let provider_map = self.provider_map(provider)?;

        for candidate in pricing_lookup_candidates(provider, model) {
            if let Some(pricing) = provider_map.get(&candidate) {
                return Some(pricing);
            }
        }

        None
    }
}

#[derive(Debug, Clone)]
struct PricingRuntimeConfig {
    api_url: String,
    cache_path: PathBuf,
    cache_enabled: bool,
    sync_enabled: bool,
    connect_timeout: Duration,
    request_timeout: Duration,
}

impl Default for PricingRuntimeConfig {
    fn default() -> Self {
        let under_test = cfg!(test);

        Self {
            api_url: MODELS_DEV_API_URL.to_string(),
            cache_path: PathBuf::from(PRICING_CACHE_PATH),
            cache_enabled: !under_test,
            sync_enabled: !under_test,
            connect_timeout: Duration::from_secs(2),
            request_timeout: Duration::from_secs(5),
        }
    }
}

type ModelsDevPayload = HashMap<String, ModelsDevProviderPayload>;

#[derive(Debug, Clone, Deserialize)]
struct ModelsDevProviderPayload {
    #[serde(default)]
    models: HashMap<String, ModelsDevModelPayload>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelsDevModelPayload {
    name: String,
    #[serde(default)]
    cost: Option<ModelsDevCostPayload>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelsDevCostPayload {
    #[serde(default)]
    input: Option<f64>,
    #[serde(default)]
    output: Option<f64>,
    #[serde(default)]
    reasoning: Option<f64>,
    #[serde(default)]
    cache_read: Option<f64>,
    #[serde(default)]
    cache_write: Option<f64>,
    #[serde(default)]
    context_over_200k: Option<ModelsDevContextCostPayload>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelsDevContextCostPayload {
    #[serde(default)]
    input: Option<f64>,
    #[serde(default)]
    output: Option<f64>,
    #[serde(default)]
    cache_read: Option<f64>,
    #[serde(default)]
    cache_write: Option<f64>,
}

fn default_pricing() -> ModelPricing {
    ModelPricing {
        name: "Unknown Model".to_string(),
        input: Some(3.0),
        output: Some(15.0),
        cache_write: Some(3.75),
        cache_read: Some(0.3),
        reasoning: None,
        input_base: None,
        input_extended: None,
        output_base: None,
        output_extended: None,
        cache_write_base: None,
        cache_write_extended: None,
        cache_read_base: None,
        cache_read_extended: None,
        threshold_tokens: None,
    }
}

fn pricing_lookup_candidates(provider: &str, model: &str) -> Vec<String> {
    let trimmed = model.trim();
    let mut candidates = Vec::new();

    push_unique_candidate(&mut candidates, trimmed);

    match provider {
        "anthropic" => {
            let normalized = normalize_anthropic_model(trimmed);
            push_unique_candidate(&mut candidates, normalized.as_str());

            if let Some(alias) = anthropic_alias(normalized.as_str()) {
                push_unique_candidate(&mut candidates, alias);
            }
        }
        "openai" => {
            let normalized = crate::llm::openai::normalize_openai_model(trimmed);
            push_unique_candidate(&mut candidates, normalized.as_str());
        }
        "openrouter" => {
            if let Some(normalized) = normalize_openrouter_model(trimmed) {
                push_unique_candidate(&mut candidates, normalized.as_str());

                if let Some(alias) = openrouter_alias(normalized.as_str()) {
                    push_unique_candidate(&mut candidates, alias);
                }
            }
        }
        "vertex" => {}
        _ => {}
    }

    candidates
}

fn push_unique_candidate(candidates: &mut Vec<String>, candidate: &str) {
    if candidate.is_empty() {
        return;
    }

    if !candidates.iter().any(|existing| existing == candidate) {
        candidates.push(candidate.to_string());
    }
}

fn normalize_anthropic_model(model: &str) -> String {
    let without_prefix = model
        .strip_prefix("us.anthropic.")
        .or_else(|| model.strip_prefix("anthropic."))
        .unwrap_or(model);
    let without_revision = without_prefix.split(':').next().unwrap_or(without_prefix);
    let without_suffix = without_revision
        .strip_suffix("-v1")
        .or_else(|| without_revision.strip_suffix("-v2"))
        .or_else(|| without_revision.strip_suffix("-v3"))
        .unwrap_or(without_revision);

    without_suffix.trim().to_string()
}

fn anthropic_alias(model: &str) -> Option<&'static str> {
    match model {
        "claude-haiku-4" | "claude-haiku-4-20250514" => Some("claude-haiku-4-5"),
        "claude-sonnet-4-5-20250514" => Some("claude-sonnet-4-5"),
        _ => None,
    }
}

fn normalize_openrouter_model(model: &str) -> Option<String> {
    if let Some(rest) = model.strip_prefix("openai/") {
        return Some(format!(
            "openai/{}",
            crate::llm::openai::normalize_openai_model(rest)
        ));
    }

    if looks_like_openai_model(model) {
        return Some(format!(
            "openai/{}",
            crate::llm::openai::normalize_openai_model(model)
        ));
    }

    if model.starts_with("anthropic/") {
        return Some(model.to_string());
    }

    None
}

fn openrouter_alias(model: &str) -> Option<&'static str> {
    match model {
        "anthropic/claude-haiku-4" => Some("anthropic/claude-haiku-4.5"),
        _ => None,
    }
}

fn looks_like_openai_model(model: &str) -> bool {
    model.starts_with("gpt-")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.starts_with("codex-")
}

fn has_tiered_pricing(pricing: &ModelPricing) -> bool {
    pricing.threshold_tokens.is_some()
        && (pricing.input_base.is_some()
            || pricing.input_extended.is_some()
            || pricing.output_base.is_some()
            || pricing.output_extended.is_some()
            || pricing.cache_write_base.is_some()
            || pricing.cache_write_extended.is_some()
            || pricing.cache_read_base.is_some()
            || pricing.cache_read_extended.is_some())
}

fn select_tier_rate(
    base: Option<f64>,
    extended: Option<f64>,
    use_extended_tier: bool,
) -> Option<f64> {
    if use_extended_tier {
        extended.or(base)
    } else {
        base.or(extended)
    }
}

/// Get pricing for a specific model.
///
/// Looks up the model by provider and model name, applying Appam's provider-
/// specific normalization rules before falling back to default pricing.
///
/// # Arguments
///
/// * `provider` - Canonical provider key (`anthropic`, `openai`, `openrouter`, `vertex`)
/// * `model` - Provider-native or Appam-legacy model identifier
///
/// # Returns
///
/// A reference to the resolved `ModelPricing` entry, or the default fallback
/// pricing when the model is unknown.
pub fn get_model_pricing(provider: &str, model: &str) -> &'static ModelPricing {
    let provider_key = provider.to_lowercase();
    let pricing = PRICING.lookup(provider_key.as_str(), model);

    if pricing.is_none() {
        warn!(
            provider = provider,
            model = model,
            "Pricing not found for model, using default fallback"
        );
    }

    pricing.unwrap_or(&PRICING.default)
}

/// Calculate cost for LLM usage.
///
/// Computes the total cost in USD based on token consumption and model pricing.
/// Cache-read tokens are excluded from the standard input rate and billed at the
/// cache-read rate when present. Tiered models use the total prompt size
/// (`billable input + cache read`) to determine whether base or extended rates
/// apply.
///
/// # Arguments
///
/// * `usage` - Token usage statistics from an LLM response
/// * `provider` - Canonical provider name
/// * `model` - Provider-native or Appam-legacy model identifier
///
/// # Returns
///
/// The total estimated cost in USD.
pub fn calculate_cost(usage: &UnifiedUsage, provider: &str, model: &str) -> f64 {
    let pricing = get_model_pricing(provider, model);

    let input_tokens = usage.input_tokens as f64;
    let cache_creation_tokens = usage.cache_creation_input_tokens.unwrap_or(0) as f64;
    let cache_read_tokens = usage.cache_read_input_tokens.unwrap_or(0) as f64;
    let output_tokens = usage.output_tokens as f64;
    let reasoning_tokens = usage.reasoning_tokens.unwrap_or(0) as f64;
    let billable_input_tokens = (input_tokens - cache_read_tokens).max(0.0);

    if has_tiered_pricing(pricing) {
        return calculate_tiered_cost(
            billable_input_tokens,
            output_tokens,
            cache_creation_tokens,
            cache_read_tokens,
            reasoning_tokens,
            pricing,
        );
    }

    let mut total_cost = 0.0;

    if let Some(input_rate) = pricing.input {
        if billable_input_tokens > 0.0 {
            total_cost += (billable_input_tokens / 1_000_000.0) * input_rate;
        }
    }

    if let Some(output_rate) = pricing.output {
        total_cost += (output_tokens / 1_000_000.0) * output_rate;
    }

    if cache_creation_tokens > 0.0 {
        if let Some(cache_write_rate) = pricing.cache_write {
            total_cost += (cache_creation_tokens / 1_000_000.0) * cache_write_rate;
        }
    }

    if cache_read_tokens > 0.0 {
        if let Some(cache_read_rate) = pricing.cache_read {
            total_cost += (cache_read_tokens / 1_000_000.0) * cache_read_rate;
        }
    }

    if reasoning_tokens > 0.0 {
        if let Some(reasoning_rate) = pricing.reasoning.or(pricing.output) {
            total_cost += (reasoning_tokens / 1_000_000.0) * reasoning_rate;
        }
    }

    total_cost
}

/// Calculate cost for tiered pricing models.
///
/// Models.dev expresses extended-context pricing under `cost.context_over_200k`.
/// Appam maps that into the existing tiered fields and then selects the tier
/// using the total prompt size, including cache hits. If an extended rate is
/// absent for a specific field, the base rate is reused instead of silently
/// zeroing that token category.
fn calculate_tiered_cost(
    billable_input_tokens: f64,
    output_tokens: f64,
    cache_creation_tokens: f64,
    cache_read_tokens: f64,
    reasoning_tokens: f64,
    pricing: &ModelPricing,
) -> f64 {
    let threshold = pricing.threshold_tokens.unwrap_or(TIER_THRESHOLD_TOKENS) as f64;
    let total_prompt_tokens = billable_input_tokens + cache_read_tokens;
    let use_extended_tier = total_prompt_tokens > threshold;
    let mut total_cost = 0.0;

    let input_rate = select_tier_rate(
        pricing.input_base,
        pricing.input_extended,
        use_extended_tier,
    );
    if let Some(input_rate) = input_rate {
        if billable_input_tokens > 0.0 {
            total_cost += (billable_input_tokens / 1_000_000.0) * input_rate;
        }
    }

    let output_rate = select_tier_rate(
        pricing.output_base,
        pricing.output_extended,
        use_extended_tier,
    );
    if let Some(output_rate) = output_rate {
        total_cost += (output_tokens / 1_000_000.0) * output_rate;
    }

    let cache_write_rate = select_tier_rate(
        pricing.cache_write_base,
        pricing.cache_write_extended,
        use_extended_tier,
    );
    if let Some(cache_write_rate) = cache_write_rate {
        if cache_creation_tokens > 0.0 {
            total_cost += (cache_creation_tokens / 1_000_000.0) * cache_write_rate;
        }
    }

    let cache_read_rate = select_tier_rate(
        pricing.cache_read_base,
        pricing.cache_read_extended,
        use_extended_tier,
    );
    if let Some(cache_read_rate) = cache_read_rate {
        if cache_read_tokens > 0.0 {
            total_cost += (cache_read_tokens / 1_000_000.0) * cache_read_rate;
        }
    }

    if reasoning_tokens > 0.0 {
        if let Some(reasoning_rate) = pricing.reasoning.or(output_rate) {
            total_cost += (reasoning_tokens / 1_000_000.0) * reasoning_rate;
        }
    }

    total_cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };
    use tempfile::TempDir;

    const RAW_MODELS_DEV_FIXTURE: &str = include_str!("../../tests/fixtures/models_dev_api.json");

    fn embedded_seed_snapshot() -> PersistedPricingSnapshot {
        PersistedPricingSnapshot::embedded_seed().expect("embedded seed should parse")
    }

    fn embedded_seed_store() -> PricingStore {
        PricingStore::from_snapshot(&embedded_seed_snapshot())
    }

    fn remote_snapshot_from_fixture() -> PersistedPricingSnapshot {
        let payload: ModelsDevPayload =
            serde_json::from_str(RAW_MODELS_DEV_FIXTURE).expect("fixture payload should parse");
        PricingStore::snapshot_from_models_dev(payload).expect("fixture payload should normalize")
    }

    fn pricing_for_provider<'a>(
        providers: &'a PricingProviders,
        provider: &str,
    ) -> &'a HashMap<String, ModelPricing> {
        match provider {
            "anthropic" => &providers.anthropic,
            "openai" => &providers.openai,
            "openrouter" => &providers.openrouter,
            "vertex" => &providers.vertex,
            _ => panic!("unknown provider {provider}"),
        }
    }

    fn assert_nearly_equal(left: f64, right: f64) {
        assert!(
            (left - right).abs() < 1e-9,
            "expected {left:.12} to equal {right:.12}"
        );
    }

    fn expected_cost(pricing: &ModelPricing, usage: &UnifiedUsage) -> f64 {
        let input_tokens = usage.input_tokens as f64;
        let cache_write_tokens = usage.cache_creation_input_tokens.unwrap_or(0) as f64;
        let cache_read_tokens = usage.cache_read_input_tokens.unwrap_or(0) as f64;
        let output_tokens = usage.output_tokens as f64;
        let reasoning_tokens = usage.reasoning_tokens.unwrap_or(0) as f64;
        let billable_input_tokens = (input_tokens - cache_read_tokens).max(0.0);

        let tiered = has_tiered_pricing(pricing);
        let use_extended_tier = tiered
            && billable_input_tokens + cache_read_tokens
                > pricing.threshold_tokens.unwrap_or(TIER_THRESHOLD_TOKENS) as f64;

        let input_rate = if tiered {
            select_tier_rate(
                pricing.input_base,
                pricing.input_extended,
                use_extended_tier,
            )
        } else {
            pricing.input
        };
        let output_rate = if tiered {
            select_tier_rate(
                pricing.output_base,
                pricing.output_extended,
                use_extended_tier,
            )
        } else {
            pricing.output
        };
        let cache_write_rate = if tiered {
            select_tier_rate(
                pricing.cache_write_base,
                pricing.cache_write_extended,
                use_extended_tier,
            )
        } else {
            pricing.cache_write
        };
        let cache_read_rate = if tiered {
            select_tier_rate(
                pricing.cache_read_base,
                pricing.cache_read_extended,
                use_extended_tier,
            )
        } else {
            pricing.cache_read
        };

        let mut total = 0.0;

        if let Some(rate) = input_rate {
            total += (billable_input_tokens / 1_000_000.0) * rate;
        }

        if let Some(rate) = output_rate {
            total += (output_tokens / 1_000_000.0) * rate;
        }

        if let Some(rate) = cache_write_rate {
            total += (cache_write_tokens / 1_000_000.0) * rate;
        }

        if let Some(rate) = cache_read_rate {
            total += (cache_read_tokens / 1_000_000.0) * rate;
        }

        if let Some(rate) = pricing.reasoning.or(output_rate) {
            total += (reasoning_tokens / 1_000_000.0) * rate;
        }

        total
    }

    fn pricing_runtime_config(cache_path: PathBuf, api_url: String) -> PricingRuntimeConfig {
        PricingRuntimeConfig {
            api_url,
            cache_path,
            cache_enabled: true,
            sync_enabled: true,
            connect_timeout: Duration::from_millis(250),
            request_timeout: Duration::from_secs(1),
        }
    }

    fn spawn_http_server(
        status_line: &'static str,
        body: String,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener
            .local_addr()
            .expect("listener should report bound address");

        let handle = thread::spawn(move || {
            if let Ok((mut stream, _peer)) = listener.accept() {
                let mut request_buffer = [0_u8; 1024];
                let _ = stream.read(&mut request_buffer);
                let response = format!(
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("response should write");
                stream.flush().expect("response should flush");
            }
        });

        (format!("http://{address}/api.json"), handle)
    }

    #[test]
    fn test_embedded_seed_matches_models_dev_fixture_for_all_canonical_models() {
        let embedded = embedded_seed_snapshot();
        let normalized_fixture = remote_snapshot_from_fixture();

        assert_eq!(embedded.default, default_pricing());
        assert_eq!(embedded.providers, normalized_fixture.providers);
    }

    #[test]
    fn test_alias_compatibility_for_legacy_model_ids() {
        let store = embedded_seed_store();
        let cases = [
            ("anthropic", "claude-haiku-4-20250514", "claude-haiku-4-5"),
            (
                "anthropic",
                "anthropic.claude-opus-4-6-v1",
                "claude-opus-4-6",
            ),
            (
                "anthropic",
                "us.anthropic.claude-opus-4-6-v1",
                "claude-opus-4-6",
            ),
            (
                "anthropic",
                "us.anthropic.claude-sonnet-4-5-20250514-v1:0",
                "claude-sonnet-4-5",
            ),
            ("openai", "openai/gpt-5.5", "gpt-5.5"),
            (
                "openrouter",
                "anthropic/claude-haiku-4",
                "anthropic/claude-haiku-4.5",
            ),
            ("openrouter", "gpt-5.5", "openai/gpt-5.5"),
        ];

        for (provider, alias, canonical) in cases {
            let alias_pricing = store
                .lookup(provider, alias)
                .unwrap_or_else(|| panic!("alias {alias} should resolve"));
            let canonical_pricing = store
                .lookup(provider, canonical)
                .unwrap_or_else(|| panic!("canonical model {canonical} should resolve"));

            assert_eq!(
                alias_pricing, canonical_pricing,
                "alias {alias} should resolve to canonical model {canonical}"
            );
        }
    }

    #[test]
    fn test_sync_persists_remote_snapshot_and_cache_reloads_when_offline() {
        let tempdir = TempDir::new().expect("tempdir should create");
        let cache_path = tempdir.path().join("pricing").join("models.dev.json");
        let (url, handle) =
            spawn_http_server("HTTP/1.1 200 OK", RAW_MODELS_DEV_FIXTURE.to_string());
        let runtime = pricing_runtime_config(cache_path.clone(), url);

        let synchronized = PricingStore::bootstrap(&embedded_seed_snapshot(), &runtime);
        let expected = PricingStore::from_snapshot(&remote_snapshot_from_fixture());

        assert_eq!(synchronized, expected);
        assert!(cache_path.exists(), "sync should persist a cache file");
        handle.join().expect("server thread should complete");

        let offline_runtime = PricingRuntimeConfig {
            api_url: "http://127.0.0.1:9/api.json".to_string(),
            ..pricing_runtime_config(
                cache_path.clone(),
                "http://127.0.0.1:9/api.json".to_string(),
            )
        };

        let cached = PricingStore::bootstrap(&embedded_seed_snapshot(), &offline_runtime);
        assert_eq!(cached, expected);
    }

    #[test]
    fn test_invalid_cache_falls_back_to_seed_without_panicking() {
        let tempdir = TempDir::new().expect("tempdir should create");
        let cache_path = tempdir.path().join("pricing").join("models.dev.json");
        fs::create_dir_all(cache_path.parent().expect("cache path should have parent"))
            .expect("cache directory should create");
        fs::write(&cache_path, "this is not valid json").expect("invalid cache should write");

        let runtime = PricingRuntimeConfig {
            api_url: "http://127.0.0.1:9/api.json".to_string(),
            cache_path,
            cache_enabled: true,
            sync_enabled: false,
            connect_timeout: Duration::from_millis(50),
            request_timeout: Duration::from_millis(100),
        };

        let store = PricingStore::bootstrap(&embedded_seed_snapshot(), &runtime);
        assert_eq!(store, embedded_seed_store());
    }

    #[test]
    fn test_calculate_cost_matches_seeded_pricing_for_all_canonical_models() {
        let snapshot = embedded_seed_snapshot();
        let providers = ["anthropic", "openai", "openrouter", "vertex"];

        let below_threshold_usage = UnifiedUsage {
            input_tokens: 120_000,
            output_tokens: 24_000,
            cache_creation_input_tokens: Some(8_000),
            cache_read_input_tokens: Some(40_000),
            reasoning_tokens: Some(6_000),
        };
        let above_threshold_usage = UnifiedUsage {
            input_tokens: 260_000,
            output_tokens: 24_000,
            cache_creation_input_tokens: Some(8_000),
            cache_read_input_tokens: Some(40_000),
            reasoning_tokens: Some(6_000),
        };

        for provider in providers {
            for (model, pricing) in pricing_for_provider(&snapshot.providers, provider) {
                let expected_below = expected_cost(pricing, &below_threshold_usage);
                let actual_below = calculate_cost(&below_threshold_usage, provider, model);
                assert_nearly_equal(actual_below, expected_below);

                if pricing.threshold_tokens.is_some() {
                    let expected_above = expected_cost(pricing, &above_threshold_usage);
                    let actual_above = calculate_cost(&above_threshold_usage, provider, model);
                    assert_nearly_equal(actual_above, expected_above);
                }
            }
        }
    }

    #[test]
    fn test_reasoning_tokens_fall_back_to_selected_output_rate_when_reasoning_price_is_missing() {
        let cases = [("openai", "gpt-5.5"), ("vertex", "gemini-3.1-pro-preview")];

        for (provider, model) in cases {
            let pricing = get_model_pricing(provider, model);
            assert!(
                pricing.reasoning.is_none(),
                "fixture model {model} should omit explicit reasoning pricing"
            );

            let usage = UnifiedUsage {
                input_tokens: 250_000,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                reasoning_tokens: Some(10_000),
            };

            let expected = expected_cost(pricing, &usage);
            let actual = calculate_cost(&usage, provider, model);
            assert_nearly_equal(actual, expected);
        }
    }

    #[test]
    fn test_unknown_model_uses_default_fallback() {
        let usage = UnifiedUsage {
            input_tokens: 10_000,
            output_tokens: 5_000,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            reasoning_tokens: None,
        };

        let cost = calculate_cost(&usage, "unknown_provider", "unknown_model");
        assert_nearly_equal(cost, 0.105);
        assert_eq!(
            get_model_pricing("unknown_provider", "unknown_model"),
            &default_pricing()
        );
    }
}
