//! OpenAI Codex subscription provider implementation.
//!
//! This module implements the ChatGPT subscription-backed Codex transport that
//! routes requests to `https://chatgpt.com/backend-api/codex/responses`.
//!
//! # Architecture
//!
//! The Codex provider deliberately lives outside the direct OpenAI client
//! because the transport contract differs in several important ways:
//!
//! - authentication uses ChatGPT OAuth access tokens rather than Platform API keys
//! - requests target the ChatGPT backend instead of `api.openai.com`
//! - requests require Codex-specific headers such as `chatgpt-account-id`
//! - the system prompt is sent via `instructions` and omitted from `input`
//!
//! Reusable OpenAI reasoning and text-verbosity types are still re-exported via
//! `crate::llm::openai`, while the transport and credential management remain
//! isolated here.
//!
//! # Security
//!
//! Cached Codex authentication files and resolved access tokens are sensitive
//! local credentials. They should never be logged, persisted in traces, or
//! copied into user-facing diagnostics.

pub mod auth;
pub mod client;
pub mod config;

pub use auth::{
    login_openai_codex_interactive, resolve_openai_codex_auth, OpenAICodexAuthSource,
    OpenAICodexAuthStorage, OpenAICodexCredentials, ResolvedOpenAICodexAuth,
};
pub use client::OpenAICodexClient;
pub use config::{resolve_reasoning_effort_for_codex_model, OpenAICodexConfig};
