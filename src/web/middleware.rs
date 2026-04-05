//! Middleware for rate limiting, authentication, and request processing.

use std::sync::Arc;
use tower_governor::{
    governor::GovernorConfigBuilder, key_extractor::PeerIpKeyExtractor, GovernorLayer,
};

/// Create rate limiting layer.
///
/// Applies per-IP rate limiting to prevent abuse. Default: 60 requests per minute.
pub fn rate_limit_layer() -> GovernorLayer<
    PeerIpKeyExtractor,
    governor::middleware::NoOpMiddleware<governor::clock::QuantaInstant>,
    axum::body::Body,
> {
    let config = Arc::new(
        GovernorConfigBuilder::default()
            .per_second(1) // 1 request per second
            .burst_size(60) // Allow bursts up to 60
            .finish()
            .unwrap(),
    );

    GovernorLayer::new(config)
}
