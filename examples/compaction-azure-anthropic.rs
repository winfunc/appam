//! Auto-compaction agent — Claude on Azure (Anthropic Messages API)
//!
//! Demonstrates appam's server-side context compaction support on
//! Azure-hosted Anthropic endpoints:
//! - `enable_auto_compaction(trigger_tokens)` on `AgentBuilder`
//! - Azure Anthropic preserves the Messages wire format, so the compaction
//!   beta header (`compact-2026-01-12`) and `context_management` edit work
//!   exactly like the direct Anthropic API; appam adds them automatically
//! - The `compaction` content block is retained in session history and
//!   replayed automatically
//! - Compaction pass tokens are tracked separately in usage/cost accounting
//!
//! Requires a compaction-capable Claude deployment (Sonnet 4.6+, Opus 4.6+,
//! or the Claude 5 family).
//!
//! Usage:
//!   export AZURE_ANTHROPIC_BASE_URL=https://my-resource.services.ai.azure.com/anthropic
//!   export AZURE_ANTHROPIC_API_KEY=...       # or AZURE_API_KEY
//!   export AZURE_ANTHROPIC_MODEL=claude-opus-4-6   # deployment name (optional)
//!   cargo run --example compaction-azure-anthropic

use anyhow::Result;
use appam::llm::UnifiedContentBlock;
use appam::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// Tool Definitions
// ============================================================================

#[derive(Serialize)]
struct ArchiveVolume {
    volume: u32,
    title: String,
    body: String,
}

/// Return a deterministic ~16K-token document so a handful of tool calls
/// pushes the conversation past the compaction trigger.
#[tool(description = "Fetch one volume of the expedition archive by number (1-4)")]
fn fetch_archive(
    #[arg(description = "Volume number to fetch, between 1 and 4")] volume: u32,
) -> Result<ArchiveVolume> {
    let titles = [
        "Flora of the Northern Ridge",
        "Glacial Meltwater Surveys",
        "Nocturnal Fauna Observations",
        "Mineral Deposits and Cave Systems",
    ];
    let title = titles
        .get((volume.saturating_sub(1)) as usize % titles.len())
        .unwrap_or(&titles[0])
        .to_string();

    // ~1,600 entries * ~10 tokens each ≈ 16K tokens per volume
    let body = (1..=1_600)
        .map(|entry| {
            format!(
                "Entry {entry:04} (vol {volume}): field team recorded routine observations \
                 near waypoint {waypoint}; conditions stable, samples catalogued.",
                waypoint = entry * 7 % 400,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    Ok(ArchiveVolume {
        volume,
        title,
        body,
    })
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    println!("🗜  Auto-Compaction Agent - Claude on Azure (Anthropic)\n");

    let base_url = std::env::var("AZURE_ANTHROPIC_BASE_URL").expect(
        "Set AZURE_ANTHROPIC_BASE_URL, e.g. https://my-resource.services.ai.azure.com/anthropic",
    );
    let model =
        std::env::var("AZURE_ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-opus-4-6".to_string());

    let agent = AgentBuilder::new("compaction-demo-azure-anthropic")
        .provider(LlmProvider::AzureAnthropic {
            base_url: base_url.clone(),
            auth_method: appam::llm::anthropic::AzureAnthropicAuthMethod::XApiKey,
        })
        .model(&model)
        .system_prompt(
            "You are an expedition archivist. Fetch archive volumes with the \
             fetch_archive tool exactly as instructed, one volume per tool call. \
             Do not summarize until every requested volume has been fetched.",
        )
        // Server-side compaction once input crosses ~50K tokens (the
        // Anthropic minimum; values below 50K are clamped upward).
        .enable_auto_compaction(50_000)
        .with_tool(Arc::new(fetch_archive()))
        .max_tokens(4096)
        .build()?;

    println!("✓ Deployment: {model} @ {base_url}");
    println!("✓ Auto-compaction enabled (trigger: 50K input tokens)");
    println!("✓ Tool: fetch_archive (~16K tokens per volume)\n");

    let compaction_count = Arc::new(AtomicUsize::new(0));
    let compaction_count_stream = Arc::clone(&compaction_count);

    let session = agent
        .stream(
            "Fetch archive volumes 1, 2, 3, and 4 one at a time using the \
             fetch_archive tool. After fetching ALL four volumes, reply with a \
             single short paragraph naming each volume's title.",
        )
        .on_content(|content| {
            print!("{}", content);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        })
        .on_tool_call(|tool_name, arguments| {
            println!("\n🔧 {} {}", tool_name, arguments);
        })
        .on_tool_result(|tool_name, _result| {
            println!("   ✓ {} completed", tool_name);
        })
        .on_compaction(move |provider, summary| {
            compaction_count_stream.fetch_add(1, Ordering::Relaxed);
            println!("\n◈ Context compacted by {provider}");
            if let Some(summary) = summary {
                let preview: String = summary.chars().take(160).collect();
                println!("  Summary: {preview}…");
            }
        })
        .run()
        .await?;

    // ------------------------------------------------------------------
    // Verify what happened
    // ------------------------------------------------------------------
    let compactions = compaction_count.load(Ordering::Relaxed);
    let history_compaction_blocks = session
        .messages
        .iter()
        .filter_map(|message| message.raw_content_blocks.as_ref())
        .flatten()
        .filter(|block| matches!(block, UnifiedContentBlock::Compaction { .. }))
        .count();

    println!("\n\n===== Session Report =====");
    println!("Compaction events observed:   {compactions}");
    println!("Compaction blocks in history: {history_compaction_blocks}");
    if let Some(usage) = &session.usage {
        println!("\n{}", usage.format_detailed());
    }

    if compactions == 0 {
        println!(
            "\n⚠ No compaction occurred — the conversation may not have crossed \
             the 50K-token trigger. Increase the number of volumes."
        );
    } else {
        println!("\n✓ Server-side compaction verified end to end");
    }

    Ok(())
}
