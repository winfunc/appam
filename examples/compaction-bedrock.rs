//! Auto-compaction agent — Claude on AWS Bedrock
//!
//! Demonstrates appam's server-side context compaction support on Bedrock:
//! - `enable_auto_compaction(trigger_tokens)` on `AgentBuilder`
//! - Bedrock carries the compaction beta as an `anthropic_beta` entry in the
//!   request body (not an HTTP header); appam handles this automatically
//! - The `compaction` content block is retained in session history and
//!   replayed automatically; Bedrock ignores content before the last block
//! - Compaction pass tokens are tracked separately in usage/cost accounting
//!
//! Bedrock currently documents compaction support for Claude Sonnet 4.6 and
//! Claude Opus 4.6 (InvokeModel / InvokeModelWithResponseStream only). A full
//! run consumes roughly 200K input tokens (~$0.60 with Sonnet 4.6).
//!
//! Usage:
//!   export AWS_ACCESS_KEY_ID=...
//!   export AWS_SECRET_ACCESS_KEY=...
//!   export AWS_REGION=us-east-1                # optional (default us-east-1)
//!   export AWS_BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-6   # optional
//!   cargo run --example compaction-bedrock

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
    println!("🗜  Auto-Compaction Agent - Claude on AWS Bedrock\n");

    let region = std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|_| "us-east-1".to_string());
    let model_id = std::env::var("AWS_BEDROCK_MODEL_ID")
        .unwrap_or_else(|_| "us.anthropic.claude-sonnet-4-6".to_string());

    let agent = AgentBuilder::new("compaction-demo-bedrock")
        .provider(LlmProvider::Bedrock {
            region: region.clone(),
            model_id: model_id.clone(),
            auth_method: appam::llm::anthropic::BedrockAuthMethod::SigV4,
        })
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

    println!("✓ Model: {model_id} @ {region}");
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
