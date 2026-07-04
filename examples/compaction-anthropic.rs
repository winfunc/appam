//! Auto-compaction agent — Anthropic Claude (direct API)
//!
//! Demonstrates appam's server-side context compaction support:
//! - `enable_auto_compaction(trigger_tokens)` on `AgentBuilder`
//! - The Anthropic API summarizes older conversation content into a
//!   `compaction` block once input tokens cross the trigger threshold
//!   (minimum 50,000 tokens; beta `compact-2026-01-12`)
//! - The summary is retained in session history and replayed automatically —
//!   the API ignores everything before the last compaction block
//! - Compaction pass tokens are tracked separately in usage/cost accounting
//!   (`total_compaction_input_tokens` / `total_compaction_output_tokens`)
//! - `on_compaction` streaming hook for observability
//!
//! The agent is told to fetch several large archive volumes one at a time via
//! a tool. Each result adds ~16K tokens, so the conversation crosses the
//! 50K-token trigger mid-session and the provider compacts it server-side.
//!
//! Note: compaction requires a compaction-capable model (Claude Sonnet 4.6+,
//! Opus 4.6+, or the Claude 5 family). A full run consumes roughly 200K input
//! tokens (~$0.60 with Sonnet 4.6).
//!
//! Usage:
//!   export ANTHROPIC_API_KEY=sk-ant-...
//!   cargo run --example compaction-anthropic

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
    println!("🗜  Auto-Compaction Agent - Claude (Anthropic)\n");

    let agent = AgentBuilder::new("compaction-demo-anthropic")
        .provider(LlmProvider::Anthropic)
        .model("claude-sonnet-4-6")
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
