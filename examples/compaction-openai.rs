//! Auto-compaction agent — OpenAI Responses API
//!
//! Demonstrates appam's server-side context compaction support:
//! - `enable_auto_compaction(trigger_tokens)` on `AgentBuilder`
//! - The Responses API compacts the conversation into an encrypted
//!   `compaction` item once the rendered context crosses the threshold
//!   (minimum 1,000 tokens; sent as `context_management`)
//! - The encrypted item is retained in session history and replayed
//!   automatically; older items are pruned client-side per the API contract
//! - Works with appam's default stateless mode (`store: false`, ZDR-friendly)
//! - `on_compaction` streaming hook for observability (OpenAI summaries are
//!   opaque, so no summary text is exposed)
//!
//! The agent fetches several ~4K-token catalog sections via a tool, crossing
//! the compaction threshold mid-session.
//!
//! Tip: choose a threshold well above the size of a typical compacted window
//! (a few thousand tokens). Thresholds close to the minimum cause the server
//! to re-compact on every reasoning step.
//!
//! Usage:
//!   export OPENAI_API_KEY=sk-...
//!   cargo run --example compaction-openai

use anyhow::Result;
use appam::llm::UnifiedContentBlock;
use appam::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// Tool Definitions
// ============================================================================

#[derive(Serialize)]
struct CatalogSection {
    section: u32,
    title: String,
    body: String,
}

/// Return a deterministic ~4K-token document so a few tool calls push the
/// conversation past the compaction threshold.
#[tool(description = "Fetch one section of the star catalog by number (1-5)")]
fn fetch_catalog(
    #[arg(description = "Section number to fetch, between 1 and 5")] section: u32,
) -> Result<CatalogSection> {
    let titles = [
        "Main Sequence Stars of the Orion Arm",
        "Variable Stars and Binary Systems",
        "Deep Sky Objects and Nebulae",
        "Star Clusters of the Southern Sky",
        "Exoplanet Host Stars",
    ];
    let title = titles
        .get((section.saturating_sub(1)) as usize % titles.len())
        .unwrap_or(&titles[0])
        .to_string();

    // ~400 entries * ~10 tokens each ≈ 4K tokens per section
    let body = (1..=400)
        .map(|entry| {
            format!(
                "Record {entry:03} (sec {section}): catalogued object at field position \
                 {position}; magnitude nominal, spectra archived.",
                position = entry * 13 % 360,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    Ok(CatalogSection {
        section,
        title,
        body,
    })
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    println!("🗜  Auto-Compaction Agent - OpenAI Responses API\n");

    let agent = AgentBuilder::new("compaction-demo-openai")
        .provider(LlmProvider::OpenAI)
        .model("gpt-5-mini")
        .system_prompt(
            "You are a star catalog librarian. Fetch catalog sections with the \
             fetch_catalog tool exactly as instructed, one section per tool call. \
             Do not summarize until every requested section has been fetched.",
        )
        // Server-side compaction once the rendered context crosses ~16K
        // tokens. Keep this comfortably above the size of a compacted window
        // so the server does not re-compact on every reasoning step.
        .enable_auto_compaction(16_000)
        .with_tool(Arc::new(fetch_catalog()))
        .max_tokens(4096)
        .build()?;

    println!("✓ Auto-compaction enabled (trigger: 16K tokens)");
    println!("✓ Tool: fetch_catalog (~4K tokens per section)\n");

    let compaction_count = Arc::new(AtomicUsize::new(0));
    let compaction_count_stream = Arc::clone(&compaction_count);

    let session = agent
        .stream(
            "Fetch catalog sections 1, 2, 3, 4, and 5 one at a time using the \
             fetch_catalog tool. After fetching ALL five sections, reply with a \
             single short paragraph naming each section's title.",
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
        .on_compaction(move |provider, _summary| {
            compaction_count_stream.fetch_add(1, Ordering::Relaxed);
            println!("\n◈ Context compacted by {provider} (encrypted summary)");
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
        .filter(|block| {
            matches!(
                block,
                UnifiedContentBlock::Compaction {
                    encrypted_content: Some(_),
                    ..
                }
            )
        })
        .count();

    println!("\n\n===== Session Report =====");
    println!("Compaction events observed:   {compactions}");
    println!("Compaction items in history:  {history_compaction_blocks}");
    if let Some(usage) = &session.usage {
        println!("\n{}", usage.format_detailed());
    }

    if compactions == 0 {
        println!(
            "\n⚠ No compaction occurred — the conversation may not have crossed \
             the 8K-token threshold. Increase the number of sections."
        );
    } else {
        println!("\n✓ Server-side compaction verified end to end");
    }

    Ok(())
}
