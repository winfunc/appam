//! Auto-compaction agent — Azure OpenAI Responses API
//!
//! Demonstrates appam's server-side context compaction support on Azure:
//! - `enable_auto_compaction(trigger_tokens)` on `AgentBuilder`
//! - Azure's Responses API accepts the same `context_management` compaction
//!   entries as direct OpenAI and emits the same encrypted `compaction` item
//! - The encrypted item is retained in session history and replayed
//!   automatically; older items are pruned client-side per the API contract
//!
//! Usage:
//!   export AZURE_OPENAI_API_KEY=...          # or OPENAI_API_KEY fallback
//!   export AZURE_OPENAI_RESOURCE=my-resource # Azure resource name
//!   export AZURE_OPENAI_MODEL=gpt-5.4-mini   # deployment name (optional)
//!   cargo run --example compaction-azure-openai

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
    println!("🗜  Auto-Compaction Agent - Azure OpenAI Responses API\n");

    let resource_name = std::env::var("AZURE_OPENAI_RESOURCE")
        .or_else(|_| std::env::var("AZURE_OPENAI_RESOURCE_NAME"))
        .unwrap_or_else(|_| "example-resource".to_string());
    let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
        .unwrap_or_else(|_| "2025-04-01-preview".to_string());
    let model = std::env::var("AZURE_OPENAI_MODEL").unwrap_or_else(|_| "gpt-5.4-mini".to_string());

    let agent = AgentBuilder::new("compaction-demo-azure-openai")
        .provider(LlmProvider::AzureOpenAI {
            resource_name: resource_name.clone(),
            api_version,
        })
        .model(&model)
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

    println!("✓ Deployment: {model} @ {resource_name}");
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
             the 16K-token threshold. Increase the number of sections."
        );
    } else {
        println!("\n✓ Server-side compaction verified end to end");
    }

    Ok(())
}
