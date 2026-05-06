//! Minimal example showing Appam's async/context-aware stateful tools.

use appam::prelude::*;

#[derive(Clone)]
struct GreetingConfig {
    prefix: String,
}

#[derive(Default)]
struct GreetingSession {
    greeting_count: u64,
}

#[tool(description = "Greet a user while tracking the per-session greeting count")]
async fn greet_user(
    config: State<GreetingConfig>,
    session: SessionState<GreetingSession>,
    #[arg(description = "Name of the user to greet")] name: String,
) -> anyhow::Result<String> {
    let greeting_count = session.update(|state| {
        state.greeting_count += 1;
        state.greeting_count
    })?;

    Ok(format!(
        "{} {}, you are greeting number {} in this session.",
        config.prefix, name, greeting_count
    ))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = AgentBuilder::new("stateful-demo")
        .provider(LlmProvider::OpenAI)
        .model("gpt-5.5")
        .system_prompt(
            "You are a friendly demo agent. Use the greeting tool whenever the user asks to be greeted.",
        )
        .manage(GreetingConfig {
            prefix: "Hello".to_string(),
        })
        .session_state::<GreetingSession>()
        .async_tool(greet_user())
        .build()?;

    let _ = agent;
    Ok(())
}
