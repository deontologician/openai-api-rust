///! Example that prints out a story from a prompt. Used in the readme.
use openai_api::Client;

#[cfg(not(feature = "is_sync"))]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_token = std::env::var("OPENAI_SK")?;
    let client = Client::new(&api_token);
    let prompt = "Once upon a time,";
    println!(
        "{}{}",
        prompt,
        client.complete_prompt(prompt).await?
    );
    Ok(())
}

#[cfg(feature = "is_sync")]
fn main() -> anyhow::Result<()> {
    let api_token = std::env::var("OPENAI_SK")?;
    let client = Client::new(&api_token);
    let prompt = "Once upon a time,";
    println!(
        "{}{}",
        prompt,
        client.complete_prompt(prompt)?
    );
    Ok(())
}
