///! Example that prints out a story from a prompt. Used in the readme.
use openai_api::Client;

#[tokio::main]
async fn main() {
    let api_token = std::env::var("OPENAI_SK").unwrap();
    let client = Client::new(&api_token);
    let prompt = String::from("Once upon a time,");
    println!(
        "{}{}",
        prompt,
        client.complete_prompt(prompt.as_str()).await.unwrap()
    );
}
