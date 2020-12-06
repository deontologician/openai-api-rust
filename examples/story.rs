///! Example that prints out a story from a prompt. Used in the readme.
use openai_api::OpenAIClient;

#[tokio::main]
async fn main() {
    let api_token = std::env::var("OPENAI_SK").unwrap();
    let client = OpenAIClient::new(&api_token);
    let prompt = String::from("Once upon a time,");
    println!(
        "{}{}",
        prompt,
        client.complete(prompt.as_str()).await.unwrap()
    );
}
