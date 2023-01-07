# openai-api-rust
![docs](https://docs.rs/openai-api/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/openai-api.svg)](https://crates.io/crates/openai-api)
![build](https://github.com/deontologician/openai-api-rust/workflows/Continuous%20Integration/badge.svg)

A simple rust client for OpenAI API.

Has a few conveniences, but is mostly at the level of the API itself.

# Installation

```
$ cargo add openai-api
```

# Quickstart

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_token = std::env::var("OPENAI_SK")?;
    let client = openai_api::Client::new(&api_token);
    let prompt = String::from("Once upon a time,");
    println!(
        "{}{}",
        prompt,
        client.complete(prompt.as_str()).await?
    );
    Ok(())
}
```
# Basic Usage

## Creating a completion

For simple demos and debugging, you can do a completion and use the `Display` instance of a `Completion` object to convert it to a string:

```rust
let response = client.complete_prompt("Once upon a time").await?;
println!("{}", response);
```

To configure the prompt more explicitly, you can use the `CompletionArgs` builder:

```rust
let args = openai_api::api::CompletionArgs::builder()
        .prompt("Once upon a time,")
        .engine("text-davinci-003")
        .max_tokens(20)
        .temperature(0.7)
        .top_p(0.9)
        .stop(vec!["\n".into()]);
let completion = client.complete_prompt(args).await?;
println!("Response: {}", response.choices[0].text);
println!("Model used: {}", response.model);
```

See [examples/](./examples)
