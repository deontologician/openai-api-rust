use openai_api::{api::CompletionArgs, Client};

const START_PROMPT: &str = "
The following is a conversation with an AI assistant.
The assistant is helpful, creative, clever, and very friendly.
Human: Hello, who are you?
AI: I am an AI. How can I help you today?";
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_token = std::env::var("OPENAI_SK")?;
    let client = Client::new(&api_token);
    let mut context = String::from(START_PROMPT);
    let args = CompletionArgs::builder()
        .engine("davinci")
        .max_tokens(45)
        .stop(vec!["\n".into()])
        .top_p(0.5)
        .temperature(0.9)
        .frequency_penalty(0.5);
    println!("\x1b[1;36m{}\x1b[1;0m", context.split('\n').last().unwrap());
    loop {
        context.push_str("\nHuman: ");
        if let Err(e) = std::io::stdin().read_line(&mut context) {
            eprintln!("Error: {}", e);
            break;
        }
        context.push_str("\nAI: ");
        match client
            .complete_prompt(args.prompt(context.as_str()).build()?)
            .await
        {
            Ok(completion) => {
                println!("\x1b[1;36m{}\x1b[1;0m", completion);
                context.push_str(&completion.choices[0].text);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }
    println!("Full conversation:\n{}", context);
    Ok(())
}
