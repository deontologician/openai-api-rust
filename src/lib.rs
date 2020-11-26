#[macro_use]
extern crate derive_builder;

use reqwest::header::HeaderMap;
use thiserror::Error;

type Result<T> = std::result::Result<T, OpenAIError>;

#[allow(clippy::clippy::default_trait_access)]
pub mod api {

    use std::collections::HashMap;

    use serde::{Deserialize, Serialize};

    /// Container type. Used in the api, but not useful for clients of this library
    #[derive(Deserialize, Debug)]
    pub(super) struct Container<T> {
        pub data: Vec<T>,
    }

    /// Engine description type
    #[derive(Deserialize, Debug, Eq, PartialEq)]
    pub struct EngineInfo {
        pub id: Engine,
        pub owner: String,
        pub ready: bool,
    }

    /// Engine types, known and unknown
    #[derive(Deserialize, Serialize, Debug, Ord, PartialOrd, Eq, PartialEq, Copy, Clone)]
    #[serde(rename_all = "kebab-case")]
    pub enum Engine {
        Ada,
        Babbage,
        Curie,
        Davinci,
        #[serde(rename = "content-filter-alpha-c4")]
        ContentFilter,
        #[serde(other)]
        Other,
    }

    // Custom Display to lowercase things
    impl std::fmt::Display for Engine {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Engine::Ada => f.write_str("ada"),
                Engine::Babbage => f.write_str("babbage"),
                Engine::Curie => f.write_str("curie"),
                Engine::Davinci => f.write_str("davinci"),
                Engine::ContentFilter => f.write_str("content-filter-alpha-c4"),
                _ => panic!("Can't write out Other engine id"),
            }
        }
    }

    /// Options that affect the result
    #[derive(Serialize, Debug, Builder)]
    pub struct CompletionArgs {
        #[builder(setter(into), default = "\"<|endoftext|>\".into()")]
        prompt: String,
        #[builder(default = "16")]
        max_tokens: u64,
        #[builder(default = "1.0")]
        temperature: f64,
        #[builder(default = "1.0")]
        top_p: f64,
        #[builder(default = "1")]
        n: u64,
        #[builder(setter(strip_option), default)]
        logprobs: Option<u64>,
        #[builder(default = "false")]
        echo: bool,
        #[builder(setter(strip_option), default)]
        stop: Option<Vec<String>>,
        #[builder(default = "0.0")]
        presence_penalty: f64,
        #[builder(default = "0.0")]
        frequency_penalty: f64,
        #[builder(default)]
        logit_bias: HashMap<String, f64>,
    }
    /* {
          "stream": false, // SSE streams back results
          "best_of": Option<u64>, //cant be used with stream
       }
    */
    // TODO: add validators for the different arguments

    impl Default for CompletionArgs {
        fn default() -> Self {
            CompletionArgsBuilder::default()
                .build()
                .expect("Client error, invalid defaults")
        }
    }

    impl From<&str> for CompletionArgs {
        fn from(prompt_string: &str) -> Self {
            Self {
                prompt: prompt_string.into(),
                ..CompletionArgs::default()
            }
        }
    }

    /// Represents a non-streamed completion response
    #[derive(Deserialize, Debug)]
    pub struct Completion {
        id: String,
        object: String,
        created: u64,
        model: String,
        choices: Vec<Choice>,
    }

    /// Represents a single choice
    #[derive(Deserialize, Debug)]
    pub struct Choice {
        text: String,
        index: u64,
        logprobs: Option<LogProbs>,
        finish_reason: FinishReason,
    }

    /// Represents a logprobs subdocument
    #[derive(Deserialize, Debug)]
    pub struct LogProbs {
        tokens: Vec<String>,
        token_logprobs: Vec<Option<f64>>,
        top_logprobs: Vec<Option<HashMap<String, f64>>>,
        text_offset: Vec<u64>,
    }

    #[derive(Deserialize, Debug)]
    #[serde(rename_all = "kebab-case")]
    pub enum FinishReason {
        Length,
        Stop,
    }

    #[derive(Deserialize, Debug, Eq, PartialEq)]
    pub struct ErrorMessage {
        pub message: String,
        #[serde(rename = "type")]
        pub error_type: String,
    }

    #[derive(Deserialize, Debug)]
    pub struct ErrorWrapper {
        pub error: ErrorMessage,
    }
}

#[derive(Error, Debug)]
pub enum OpenAIError {
    #[error("Invalid secret key")]
    InvalidAPIKey {
        #[from]
        source: reqwest::header::InvalidHeaderValue,
    },
    #[error("API Returned an Error document")]
    APIError(api::ErrorMessage),
}

pub struct OpenAIClient {
    client: reqwest::Client,
    root: String,
}

impl OpenAIClient {
    /// Creates a new `OpenAIClient`
    ///
    /// # Errors
    /// `OpenAIError::InvalidAPIKey` if the api token has invalid characters
    pub fn new(token: &str) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", token))?,
        );
        Ok(Self {
            client: reqwest::Client::builder()
                .default_headers(headers)
                .build()
                .expect("Client library error. Should have constructed a valid http client."),
            root: "https://api.openai.com/v1".into(),
        })
    }

    /// Private helper for making gets
    async fn get<T: serde::de::DeserializeOwned>(&self, endpoint: &str) -> Result<T> {
        let url = &format!("{}/{}", self.root, endpoint);
        let response = self
            .client
            .get(url)
            .send()
            .await
            .expect("Client error. Should have passed a valid url");
        if response.status() != 200 {
            return Err(OpenAIError::APIError(
                response
                    .json::<api::ErrorWrapper>()
                    .await
                    .expect("The API has returned something funky")
                    .error,
            ));
        }
        Ok(response.json::<T>().await.unwrap())
    }

    /// Lists the currently available engines.
    ///
    /// Provides basic information about each one such as the owner and availability.
    ///
    /// # Errors
    /// - `OpenAIError::APIError` if the server returns an error
    /// - `OpenAIError::ServerFormatError` if the json response wasn't parseable (most
    ///    likely a bug in this client, please report it)
    pub async fn engines(&self) -> Result<Vec<api::EngineInfo>> {
        self.get("engines").await.map(|r: api::Container<_>| r.data)
    }

    /// Retrieves an engine instance
    /// Provides basic information about the engine such as the owner and availability.
    ///
    /// # Errors
    /// - `OpenAIError::APIError` if the server returns an error
    /// - `OpenAIError::ServerFormatError` if the json response wasn't parseable (most
    ///    likely a bug in this client, please report it)
    pub async fn engine(&self, engine: api::Engine) -> Result<api::EngineInfo> {
        self.get(&format!("engines/{}", engine)).await
    }

    // Private helper to generate post requests. Needs to be a bit more flexible than
    // get because it should support SSE eventually
    async fn post<B: serde::ser::Serialize>(
        &self,
        endpoint: &str,
        body: B,
    ) -> Result<reqwest::Response> {
        let url = &format!("{}/{}", self.root, endpoint);
        let response = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await
            .expect("Client library error, json failed to parse");
        if response.status() != 200 {
            return Err(OpenAIError::APIError(
                response
                    .json::<api::ErrorWrapper>()
                    .await
                    .expect("The API has returned something funky")
                    .error,
            ));
        }
        Ok(response)
    }
    /// Get predicted completion of the prompt
    ///
    /// # Errors
    ///  - `OpenAIError::APIError` if the api returns an error
    pub async fn complete(
        &self,
        engine: api::Engine,
        prompt: impl Into<api::CompletionArgs>,
    ) -> Result<api::Completion> {
        Ok(self
            .post(&format!("engines/{}/completions", engine), prompt.into())
            .await?
            //.text()
            .json()
            .await
            .expect("Client error. JSON didn't parse correctly."))
    }
}

#[cfg(test)]
mod unit {

    use crate::{api, OpenAIClient, OpenAIError};

    fn mocked_client() -> OpenAIClient {
        let mut client = OpenAIClient::new("bogus").unwrap();
        client.root = mockito::server_url();
        client
    }

    #[test]
    fn can_create_client() {
        let _c = mocked_client();
    }

    #[test]
    fn parse_engine_info() -> Result<(), Box<dyn std::error::Error>> {
        let example = r#"{
            "id": "ada",
            "object": "engine",
            "owner": "openai",
            "ready": true
        }"#;
        let ei: api::EngineInfo = serde_json::from_str(example)?;
        assert_eq!(
            ei,
            api::EngineInfo {
                id: api::Engine::Ada,
                owner: "openai".into(),
                ready: true,
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn parse_engines() -> crate::Result<()> {
        use api::{Engine, EngineInfo};
        let _m = mockito::mock("GET", "/engines")
            .with_status(200)
            .with_header("content-type", "text/json")
            .with_body(
                r#"{
            "object": "list",
            "data": [
              {
                "id": "ada",
                "object": "engine",
                "owner": "openai",
                "ready": true
              },
              {
                "id": "babbage",
                "object": "engine",
                "owner": "openai",
                "ready": true
              },
              {
                "id": "experimental-engine-v7",
                "object": "engine",
                "owner": "openai",
                "ready": false
              },
              {
                "id": "curie",
                "object": "engine",
                "owner": "openai",
                "ready": true
              },
              {
                "id": "davinci",
                "object": "engine",
                "owner": "openai",
                "ready": true
              },
              {
                 "id": "content-filter-alpha-c4",
                 "object": "engine",
                 "owner": "openai",
                 "ready": true
              }
            ]
          }"#,
            )
            .create();
        let expected = vec![
            EngineInfo {
                id: Engine::Ada,
                owner: "openai".into(),
                ready: true,
            },
            EngineInfo {
                id: Engine::Babbage,
                owner: "openai".into(),
                ready: true,
            },
            EngineInfo {
                id: Engine::Other,
                owner: "openai".into(),
                ready: false,
            },
            EngineInfo {
                id: Engine::Curie,
                owner: "openai".into(),
                ready: true,
            },
            EngineInfo {
                id: Engine::Davinci,
                owner: "openai".into(),
                ready: true,
            },
            EngineInfo {
                id: Engine::ContentFilter,
                owner: "openai".into(),
                ready: true,
            },
        ];
        let response = mocked_client().engines().await?;
        assert_eq!(response, expected);
        Ok(())
    }

    #[tokio::test]
    async fn engine_error_response() -> crate::Result<()> {
        let _m = mockito::mock("GET", "/engines/davinci")
            .with_status(404)
            .with_header("content-type", "text/json")
            .with_body(
                r#"{
                    "error": {
                        "code": null,
                        "message": "Some kind of error happened",
                        "type": "some_error_type"
                    }
                }"#,
            )
            .create();
        let expected = api::ErrorMessage {
            message: "Some kind of error happened".into(),
            error_type: "some_error_type".into(),
        };
        let response = mocked_client().engine(api::Engine::Davinci).await;
        if let Result::Err(OpenAIError::APIError(msg)) = response {
            assert_eq!(expected, msg);
        }
        Ok(())
    }
}
#[cfg(test)]
mod integration {

    use api::ErrorMessage;

    use crate::{OpenAIClient, OpenAIError, api};

    /// Used by tests to get a client to the actual api
    fn get_client() -> OpenAIClient {
        let sk = std::env::var("OPENAI_SK").expect(
            "To run integration tests, you must put set the OPENAI_SK env var to your api token",
        );
        OpenAIClient::new(&sk).expect("client build failed")
    }

    #[tokio::test]
    async fn can_get_engines() {
        let client = get_client();
        client.engines().await.unwrap();
    }
    #[tokio::test]
    async fn can_get_engine() {
        let client = get_client();
        let result = client.engine(api::Engine::Ada).await;
        match result {
            Err(OpenAIError::APIError(ErrorMessage{message, error_type})) => {
                assert_eq!(message, "No engine with that ID: ada");
                assert_eq!(error_type, "invalid_request_error");
            }
            _ => {panic!("Expected an error message, got {:?}", result)}
        }
    }

    #[tokio::test]
    async fn complete_string() -> crate::Result<()> {
        let client = get_client();
        client.complete(api::Engine::Ada, "Hey there").await?;
        Ok(())
    }

    #[tokio::test]
    async fn complete_explicit_params() -> crate::Result<()> {
        let client = get_client();
        let args = api::CompletionArgsBuilder::default()
            .prompt("Once upon a time,")
            .max_tokens(10)
            .temperature(0.5)
            .top_p(0.5)
            .n(1)
            .logprobs(3)
            .echo(false)
            .stop(vec!["\n".into()])
            .presence_penalty(0.5)
            .frequency_penalty(0.5)
            .logit_bias(maplit::hashmap! {
                "1".into() => 1.0,
                "23".into() => 0.0,
            })
            .build()
            .expect("Build should have succeeded");
        client.complete(api::Engine::Ada, args).await?;
        Ok(())
    }
}
