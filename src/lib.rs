///! `OpenAI` API client library
#[macro_use]
extern crate derive_builder;

use thiserror::Error;

type Result<T> = std::result::Result<T, Error>;

#[allow(clippy::default_trait_access)]
pub mod api {
    //! Data types corresponding to requests and responses from the API
    use std::{collections::HashMap, fmt::Display};

    use super::Client;
    use serde::{Deserialize, Serialize};

    /// Container type. Used in the api, but not useful for clients of this library
    #[derive(Deserialize, Debug)]
    pub(super) struct Container<T> {
        pub data: Vec<T>,
    }

    /// Engine description type
    #[derive(Deserialize, Debug, Eq, PartialEq, Clone)]
    pub struct EngineInfo {
        pub id: Engine,
        pub owner: String,
        pub ready: bool,
    }

    /// Engine types, known and unknown
    #[derive(Deserialize, Serialize, Debug, Ord, PartialOrd, Eq, PartialEq, Copy, Clone)]
    #[serde(rename_all = "kebab-case")]
    #[non_exhaustive] // prevent clients from matching on every option
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
    #[derive(Serialize, Debug, Builder, Clone)]
    pub struct CompletionArgs {
        #[builder(setter(into), default = "\"<|endoftext|>\".into()")]
        prompt: String,
        #[builder(default = "Engine::Davinci")]
        #[serde(skip_serializing)]
        pub(super) engine: Engine,
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

    impl From<&str> for CompletionArgs {
        fn from(prompt_string: &str) -> Self {
            Self {
                prompt: prompt_string.into(),
                ..CompletionArgsBuilder::default()
                    .build()
                    .expect("default should build")
            }
        }
    }

    impl CompletionArgs {
        /// Build a `CompletionArgs` from the defaults
        #[must_use]
        pub fn builder() -> CompletionArgsBuilder {
            CompletionArgsBuilder::default()
        }

        /// Request a completion from the api
        ///
        /// # Errors
        /// `Error::APIError` if the api returns an error
        #[cfg(feature = "async")]
        pub async fn complete_prompt(self, client: &Client) -> super::Result<Completion> {
            client.complete_prompt(self).await
        }

        #[cfg(feature = "sync")]
        pub fn complete_prompt_sync(self, client: &Client) -> super::Result<Completion> {
            client.complete_prompt_sync(self)
        }
    }

    impl CompletionArgsBuilder {
        /// Request a completion from the api
        ///
        /// # Errors
        /// `Error::BadArguments` if the arguments to complete are not valid
        /// `Error::APIError` if the api returns an error
        #[cfg(feature = "async")]
        pub async fn complete_prompt(&self, client: &Client) -> super::Result<Completion> {
            client.complete_prompt(self.build()?).await
        }

        #[cfg(feature = "sync")]
        pub fn complete_prompt_sync(&self, client: &Client) -> super::Result<Completion> {
            client.complete_prompt_sync(self.build()?)
        }
    }

    /// Represents a non-streamed completion response
    #[derive(Deserialize, Debug, Clone)]
    pub struct Completion {
        /// Completion unique identifier
        pub id: String,
        /// Unix timestamp when the completion was generated
        pub created: u64,
        /// Exact model type and version used for the completion
        pub model: String,
        /// Timestamp
        pub choices: Vec<Choice>,
    }

    impl std::fmt::Display for Completion {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.choices[0])
        }
    }

    /// A single completion result
    #[derive(Deserialize, Debug, Clone)]
    pub struct Choice {
        /// The text of the completion. Will contain the prompt if echo is True.
        pub text: String,
        /// Offset in the result where the completion began. Useful if using echo.
        pub index: u64,
        /// If requested, the log probabilities of the completion tokens
        pub logprobs: Option<LogProbs>,
        /// Why the completion ended when it did
        pub finish_reason: FinishReason,
    }

    impl std::fmt::Display for Choice {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.text.fmt(f)
        }
    }

    /// Represents a logprobs subdocument
    #[derive(Deserialize, Debug, Clone)]
    pub struct LogProbs {
        pub tokens: Vec<String>,
        pub token_logprobs: Vec<Option<f64>>,
        pub top_logprobs: Vec<Option<HashMap<String, f64>>>,
        pub text_offset: Vec<u64>,
    }

    /// Reason a prompt completion finished.
    #[derive(Deserialize, Debug, Eq, PartialEq, Clone, Copy)]
    #[non_exhaustive]
    pub enum FinishReason {
        /// The maximum length was reached
        #[serde(rename = "length")]
        MaxTokensReached,
        /// The stop token was encountered
        #[serde(rename = "stop")]
        StopSequenceReached,
    }

    /// Error response object from the server
    #[derive(Deserialize, Debug, Eq, PartialEq, Clone)]
    pub struct ErrorMessage {
        pub message: String,
        #[serde(rename = "type")]
        pub error_type: String,
    }

    impl Display for ErrorMessage {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.message.fmt(f)
        }
    }

    /// API-level wrapper used in deserialization
    #[derive(Deserialize, Debug)]
    pub(crate) struct ErrorWrapper {
        pub error: ErrorMessage,
    }
}

/// This library's main `Error` type.
#[derive(Error, Debug)]
pub enum Error {
    /// An error returned by the API itself
    #[error("API returned an Error: {}", .0.message)]
    APIError(api::ErrorMessage),
    /// An error the client discovers before talking to the API
    #[error("Bad arguments: {0}")]
    BadArguments(String),
    /// Network / protocol related errors
    #[cfg(feature = "async")]
    #[error("Error at the protocol level: {0}")]
    AsyncProtocolError(surf::Error),
    #[cfg(feature = "sync")]
    #[error("Error at the protocol level, sync client")]
    SyncProtocolError(ureq::Error),
}

impl From<api::ErrorMessage> for Error {
    fn from(e: api::ErrorMessage) -> Self {
        Error::APIError(e)
    }
}

impl From<String> for Error {
    fn from(e: String) -> Self {
        Error::BadArguments(e)
    }
}

#[cfg(feature = "async")]
impl From<surf::Error> for Error {
    fn from(e: surf::Error) -> Self {
        Error::AsyncProtocolError(e)
    }
}

#[cfg(feature = "sync")]
impl From<ureq::Error> for Error {
    fn from(e: ureq::Error) -> Self {
        Error::SyncProtocolError(e)
    }
}

/// Authentication middleware
#[cfg(feature = "async")]
struct BearerToken {
    token: String,
}

#[cfg(feature = "async")]
impl std::fmt::Debug for BearerToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Get the first few characters to help debug, but not accidentally log key
        write!(
            f,
            r#"Bearer {{ token: "{}" }}"#,
            self.token.get(0..8).ok_or(std::fmt::Error)?
        )
    }
}

#[cfg(feature = "async")]
impl BearerToken {
    fn new(token: &str) -> Self {
        Self {
            token: String::from(token),
        }
    }
}

#[cfg(feature = "async")]
#[surf::utils::async_trait]
impl surf::middleware::Middleware for BearerToken {
    async fn handle(
        &self,
        mut req: surf::Request,
        client: surf::Client,
        next: surf::middleware::Next<'_>,
    ) -> surf::Result<surf::Response> {
        log::debug!("Request: {:?}", req);
        req.insert_header("Authorization", format!("Bearer {}", self.token));
        let response: surf::Response = next.run(req, client).await?;
        log::debug!("Response: {:?}", response);
        Ok(response)
    }
}

#[cfg(feature = "async")]
fn async_client(token: &str, base_url: &str) -> surf::Client {
    let mut async_client = surf::client();
    async_client.set_base_url(surf::Url::parse(base_url).expect("Static string should parse"));
    async_client.with(BearerToken::new(token))
}

#[cfg(feature = "sync")]
fn sync_client(token: &str) -> ureq::Agent {
    ureq::agent().auth_kind("Bearer", token).build()
}

/// Client object. Must be constructed to talk to the API.
#[derive(Debug, Clone)]
pub struct Client {
    #[cfg(feature = "async")]
    async_client: surf::Client,
    #[cfg(feature = "sync")]
    sync_client: ureq::Agent,
    #[cfg(feature = "sync")]
    base_url: String,
}

impl Client {
    // Creates a new `Client` given an api token
    #[must_use]
    pub fn new(token: &str) -> Self {
        let base_url = String::from("https://api.openai.com/v1/");
        Self {
            #[cfg(feature = "async")]
            async_client: async_client(token, &base_url),
            #[cfg(feature = "sync")]
            sync_client: sync_client(token),
            #[cfg(feature = "sync")]
            base_url,
        }
    }

    // Allow setting the api root in the tests
    #[cfg(test)]
    fn set_api_root(mut self, base_url: &str) -> Self {
        #[cfg(feature = "async")]
        {
            self.async_client.set_base_url(
                surf::Url::parse(base_url).expect("static URL expected to parse correctly"),
            );
        }
        #[cfg(feature = "sync")]
        {
            self.base_url = String::from(base_url);
        }
        self
    }

    /// Private helper for making gets
    #[cfg(feature = "async")]
    async fn get<T>(&self, endpoint: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let mut response = self.async_client.get(endpoint).await?;
        if let surf::StatusCode::Ok = response.status() {
            Ok(response.body_json::<T>().await?)
        } else {
            let err = response.body_json::<api::ErrorWrapper>().await?.error;
            Err(Error::APIError(err))
        }
    }

    #[cfg(feature = "sync")]
    fn get_sync<T>(&self, endpoint: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let response = dbg!(self
            .sync_client
            .get(&format!("{}{}", self.base_url, endpoint)))
        .call();
        if let 200 = response.status() {
            Ok(response
                .into_json_deserialize()
                .expect("Bug: client couldn't deserialize api response"))
        } else {
            let err = response
                .into_json_deserialize::<api::ErrorWrapper>()
                .expect("Bug: client couldn't deserialize api error response")
                .error;
            Err(Error::APIError(err))
        }
    }

    /// Lists the currently available engines.
    ///
    /// Provides basic information about each one such as the owner and availability.
    ///
    /// # Errors
    /// - `Error::APIError` if the server returns an error
    #[cfg(feature = "async")]
    pub async fn engines(&self) -> Result<Vec<api::EngineInfo>> {
        self.get("engines").await.map(|r: api::Container<_>| r.data)
    }

    /// Lists the currently available engines.
    ///
    /// Provides basic information about each one such as the owner and availability.
    ///
    /// # Errors
    /// - `Error::APIError` if the server returns an error
    #[cfg(feature = "sync")]
    pub fn engines_sync(&self) -> Result<Vec<api::EngineInfo>> {
        self.get_sync("engines").map(|r: api::Container<_>| r.data)
    }

    /// Retrieves an engine instance
    ///
    /// Provides basic information about the engine such as the owner and availability.
    ///
    /// # Errors
    /// - `Error::APIError` if the server returns an error
    #[cfg(feature = "async")]
    pub async fn engine(&self, engine: api::Engine) -> Result<api::EngineInfo> {
        self.get(&format!("engines/{}", engine)).await
    }

    #[cfg(feature = "sync")]
    pub fn engine_sync(&self, engine: api::Engine) -> Result<api::EngineInfo> {
        self.get_sync(&format!("engines/{}", engine))
    }

    // Private helper to generate post requests. Needs to be a bit more flexible than
    // get because it should support SSE eventually
    #[cfg(feature = "async")]
    async fn post<B, R>(&self, endpoint: &str, body: B) -> Result<R>
    where
        B: serde::ser::Serialize,
        R: serde::de::DeserializeOwned,
    {
        let mut response = self
            .async_client
            .post(endpoint)
            .body(surf::Body::from_json(&body)?)
            .await?;
        match response.status() {
            surf::StatusCode::Ok => Ok(response.body_json::<R>().await?),
            _ => Err(Error::APIError(
                response
                    .body_json::<api::ErrorWrapper>()
                    .await
                    .expect("The API has returned something funky")
                    .error,
            )),
        }
    }

    #[cfg(feature = "sync")]
    fn post_sync<B, R>(&self, endpoint: &str, body: B) -> Result<R>
    where
        B: serde::ser::Serialize,
        R: serde::de::DeserializeOwned,
    {
        let response = self
            .sync_client
            .post(&format!("{}{}", self.base_url, endpoint))
            .send_json(
                serde_json::to_value(body).expect("Bug: client couldn't serialize its own type"),
            );
        match response.status() {
            200 => Ok(response
                .into_json_deserialize()
                .expect("Bug: client couldn't deserialize api response")),
            _ => Err(Error::APIError(
                response
                    .into_json_deserialize::<api::ErrorWrapper>()
                    .expect("Bug: client couldn't deserialize api error response")
                    .error,
            )),
        }
    }

    /// Get predicted completion of the prompt
    ///
    /// # Errors
    ///  - `Error::APIError` if the api returns an error
    #[cfg(feature = "async")]
    pub async fn complete_prompt(
        &self,
        prompt: impl Into<api::CompletionArgs>,
    ) -> Result<api::Completion> {
        let args = prompt.into();
        Ok(self
            .post(&format!("engines/{}/completions", args.engine), args)
            .await?)
    }

    /// Get predicted completion of the prompt synchronously
    ///
    /// # Error
    /// - `Error::APIError` if the api returns an error
    #[cfg(feature = "sync")]
    pub fn complete_prompt_sync(
        &self,
        prompt: impl Into<api::CompletionArgs>,
    ) -> Result<api::Completion> {
        let args = prompt.into();
        self.post_sync(&format!("engines/{}/completions", args.engine), args)
    }
}

// TODO: add a macro to de-boilerplate the sync and async tests

#[allow(unused_macros)]
macro_rules! async_test {
    ($test_name: ident, $test_body: block) => {
        #[cfg(feature = "async")]
        #[tokio::test]
        async fn $test_name() -> crate::Result<()> {
            $test_body;
            Ok(())
        }
    };
}

#[allow(unused_macros)]
macro_rules! sync_test {
    ($test_name: ident, $test_body: expr) => {
        #[cfg(feature = "sync")]
        #[test]
        fn $test_name() -> crate::Result<()> {
            $test_body;
            Ok(())
        }
    };
}

#[cfg(test)]
mod unit {

    use mockito::Mock;

    use crate::{
        api::{self, Completion, CompletionArgs, Engine, EngineInfo},
        Client, Error,
    };

    fn mocked_client() -> Client {
        let _ = env_logger::builder().is_test(true).try_init();
        Client::new("bogus").set_api_root(&format!("{}/", mockito::server_url()))
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

    fn mock_engines() -> (Mock, Vec<EngineInfo>) {
        let mock = mockito::mock("GET", "/engines")
            .with_status(200)
            .with_header("content-type", "application/json")
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
        (mock, expected)
    }

    async_test!(parse_engines_async, {
        let (_m, expected) = mock_engines();
        let response = mocked_client().engines().await?;
        assert_eq!(response, expected);
    });

    sync_test!(parse_engines_sync, {
        let (_m, expected) = mock_engines();
        let response = mocked_client().engines_sync()?;
        assert_eq!(response, expected);
    });

    fn mock_engine() -> (Mock, api::ErrorMessage) {
        let mock = mockito::mock("GET", "/engines/davinci")
            .with_status(404)
            .with_header("content-type", "application/json")
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
        (mock, expected)
    }

    async_test!(engine_error_response_async, {
        let (_m, expected) = mock_engine();
        let response = mocked_client().engine(api::Engine::Davinci).await;
        if let Result::Err(Error::APIError(msg)) = response {
            assert_eq!(expected, msg);
        }
    });

    sync_test!(engine_error_response_sync, {
        let (_m, expected) = mock_engine();
        let response = mocked_client().engine_sync(api::Engine::Davinci);
        if let Result::Err(Error::APIError(msg)) = response {
            assert_eq!(expected, msg);
        }
    });
    fn mock_completion() -> crate::Result<(Mock, CompletionArgs, Completion)> {
        let mock = mockito::mock("POST", "/engines/davinci/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
                "object": "text_completion",
                "created": 1589478378,
                "model": "davinci:2020-05-03",
                "choices": [
                    {
                    "text": " there was a girl who",
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": "length"
                    }
                ]
                }"#,
            )
            .expect(1)
            .create();
        let args = api::CompletionArgs::builder()
            .engine(api::Engine::Davinci)
            .prompt("Once upon a time")
            .max_tokens(5)
            .temperature(1.0)
            .top_p(1.0)
            .n(1)
            .stop(vec!["\n".into()])
            .build()?;
        let expected = api::Completion {
            id: "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7".into(),
            created: 1589478378,
            model: "davinci:2020-05-03".into(),
            choices: vec![api::Choice {
                text: " there was a girl who".into(),
                index: 0,
                logprobs: None,
                finish_reason: api::FinishReason::MaxTokensReached,
            }],
        };
        Ok((mock, args, expected))
    }

    // Defines boilerplate here. The Completion can't derive Eq since it contains
    // floats in various places.
    fn assert_completion_equal(a: Completion, b: Completion) {
        assert_eq!(a.model, b.model);
        assert_eq!(a.id, b.id);
        assert_eq!(a.created, b.created);
        let (a_choice, b_choice) = (&a.choices[0], &b.choices[0]);
        assert_eq!(a_choice.text, b_choice.text);
        assert_eq!(a_choice.index, b_choice.index);
        assert!(a_choice.logprobs.is_none());
        assert_eq!(a_choice.finish_reason, b_choice.finish_reason);
    }

    async_test!(completion_args_async, {
        let (m, args, expected) = mock_completion()?;
        let response = mocked_client().complete_prompt(args).await?;
        assert_completion_equal(response, expected);
        m.assert();
    });

    sync_test!(completion_args_sync, {
        let (m, args, expected) = mock_completion()?;
        let response = mocked_client().complete_prompt_sync(args)?;
        assert_completion_equal(response, expected);
        m.assert();
    });
}
#[cfg(test)]
mod integration {
    use crate::{
        api::{self, Completion},
        Client, Error,
    };
    /// Used by tests to get a client to the actual api
    fn get_client() -> Client {
        let _ = env_logger::builder().is_test(true).try_init();
        let sk = std::env::var("OPENAI_SK").expect(
            "To run integration tests, you must put set the OPENAI_SK env var to your api token",
        );
        Client::new(&sk)
    }

    async_test!(can_get_engines_async, {
        let client = get_client();
        client.engines().await?
    });

    sync_test!(can_get_engines_sync, {
        let client = get_client();
        let engines = client
            .engines_sync()?
            .into_iter()
            .map(|ei| ei.id)
            .collect::<Vec<_>>();
        assert!(engines.contains(&api::Engine::Ada));
        assert!(engines.contains(&api::Engine::Babbage));
        assert!(engines.contains(&api::Engine::Curie));
        assert!(engines.contains(&api::Engine::Davinci));
    });

    fn assert_expected_engine_failure<T>(result: Result<T, Error>)
    where
        T: std::fmt::Debug,
    {
        match result {
            Err(Error::APIError(api::ErrorMessage {
                message,
                error_type,
            })) => {
                assert_eq!(message, "No engine with that ID: ada");
                assert_eq!(error_type, "invalid_request_error");
            }
            _ => {
                panic!("Expected an error message, got {:?}", result)
            }
        }
    }
    async_test!(can_get_engine_async, {
        let client = get_client();
        assert_expected_engine_failure(client.engine(api::Engine::Ada).await);
    });

    sync_test!(can_get_engine_sync, {
        let client = get_client();
        assert_expected_engine_failure(client.engine_sync(api::Engine::Ada));
    });

    async_test!(complete_string_async, {
        let client = get_client();
        client.complete_prompt("Hey there").await?;
    });

    sync_test!(complete_string_sync, {
        let client = get_client();
        client.complete_prompt_sync("Hey there")?;
    });

    fn create_args() -> api::CompletionArgs {
        api::CompletionArgsBuilder::default()
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
            .expect("Bug: build should succeed")
    }
    async_test!(complete_explicit_params_async, {
        let client = get_client();
        let args = create_args();
        client.complete_prompt(args).await?;
    });

    sync_test!(complete_explicit_params_sync, {
        let client = get_client();
        let args = create_args();
        client.complete_prompt_sync(args)?
    });

    fn stop_condition_args() -> api::CompletionArgs {
        let mut args = api::CompletionArgs::builder();
        args.prompt(
            r#"
Q: Please type `#` now
A:"#,
        )
        // turn temp & top_p way down to prevent test flakiness
        .temperature(0.0)
        .top_p(0.0)
        .max_tokens(100)
        .stop(vec!["#".into(), "\n".into()])
        .build()
        .expect("Bug: build should succeed")
    }

    fn assert_completion_finish_reason(completion: Completion) {
        assert_eq!(
            completion.choices[0].finish_reason,
            api::FinishReason::StopSequenceReached
        );
    }

    async_test!(complete_stop_condition_async, {
        let client = get_client();
        let args = stop_condition_args();
        assert_completion_finish_reason(client.complete_prompt(args).await?);
    });

    sync_test!(complete_stop_condition_sync, {
        let client = get_client();
        let args = stop_condition_args();
        assert_completion_finish_reason(client.complete_prompt_sync(args)?);
    });
}
