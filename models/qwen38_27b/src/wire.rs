use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(tag = "method", content = "params", rename_all = "snake_case")]
pub enum Request {
    Health,
    Models,
    ResponsesCreate(CreateResponse),
}

#[derive(Debug, Deserialize)]
pub struct CreateResponse {
    pub model: String,
    pub input: Value,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response<'a> {
    Health {
        status: &'a str,
        model: &'a str,
        backend: &'a str,
        promotion_state: &'a str,
    },
    Models {
        data: Vec<ModelRecord<'a>>,
    },
    Error {
        code: &'a str,
        message: String,
    },
}

#[derive(Debug, Serialize)]
pub struct ModelRecord<'a> {
    pub id: &'a str,
    pub object: &'a str,
    pub ready: bool,
}
