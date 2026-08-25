//! Versioned JSON-lines protocol for the thin local engine owner.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::backend::{BackendKind, PromotionState};
use crate::engine::{
    AllocationSnapshot, EngineLifecycle, EngineMetrics, ExecutorCapabilities, LoadProgress,
    SessionStatus,
};
use crate::sampler::SamplerConfig;

pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireRequest {
    pub protocol_version: u32,
    pub request_id: u64,
    #[serde(flatten)]
    pub request: Request,
}

impl WireRequest {
    pub fn validate_version(&self) -> std::result::Result<(), ProtocolVersionError> {
        if self.protocol_version == PROTOCOL_VERSION {
            Ok(())
        } else {
            Err(ProtocolVersionError {
                received: self.protocol_version,
                supported: PROTOCOL_VERSION,
            })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolVersionError {
    pub received: u32,
    pub supported: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", content = "params", rename_all = "snake_case")]
pub enum Request {
    Protocol,
    Health,
    Models,
    Capabilities,
    Load(LoadRequest),
    Warmup,
    Prefill(PrefillRequest),
    Decode(DecodeRequest),
    Cancel(CancelRequest),
    ResetSession(SessionControl),
    Unload,
    ResponsesCreate(CreateResponse),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadRequest {
    pub release_id: String,
    pub pack_id: String,
    pub memory_profile_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillRequest {
    pub operation_id: u64,
    pub session_id: u64,
    pub token_ids: Vec<u32>,
    pub mtp_enabled: bool,
    pub sampling: SamplingParameters,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DecodeRequest {
    pub operation_id: u64,
    pub session_id: u64,
    pub token_id: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CancelRequest {
    pub operation_id: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SessionControl {
    pub session_id: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplingParameters {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub seed: u64,
}

impl From<SamplingParameters> for SamplerConfig {
    fn from(value: SamplingParameters) -> Self {
        Self {
            temperature: value.temperature,
            top_k: value.top_k as usize,
            top_p: value.top_p,
            seed: value.seed,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateResponse {
    pub model: String,
    pub input: Value,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireResponse {
    pub protocol_version: u32,
    pub request_id: u64,
    #[serde(flatten)]
    pub response: Response,
}

impl WireResponse {
    pub fn new(request_id: u64, response: Response) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id,
            response,
        }
    }

    pub fn error(request_id: u64, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(
            request_id,
            Response::Error {
                code: code.into(),
                message: message.into(),
                retryable: false,
            },
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Protocol {
        minimum_version: u32,
        maximum_version: u32,
    },
    Health {
        status: String,
        model: String,
        release_id: Option<String>,
        pack_id: Option<String>,
        memory_profile_id: Option<String>,
        lifecycle: EngineLifecycle,
        backend: BackendKind,
        hardware_profile: String,
        promotion_state: PromotionState,
        session: Option<SessionStatus>,
        allocations: AllocationSnapshot,
    },
    Models {
        data: Vec<ModelRecord>,
    },
    Capabilities {
        capabilities: ExecutorCapabilities,
    },
    Ack {
        operation: String,
        lifecycle: EngineLifecycle,
    },
    Progress {
        operation_id: u64,
        stage: LoadProgress,
        completed_bytes: Option<u64>,
        total_bytes: Option<u64>,
    },
    Token {
        operation_id: u64,
        session_id: u64,
        sequence_index: u64,
        token_id: u32,
        text: String,
        accepted_via_mtp: bool,
    },
    Completed {
        operation_id: u64,
        session_id: u64,
        context_tokens: u64,
        finish_reason: FinishReason,
        metrics: EngineMetrics,
    },
    Error {
        code: String,
        message: String,
        retryable: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub id: String,
    pub object: String,
    pub ready: bool,
    pub release_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_request_round_trips_with_explicit_protocol_version() {
        let request = WireRequest {
            protocol_version: PROTOCOL_VERSION,
            request_id: 9,
            request: Request::Prefill(PrefillRequest {
                operation_id: 11,
                session_id: 7,
                token_ids: vec![1, 2, 3],
                mtp_enabled: true,
                sampling: SamplingParameters {
                    temperature: 0.0,
                    top_k: 1,
                    top_p: 1.0,
                    seed: 42,
                },
            }),
        };
        let encoded = serde_json::to_vec(&request).unwrap();
        let decoded: WireRequest = serde_json::from_slice(&encoded).unwrap();
        decoded.validate_version().unwrap();
        assert_eq!(decoded.request_id, 9);
        let Request::Prefill(prefill) = decoded.request else {
            panic!("expected prefill request");
        };
        let sampler = SamplerConfig::from(prefill.sampling);
        assert_eq!(sampler.seed, 42);
        assert_eq!(sampler.top_k, 1);
    }

    #[test]
    fn incompatible_protocol_version_is_rejected_before_dispatch() {
        let request = WireRequest {
            protocol_version: PROTOCOL_VERSION + 1,
            request_id: 1,
            request: Request::Health,
        };
        assert_eq!(
            request.validate_version(),
            Err(ProtocolVersionError {
                received: PROTOCOL_VERSION + 1,
                supported: PROTOCOL_VERSION,
            })
        );
    }

    #[test]
    fn streaming_events_retain_one_request_and_operation_identity() {
        let response = WireResponse::new(
            17,
            Response::Token {
                operation_id: 21,
                session_id: 4,
                sequence_index: 3,
                token_id: 99,
                text: "test".into(),
                accepted_via_mtp: true,
            },
        );
        let encoded = serde_json::to_vec(&response).unwrap();
        let decoded: WireResponse = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.request_id, 17);
        assert!(matches!(
            decoded.response,
            Response::Token {
                operation_id: 21,
                ..
            }
        ));
    }
}
