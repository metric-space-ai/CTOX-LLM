use crate::backend::{Backend, BackendKind, ExecutionPolicy, PromotionState};
use crate::config::MODEL_ID;
use crate::engine::{
    AllocationSnapshot, CancellationToken, Engine, EngineLifecycle, GeneratedStep, LoadProgress,
    ModelExecutor, SessionOptions,
};
use crate::loader::{ChecksumPolicy, ModelArtifact};
use crate::release::ReleaseManifest;
use crate::sampler::SamplerConfig;
use crate::tokenizer::{
    ChatMessage, ChatRole, ChatTemplateOptions, IncrementalDecoder, Qwen38Tokenizer, ToolCall,
    END_OF_TEXT_ID, IM_END_ID,
};
use crate::wire::{
    FinishReason, ModelRecord, Request, Response, WireRequest, WireResponse, PROTOCOL_VERSION,
};
use crate::{EngineError, Result};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::Mutex;

fn responses_messages(input: &serde_json::Value) -> Result<Vec<ChatMessage>> {
    if let Some(text) = input.as_str() {
        return Ok(vec![ChatMessage::text(ChatRole::User, text)]);
    }
    let values = input.as_array().ok_or_else(|| {
        EngineError::InvalidArtifact(
            "Responses input must be a string or an ordered message array".into(),
        )
    })?;
    if values.is_empty() {
        return Err(EngineError::InvalidArtifact(
            "Responses message array is empty".into(),
        ));
    }
    values.iter().map(responses_message).collect()
}

fn responses_message(value: &serde_json::Value) -> Result<ChatMessage> {
    if let Some(text) = value.as_str() {
        return Ok(ChatMessage::text(ChatRole::User, text));
    }
    let object = value
        .as_object()
        .ok_or_else(|| EngineError::InvalidArtifact("Responses message is not an object".into()))?;
    let role = match object.get("role").and_then(serde_json::Value::as_str) {
        Some("system" | "developer") => ChatRole::System,
        Some("user") => ChatRole::User,
        Some("assistant") => ChatRole::Assistant,
        Some("tool") => ChatRole::Tool,
        Some(role) => {
            return Err(EngineError::InvalidArtifact(format!(
                "unsupported Responses message role {role}"
            )))
        }
        None => {
            return Err(EngineError::InvalidArtifact(
                "Responses message has no role".into(),
            ))
        }
    };
    let content = responses_content(object.get("content"))?;
    let reasoning_content = object
        .get("reasoning_content")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned);
    let tool_calls = object
        .get("tool_calls")
        .map(responses_tool_calls)
        .transpose()?
        .unwrap_or_default();
    Ok(ChatMessage {
        role,
        content: Some(content),
        reasoning_content,
        tool_calls,
    })
}

fn responses_content(value: Option<&serde_json::Value>) -> Result<String> {
    let Some(value) = value else {
        return Ok(String::new());
    };
    if value.is_null() {
        return Ok(String::new());
    }
    if let Some(text) = value.as_str() {
        return Ok(text.into());
    }
    let parts = value.as_array().ok_or_else(|| {
        EngineError::InvalidArtifact("Responses message content is not text or an array".into())
    })?;
    let mut output = String::new();
    for part in parts {
        let object = part.as_object().ok_or_else(|| {
            EngineError::InvalidArtifact("Responses content part is not an object".into())
        })?;
        let kind = object
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("input_text");
        if !matches!(kind, "input_text" | "output_text" | "text") {
            return Err(EngineError::UnsupportedOperation {
                backend: "responses_frontend",
                operation: "multimodal content",
                reason: format!("content part {kind} belongs to the separate vision package"),
            });
        }
        let text = object
            .get("text")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                EngineError::InvalidArtifact("Responses text part has no text".into())
            })?;
        output.push_str(text);
    }
    Ok(output)
}

fn responses_tool_calls(value: &serde_json::Value) -> Result<Vec<ToolCall>> {
    let calls = value.as_array().ok_or_else(|| {
        EngineError::InvalidArtifact("Responses tool_calls is not an array".into())
    })?;
    calls
        .iter()
        .map(|call| {
            let object = call.as_object().ok_or_else(|| {
                EngineError::InvalidArtifact("Responses tool call is not an object".into())
            })?;
            let function = object
                .get("function")
                .and_then(serde_json::Value::as_object)
                .unwrap_or(object);
            let name = function
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    EngineError::InvalidArtifact("Responses tool call has no function name".into())
                })?;
            let arguments = match function.get("arguments") {
                None | Some(serde_json::Value::Null) => serde_json::Map::new(),
                Some(serde_json::Value::Object(arguments)) => arguments.clone(),
                Some(serde_json::Value::String(arguments)) => serde_json::from_str(arguments)
                    .map_err(|error| {
                        EngineError::InvalidArtifact(format!(
                            "Responses tool-call arguments are invalid JSON: {error}"
                        ))
                    })?,
                Some(_) => {
                    return Err(EngineError::InvalidArtifact(
                        "Responses tool-call arguments are not an object or JSON string".into(),
                    ))
                }
            };
            Ok(ToolCall {
                name: name.into(),
                arguments,
            })
        })
        .collect()
}

pub trait WireService {
    fn respond(&self, envelope: WireRequest) -> Vec<WireResponse>;

    fn respond_stream(
        &self,
        envelope: WireRequest,
        emit: &mut dyn FnMut(WireResponse) -> Result<()>,
    ) -> Result<()> {
        for response in self.respond(envelope) {
            emit(response)?;
        }
        Ok(())
    }
}

pub struct ServerState {
    _artifact: ModelArtifact,
    backend_profile: &'static str,
    backend_kind: BackendKind,
    promotion_state: PromotionState,
}

impl ServerState {
    pub fn load(artifact: impl AsRef<Path>) -> Result<Self> {
        let artifact = ModelArtifact::open(artifact, ChecksumPolicy::ManifestOnly)?;
        let backend = crate::backend::cpu::CpuBackend::detect(ExecutionPolicy::Production)?;
        Ok(Self {
            _artifact: artifact,
            backend_profile: backend.profile(),
            backend_kind: backend.kind(),
            promotion_state: backend.promotion_state(),
        })
    }

    fn respond_one(&self, envelope: WireRequest) -> WireResponse {
        let request_id = envelope.request_id;
        if let Err(version) = envelope.validate_version() {
            return WireResponse::error(
                request_id,
                "protocol_version_mismatch",
                format!(
                    "protocol version {} is unsupported; this server requires {}",
                    version.received, version.supported
                ),
            );
        }
        let response = match envelope.request {
            Request::Protocol => Response::Protocol {
                minimum_version: PROTOCOL_VERSION,
                maximum_version: PROTOCOL_VERSION,
            },
            Request::Health => Response::Health {
                status: "bringup".into(),
                model: MODEL_ID.into(),
                release_id: None,
                pack_id: None,
                memory_profile_id: None,
                lifecycle: EngineLifecycle::Loaded,
                backend: self.backend_kind,
                hardware_profile: self.backend_profile.into(),
                promotion_state: self.promotion_state,
                session: None,
                allocations: AllocationSnapshot::default(),
            },
            Request::Models => Response::Models {
                data: vec![ModelRecord {
                    id: MODEL_ID.into(),
                    object: "model".into(),
                    ready: false,
                    release_id: None,
                }],
            },
            Request::ResponsesCreate(request) => Response::Error {
                code: "engine_not_ready".into(),
                message: format!("{} cannot generate yet: the full decoder executor is not promoted (input={}, max_output_tokens={:?}, seed={:?})", request.model, request.input, request.max_output_tokens, request.seed),
                retryable: false,
            },
            Request::Capabilities
            | Request::Load(_)
            | Request::Warmup
            | Request::Prefill(_)
            | Request::Decode(_)
            | Request::Cancel(_)
            | Request::ResetSession(_)
            | Request::Unload => Response::Error {
                code: "engine_not_ready".into(),
                message: "the IPC contract is active, but no complete decoder executor is promoted"
                    .into(),
                retryable: false,
            },
        };
        WireResponse::new(request_id, response)
    }
}

impl WireService for ServerState {
    fn respond(&self, envelope: WireRequest) -> Vec<WireResponse> {
        vec![self.respond_one(envelope)]
    }
}

pub struct EngineServer<E: ModelExecutor + Send> {
    engine: Mutex<Engine<E>>,
    tokenizer: Mutex<Qwen38Tokenizer>,
    cancellations: Mutex<HashMap<u64, CancellationToken>>,
    sequence_index: Mutex<u64>,
}

impl<E: ModelExecutor + Send> EngineServer<E> {
    pub fn new(engine: Engine<E>, tokenizer: Qwen38Tokenizer) -> Self {
        Self {
            engine: Mutex::new(engine),
            tokenizer: Mutex::new(tokenizer),
            cancellations: Mutex::new(HashMap::new()),
            sequence_index: Mutex::new(0),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_signed(
        release_root: impl AsRef<Path>,
        release: &ReleaseManifest,
        pack_id: &str,
        memory_profile_id: &str,
        expected_key_id: &str,
        trusted_public_key: &[u8; 32],
        policy: ExecutionPolicy,
        executor: E,
        mut progress: impl FnMut(LoadProgress),
    ) -> Result<Self> {
        let release_root = release_root.as_ref();
        release.verify_signature(expected_key_id, trusted_public_key)?;
        progress(LoadProgress::SignatureVerified);
        let tokenizer = release.load_tokenizer(release_root)?;
        progress(LoadProgress::TokenizerVerified);
        let engine = Engine::load_preverified_release(
            release_root,
            release,
            pack_id,
            memory_profile_id,
            policy,
            executor,
            progress,
        )?;
        Ok(Self::new(engine, tokenizer))
    }

    fn engine_error(request_id: u64, error: impl std::fmt::Display) -> Vec<WireResponse> {
        vec![WireResponse::error(
            request_id,
            "engine_error",
            error.to_string(),
        )]
    }

    fn begin_operation(
        &self,
        request_id: u64,
        operation_id: u64,
    ) -> std::result::Result<CancellationToken, Vec<WireResponse>> {
        let mut operations = self.cancellations.lock().map_err(|_| {
            Self::engine_error(request_id, "cancellation registry lock is poisoned")
        })?;
        if operations.contains_key(&operation_id) {
            return Err(vec![WireResponse::error(
                request_id,
                "operation_conflict",
                format!("operation {operation_id} is already active"),
            )]);
        }
        let cancellation = CancellationToken::default();
        operations.insert(operation_id, cancellation.clone());
        Ok(cancellation)
    }

    fn finish_operation(&self, operation_id: u64) {
        if let Ok(mut operations) = self.cancellations.lock() {
            operations.remove(&operation_id);
        }
    }

    fn next_sequence_indices(&self, count: usize) -> Result<Vec<u64>> {
        let mut next = self
            .sequence_index
            .lock()
            .map_err(|_| EngineError::InvalidState("sequence-index lock is poisoned".into()))?;
        let start = *next;
        *next = next
            .checked_add(count as u64)
            .ok_or_else(|| EngineError::InvalidState("sequence index overflows".into()))?;
        Ok((start..*next).collect())
    }

    fn reset_sequence_index(&self) -> Result<()> {
        *self
            .sequence_index
            .lock()
            .map_err(|_| EngineError::InvalidState("sequence-index lock is poisoned".into()))? = 0;
        Ok(())
    }

    fn static_response(&self, request_id: u64, request: Request) -> Option<Vec<WireResponse>> {
        let response = match request {
            Request::Protocol => Response::Protocol {
                minimum_version: PROTOCOL_VERSION,
                maximum_version: PROTOCOL_VERSION,
            },
            Request::Health => {
                let engine = match self.engine.lock() {
                    Ok(engine) => engine,
                    Err(_) => {
                        return Some(Self::engine_error(request_id, "engine lock is poisoned"))
                    }
                };
                let health = engine.health();
                Response::Health {
                    status: match health.lifecycle {
                        EngineLifecycle::Warm | EngineLifecycle::Active => "ready",
                        EngineLifecycle::Loaded => "loaded",
                        EngineLifecycle::UnloadFailed => "unload_failed",
                        EngineLifecycle::Unloaded => "unloaded",
                    }
                    .into(),
                    model: MODEL_ID.into(),
                    release_id: Some(health.release_id),
                    pack_id: Some(health.pack_id),
                    memory_profile_id: Some(health.memory_profile_id),
                    lifecycle: health.lifecycle,
                    backend: health.backend,
                    hardware_profile: health.hardware_profile,
                    promotion_state: health.promotion_state,
                    session: health.session,
                    allocations: health.allocations,
                }
            }
            Request::Models => {
                let engine = match self.engine.lock() {
                    Ok(engine) => engine,
                    Err(_) => {
                        return Some(Self::engine_error(request_id, "engine lock is poisoned"))
                    }
                };
                let health = engine.health();
                Response::Models {
                    data: vec![ModelRecord {
                        id: MODEL_ID.into(),
                        object: "model".into(),
                        ready: matches!(
                            health.lifecycle,
                            EngineLifecycle::Warm | EngineLifecycle::Active
                        ),
                        release_id: Some(health.release_id),
                    }],
                }
            }
            Request::Capabilities => {
                let engine = match self.engine.lock() {
                    Ok(engine) => engine,
                    Err(_) => {
                        return Some(Self::engine_error(request_id, "engine lock is poisoned"))
                    }
                };
                Response::Capabilities {
                    capabilities: engine.capabilities().clone(),
                }
            }
            other => return self.control_response(request_id, other),
        };
        Some(vec![WireResponse::new(request_id, response)])
    }

    fn control_response(&self, request_id: u64, request: Request) -> Option<Vec<WireResponse>> {
        match request {
            Request::Load(load) => {
                let engine = match self.engine.lock() {
                    Ok(engine) => engine,
                    Err(_) => {
                        return Some(Self::engine_error(request_id, "engine lock is poisoned"))
                    }
                };
                let health = engine.health();
                if load.release_id != health.release_id
                    || load.pack_id != health.pack_id
                    || load.memory_profile_id != health.memory_profile_id
                {
                    return Some(vec![WireResponse::error(
                        request_id,
                        "release_identity_mismatch",
                        "requested release, pack, or memory profile differs from the loaded engine",
                    )]);
                }
                Some(vec![WireResponse::new(
                    request_id,
                    Response::Ack {
                        operation: "load".into(),
                        lifecycle: health.lifecycle,
                    },
                )])
            }
            Request::Warmup => {
                let mut engine = match self.engine.lock() {
                    Ok(engine) => engine,
                    Err(_) => {
                        return Some(Self::engine_error(request_id, "engine lock is poisoned"))
                    }
                };
                Some(match engine.warmup() {
                    Ok(()) => vec![WireResponse::new(
                        request_id,
                        Response::Ack {
                            operation: "warmup".into(),
                            lifecycle: engine.health().lifecycle,
                        },
                    )],
                    Err(error) => Self::engine_error(request_id, error),
                })
            }
            Request::Cancel(cancel) => {
                let operation = match self.cancellations.lock() {
                    Ok(operations) => operations.get(&cancel.operation_id).cloned(),
                    Err(_) => {
                        return Some(Self::engine_error(
                            request_id,
                            "cancellation registry lock is poisoned",
                        ))
                    }
                };
                Some(if let Some(operation) = operation {
                    operation.cancel();
                    // Never wait for the engine while holding the cancellation
                    // registry: the executing thread removes itself after it
                    // releases the engine lock.
                    let lifecycle = self
                        .engine
                        .try_lock()
                        .map(|engine| engine.health().lifecycle)
                        .unwrap_or(EngineLifecycle::Active);
                    vec![WireResponse::new(
                        request_id,
                        Response::Ack {
                            operation: "cancel".into(),
                            lifecycle,
                        },
                    )]
                } else {
                    vec![WireResponse::error(
                        request_id,
                        "operation_not_found",
                        format!("operation {} is not active", cancel.operation_id),
                    )]
                })
            }
            Request::ResetSession(control) => {
                let mut engine = match self.engine.lock() {
                    Ok(engine) => engine,
                    Err(_) => {
                        return Some(Self::engine_error(request_id, "engine lock is poisoned"))
                    }
                };
                let health = engine.health();
                if health
                    .session
                    .is_some_and(|session| session.id != control.session_id)
                {
                    return Some(vec![WireResponse::error(
                        request_id,
                        "session_identity_mismatch",
                        "reset session id differs from the active session",
                    )]);
                }
                Some(match engine.reset_session() {
                    Ok(()) => vec![WireResponse::new(
                        request_id,
                        Response::Ack {
                            operation: "reset_session".into(),
                            lifecycle: engine.health().lifecycle,
                        },
                    )],
                    Err(error) => Self::engine_error(request_id, error),
                })
            }
            Request::Unload => {
                let mut engine = match self.engine.lock() {
                    Ok(engine) => engine,
                    Err(_) => {
                        return Some(Self::engine_error(request_id, "engine lock is poisoned"))
                    }
                };
                Some(match engine.unload() {
                    Ok(()) => vec![WireResponse::new(
                        request_id,
                        Response::Ack {
                            operation: "unload".into(),
                            lifecycle: engine.health().lifecycle,
                        },
                    )],
                    Err(error) => Self::engine_error(request_id, error),
                })
            }
            Request::ResponsesCreate(_) => Some(vec![WireResponse::error(
                request_id,
                "responses_frontend_not_ready",
                "the token-ID engine is active, but Responses input rendering is not bound",
            )]),
            Request::Prefill(_) | Request::Decode(_) => None,
            Request::Protocol | Request::Health | Request::Models | Request::Capabilities => {
                unreachable!("static requests are handled before control dispatch")
            }
        }
    }

    fn prefill_response(
        &self,
        request_id: u64,
        request: crate::wire::PrefillRequest,
    ) -> Vec<WireResponse> {
        let cancellation = match self.begin_operation(request_id, request.operation_id) {
            Ok(cancellation) => cancellation,
            Err(response) => return response,
        };
        let (result, metrics) = match self.engine.lock() {
            Ok(mut engine) => {
                let result = engine.prefill(
                    SessionOptions {
                        id: request.session_id,
                        mtp_enabled: request.mtp_enabled,
                        sampling: request.sampling.into(),
                    },
                    &request.token_ids,
                    &cancellation,
                );
                (result, engine.metrics().clone())
            }
            Err(_) => (
                Err(EngineError::InvalidState(
                    "engine lock is poisoned during prefill".into(),
                )),
                Default::default(),
            ),
        };
        self.finish_operation(request.operation_id);
        match result {
            Ok(step) => {
                if let Err(error) = self.reset_sequence_index() {
                    return Self::engine_error(request_id, error);
                }
                let sequence = match self.next_sequence_indices(1) {
                    Ok(sequence) => sequence[0],
                    Err(error) => return Self::engine_error(request_id, error),
                };
                vec![WireResponse::new(
                    request_id,
                    Response::Token {
                        operation_id: request.operation_id,
                        session_id: request.session_id,
                        sequence_index: sequence,
                        token_id: step.token_id,
                        text: String::new(),
                        accepted_via_mtp: false,
                    },
                )]
            }
            Err(EngineError::Cancelled) => vec![WireResponse::new(
                request_id,
                Response::Completed {
                    operation_id: request.operation_id,
                    session_id: request.session_id,
                    context_tokens: 0,
                    finish_reason: FinishReason::Cancelled,
                    metrics,
                },
            )],
            Err(error) => Self::engine_error(request_id, error),
        }
    }

    fn decode_response(
        &self,
        request_id: u64,
        request: crate::wire::DecodeRequest,
    ) -> Vec<WireResponse> {
        let cancellation = match self.begin_operation(request_id, request.operation_id) {
            Ok(cancellation) => cancellation,
            Err(response) => return response,
        };
        let (result, metrics, previous_context_tokens) = match self.engine.lock() {
            Ok(mut engine) => {
                let previous_context_tokens = engine
                    .health()
                    .session
                    .map(|session| session.context_tokens)
                    .unwrap_or(0);
                let result = engine.decode(request.session_id, request.token_id, &cancellation);
                (result, engine.metrics().clone(), previous_context_tokens)
            }
            Err(_) => (
                Err(EngineError::InvalidState(
                    "engine lock is poisoned during decode".into(),
                )),
                Default::default(),
                0,
            ),
        };
        self.finish_operation(request.operation_id);
        match result {
            Ok(step) => {
                let count = step.accepted_draft_tokens.len() + 1;
                let indices = match self.next_sequence_indices(count) {
                    Ok(indices) => indices,
                    Err(error) => return Self::engine_error(request_id, error),
                };
                generated_step_responses(
                    request_id,
                    request.operation_id,
                    request.session_id,
                    step,
                    &indices,
                )
            }
            Err(EngineError::Cancelled) => vec![WireResponse::new(
                request_id,
                Response::Completed {
                    operation_id: request.operation_id,
                    session_id: request.session_id,
                    context_tokens: previous_context_tokens,
                    finish_reason: FinishReason::Cancelled,
                    metrics,
                },
            )],
            Err(error) => Self::engine_error(request_id, error),
        }
    }

    fn responses_create_stream(
        &self,
        request_id: u64,
        request: crate::wire::CreateResponse,
        emit: &mut dyn FnMut(WireResponse) -> Result<()>,
    ) -> Result<()> {
        if request.model != MODEL_ID {
            return emit(WireResponse::error(
                request_id,
                "model_identity_mismatch",
                format!("requested model {} differs from {MODEL_ID}", request.model),
            ));
        }
        let max_output_tokens = request.max_output_tokens.unwrap_or(256);
        if max_output_tokens == 0 {
            return emit(WireResponse::error(
                request_id,
                "invalid_request",
                "max_output_tokens must be positive",
            ));
        }
        let messages = match responses_messages(&request.input) {
            Ok(messages) => messages,
            Err(error) => {
                return emit(WireResponse::error(
                    request_id,
                    "invalid_input",
                    error.to_string(),
                ))
            }
        };
        let options = ChatTemplateOptions {
            tools: request.tools,
            add_generation_prompt: true,
            enable_thinking: request.enable_thinking.unwrap_or(true),
            reasoning_effort: request.reasoning_effort.unwrap_or_default(),
            preserve_thinking: true,
        };
        let prompt_tokens = match self.tokenizer.lock() {
            Ok(tokenizer) => match tokenizer.render_and_encode(&messages, &options) {
                Ok((_, tokens)) => tokens,
                Err(error) => {
                    return emit(WireResponse::error(
                        request_id,
                        "invalid_input",
                        error.to_string(),
                    ))
                }
            },
            Err(_) => {
                return emit(WireResponse::error(
                    request_id,
                    "engine_error",
                    "tokenizer lock is poisoned",
                ))
            }
        };
        let cancellation = match self.begin_operation(request_id, request_id) {
            Ok(cancellation) => cancellation,
            Err(responses) => {
                for response in responses {
                    emit(response)?;
                }
                return Ok(());
            }
        };
        let result = self.run_response_generation(
            request_id,
            &prompt_tokens,
            max_output_tokens,
            request.seed.unwrap_or(0),
            request.mtp_enabled,
            &cancellation,
            emit,
        );
        self.finish_operation(request_id);
        match result {
            Ok(completion) => emit(WireResponse::new(
                request_id,
                Response::Completed {
                    operation_id: request_id,
                    session_id: request_id,
                    context_tokens: completion.context_tokens,
                    finish_reason: completion.finish_reason,
                    metrics: completion.metrics,
                },
            )),
            Err(EngineError::Cancelled) => {
                let metrics = self
                    .engine
                    .lock()
                    .map(|engine| engine.metrics().clone())
                    .unwrap_or_default();
                emit(WireResponse::new(
                    request_id,
                    Response::Completed {
                        operation_id: request_id,
                        session_id: request_id,
                        context_tokens: 0,
                        finish_reason: FinishReason::Cancelled,
                        metrics,
                    },
                ))
            }
            Err(error) => emit(WireResponse::error(
                request_id,
                "engine_error",
                error.to_string(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_response_generation(
        &self,
        request_id: u64,
        prompt_tokens: &[u32],
        max_output_tokens: u32,
        seed: u64,
        mtp_enabled: bool,
        cancellation: &CancellationToken,
        emit: &mut dyn FnMut(WireResponse) -> Result<()>,
    ) -> Result<GenerationCompletion> {
        let mut engine = self
            .engine
            .lock()
            .map_err(|_| EngineError::InvalidState("engine lock is poisoned".into()))?;
        let generation = (|| -> Result<(FinishReason, u64)> {
            let mut step = engine.prefill(
                SessionOptions {
                    id: request_id,
                    mtp_enabled,
                    sampling: SamplerConfig {
                        temperature: 0.0,
                        top_k: 1,
                        top_p: 1.0,
                        seed,
                    },
                },
                prompt_tokens,
                cancellation,
            )?;
            let mut incremental = IncrementalDecoder::default();
            let mut pending = Vec::<(u32, bool)>::new();
            let mut sequence_index = 0_u64;
            let mut generated = 0_u32;
            let finish_reason = 'generation: loop {
                let next_input = step.token_id;
                let mut output = step
                    .accepted_draft_tokens
                    .into_iter()
                    .map(|token| (token, true))
                    .chain(std::iter::once((step.token_id, false)));
                for (token_id, accepted_via_mtp) in &mut output {
                    if generated == max_output_tokens {
                        break 'generation FinishReason::Length;
                    }
                    if cancellation.is_cancelled() {
                        return Err(EngineError::Cancelled);
                    }
                    generated += 1;
                    if token_id == END_OF_TEXT_ID || token_id == IM_END_ID {
                        break 'generation FinishReason::Stop;
                    }
                    pending.push((token_id, accepted_via_mtp));
                    let delta = {
                        let tokenizer = self.tokenizer.lock().map_err(|_| {
                            EngineError::InvalidState("tokenizer lock is poisoned".into())
                        })?;
                        incremental.push(&tokenizer, token_id)?
                    };
                    if let Some(delta) = delta {
                        emit_pending_tokens(
                            request_id,
                            &mut sequence_index,
                            &mut pending,
                            delta,
                            emit,
                        )?;
                    }
                }
                if generated == max_output_tokens {
                    break FinishReason::Length;
                }
                step = engine.decode(request_id, next_input, cancellation)?;
            };
            if !pending.is_empty() {
                let delta = {
                    let tokenizer = self.tokenizer.lock().map_err(|_| {
                        EngineError::InvalidState("tokenizer lock is poisoned".into())
                    })?;
                    incremental.finish(&tokenizer)?
                };
                emit_pending_tokens(request_id, &mut sequence_index, &mut pending, delta, emit)?;
            }
            let context_tokens = engine
                .health()
                .session
                .map(|session| session.context_tokens)
                .unwrap_or(0);
            Ok((finish_reason, context_tokens))
        })();
        let metrics = engine.metrics().clone();
        let reset = engine.reset_session();
        match (generation, reset) {
            (Ok((finish_reason, context_tokens)), Ok(())) => Ok(GenerationCompletion {
                context_tokens,
                finish_reason,
                metrics,
            }),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }
}

fn generated_step_responses(
    request_id: u64,
    operation_id: u64,
    session_id: u64,
    step: GeneratedStep,
    indices: &[u64],
) -> Vec<WireResponse> {
    debug_assert_eq!(indices.len(), step.accepted_draft_tokens.len() + 1);
    let mut responses = Vec::with_capacity(indices.len());
    for (index, token_id) in step.accepted_draft_tokens.into_iter().enumerate() {
        responses.push(WireResponse::new(
            request_id,
            Response::Token {
                operation_id,
                session_id,
                sequence_index: indices[index],
                token_id,
                text: String::new(),
                accepted_via_mtp: true,
            },
        ));
    }
    responses.push(WireResponse::new(
        request_id,
        Response::Token {
            operation_id,
            session_id,
            sequence_index: *indices.last().expect("one target token is emitted"),
            token_id: step.token_id,
            text: String::new(),
            accepted_via_mtp: false,
        },
    ));
    responses
}

fn emit_pending_tokens(
    request_id: u64,
    sequence_index: &mut u64,
    pending: &mut Vec<(u32, bool)>,
    mut text_delta: String,
    emit: &mut dyn FnMut(WireResponse) -> Result<()>,
) -> Result<()> {
    let count = pending.len();
    for (index, (token_id, accepted_via_mtp)) in pending.drain(..).enumerate() {
        let text = if index + 1 == count {
            std::mem::take(&mut text_delta)
        } else {
            String::new()
        };
        emit(WireResponse::new(
            request_id,
            Response::Token {
                operation_id: request_id,
                session_id: request_id,
                sequence_index: *sequence_index,
                token_id,
                text,
                accepted_via_mtp,
            },
        ))?;
        *sequence_index = sequence_index
            .checked_add(1)
            .ok_or_else(|| EngineError::InvalidState("sequence index overflows".into()))?;
    }
    Ok(())
}

#[derive(Debug)]
struct GenerationCompletion {
    context_tokens: u64,
    finish_reason: FinishReason,
    metrics: crate::engine::EngineMetrics,
}

impl<E: ModelExecutor + Send> WireService for EngineServer<E> {
    fn respond(&self, envelope: WireRequest) -> Vec<WireResponse> {
        let request_id = envelope.request_id;
        if let Err(version) = envelope.validate_version() {
            return vec![WireResponse::error(
                request_id,
                "protocol_version_mismatch",
                format!(
                    "protocol version {} is unsupported; this server requires {}",
                    version.received, version.supported
                ),
            )];
        }
        match envelope.request {
            Request::Prefill(request) => self.prefill_response(request_id, request),
            Request::Decode(request) => self.decode_response(request_id, request),
            request => self
                .static_response(request_id, request)
                .expect("non-inference request must produce a response"),
        }
    }

    fn respond_stream(
        &self,
        envelope: WireRequest,
        emit: &mut dyn FnMut(WireResponse) -> Result<()>,
    ) -> Result<()> {
        let request_id = envelope.request_id;
        if let Err(version) = envelope.validate_version() {
            return emit(WireResponse::error(
                request_id,
                "protocol_version_mismatch",
                format!(
                    "protocol version {} is unsupported; this server requires {}",
                    version.received, version.supported
                ),
            ));
        }
        match envelope.request {
            Request::ResponsesCreate(request) => {
                self.responses_create_stream(request_id, request, emit)
            }
            request => {
                for response in self.respond(WireRequest {
                    protocol_version: PROTOCOL_VERSION,
                    request_id,
                    request,
                }) {
                    emit(response)?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(unix)]
pub fn run_unix<S: WireService + Sync>(socket: impl AsRef<Path>, state: &S) -> Result<()> {
    use std::os::unix::net::{UnixListener, UnixStream};

    fn handle<S: WireService>(mut stream: UnixStream, state: &S) -> Result<()> {
        let reader_stream = stream.try_clone()?;
        let reader = BufReader::new(reader_stream);
        for line in reader.lines() {
            let line = line?;
            let mut emit = |response: WireResponse| -> Result<()> {
                serde_json::to_writer(&mut stream, &response)?;
                stream.write_all(b"\n")?;
                stream.flush()?;
                Ok(())
            };
            match serde_json::from_str::<WireRequest>(&line) {
                Ok(request) => state.respond_stream(request, &mut emit)?,
                Err(error) => emit(WireResponse::error(0, "invalid_request", error.to_string()))?,
            };
        }
        Ok(())
    }

    let socket = socket.as_ref();
    if socket.exists() {
        return Err(EngineError::InvalidArtifact(format!(
            "refusing to replace existing socket {}",
            socket.display()
        )));
    }
    let listener = UnixListener::bind(socket)?;
    std::thread::scope(|scope| -> Result<()> {
        for connection in listener.incoming() {
            let stream = connection?;
            scope.spawn(move || {
                let _ = handle(stream, state);
            });
        }
        Ok(())
    })
}

#[cfg(not(unix))]
pub fn run_unix<S: WireService + Sync>(_socket: impl AsRef<Path>, _state: &S) -> Result<()> {
    Err(EngineError::UnsupportedOperation {
        backend: "transport",
        operation: "unix socket server",
        reason: "Windows named-pipe owner has not been implemented".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn mtp_stream_emits_verified_prefix_before_target_bonus() {
        let responses = generated_step_responses(
            11,
            17,
            23,
            GeneratedStep {
                token_id: 99,
                draft_tokens_proposed: 3,
                draft_tokens_verified: 3,
                accepted_draft_tokens: vec![41, 42],
            },
            &[7, 8, 9],
        );
        let observed = responses
            .into_iter()
            .map(|response| match response.response {
                Response::Token {
                    operation_id,
                    session_id,
                    sequence_index,
                    token_id,
                    accepted_via_mtp,
                    ..
                } => (
                    operation_id,
                    session_id,
                    sequence_index,
                    token_id,
                    accepted_via_mtp,
                ),
                _ => panic!("expected token response"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            observed,
            vec![
                (17, 23, 7, 41, true),
                (17, 23, 8, 42, true),
                (17, 23, 9, 99, false),
            ]
        );
    }

    #[test]
    fn responses_input_preserves_multilingual_history_and_tool_arguments() {
        let messages = responses_messages(&json!([
            {"role": "system", "content": "Antworte präzise."},
            {"role": "user", "content": [
                {"type": "input_text", "text": "東京の天気?"}
            ]},
            {"role": "assistant", "content": "", "tool_calls": [{
                "function": {"name": "weather", "arguments": "{\"city\":\"東京\"}"}
            }]},
            {"role": "tool", "content": "晴れ"}
        ]))
        .unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, ChatRole::System);
        assert_eq!(messages[1].content.as_deref(), Some("東京の天気?"));
        assert_eq!(messages[2].tool_calls[0].name, "weather");
        assert_eq!(messages[2].tool_calls[0].arguments["city"], "東京");
        assert_eq!(messages[3].role, ChatRole::Tool);
    }

    #[test]
    fn responses_text_frontend_rejects_vision_parts() {
        let error = responses_messages(&json!([{
            "role": "user",
            "content": [{"type": "input_image", "image_url": "x"}]
        }]))
        .unwrap_err();
        assert!(matches!(error, EngineError::UnsupportedOperation { .. }));
    }

    #[test]
    fn cpu_engine_server_satisfies_concurrent_transport_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EngineServer<crate::decoder::CpuCorrectnessExecutor>>();
    }
}
