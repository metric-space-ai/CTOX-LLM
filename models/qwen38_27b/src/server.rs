use crate::backend::{Backend, BackendKind, ExecutionPolicy, PromotionState};
use crate::config::MODEL_ID;
use crate::engine::{
    AllocationSnapshot, CancellationToken, Engine, EngineLifecycle, GeneratedStep, ModelExecutor,
    SessionOptions,
};
use crate::loader::{ChecksumPolicy, ModelArtifact};
use crate::wire::{
    FinishReason, ModelRecord, Request, Response, WireRequest, WireResponse, PROTOCOL_VERSION,
};
use crate::{EngineError, Result};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::Mutex;

pub trait WireService {
    fn respond(&self, envelope: WireRequest) -> Vec<WireResponse>;
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
    cancellations: Mutex<HashMap<u64, CancellationToken>>,
    sequence_index: Mutex<u64>,
}

impl<E: ModelExecutor + Send> EngineServer<E> {
    pub fn new(engine: Engine<E>) -> Self {
        Self {
            engine: Mutex::new(engine),
            cancellations: Mutex::new(HashMap::new()),
            sequence_index: Mutex::new(0),
        }
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
}

#[cfg(unix)]
pub fn run_unix<S: WireService + Sync>(socket: impl AsRef<Path>, state: &S) -> Result<()> {
    use std::os::unix::net::{UnixListener, UnixStream};

    fn handle<S: WireService>(mut stream: UnixStream, state: &S) -> Result<()> {
        let reader_stream = stream.try_clone()?;
        let reader = BufReader::new(reader_stream);
        for line in reader.lines() {
            let line = line?;
            let responses = match serde_json::from_str::<WireRequest>(&line) {
                Ok(request) => state.respond(request),
                Err(error) => vec![WireResponse::error(0, "invalid_request", error.to_string())],
            };
            for response in responses {
                serde_json::to_writer(&mut stream, &response)?;
                stream.write_all(b"\n")?;
            }
            stream.flush()?;
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
}
