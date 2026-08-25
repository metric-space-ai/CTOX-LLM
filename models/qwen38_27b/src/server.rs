use crate::backend::{Backend, BackendKind, ExecutionPolicy, PromotionState};
use crate::config::MODEL_ID;
use crate::engine::{AllocationSnapshot, EngineLifecycle};
use crate::loader::{ChecksumPolicy, ModelArtifact};
use crate::wire::{ModelRecord, Request, Response, WireRequest, WireResponse, PROTOCOL_VERSION};
use crate::{EngineError, Result};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

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

    fn respond(&self, envelope: WireRequest) -> WireResponse {
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

#[cfg(unix)]
pub fn run_unix(socket: impl AsRef<Path>, state: &ServerState) -> Result<()> {
    use std::os::unix::net::{UnixListener, UnixStream};

    fn handle(mut stream: UnixStream, state: &ServerState) -> Result<()> {
        let reader_stream = stream.try_clone()?;
        let reader = BufReader::new(reader_stream);
        for line in reader.lines() {
            let line = line?;
            let response = match serde_json::from_str::<WireRequest>(&line) {
                Ok(request) => state.respond(request),
                Err(error) => WireResponse::error(0, "invalid_request", error.to_string()),
            };
            serde_json::to_writer(&mut stream, &response)?;
            stream.write_all(b"\n")?;
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
    for connection in listener.incoming() {
        handle(connection?, state)?;
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn run_unix(_socket: impl AsRef<Path>, _state: &ServerState) -> Result<()> {
    Err(EngineError::UnsupportedOperation {
        backend: "transport",
        operation: "unix socket server",
        reason: "Windows named-pipe owner has not been implemented".into(),
    })
}
