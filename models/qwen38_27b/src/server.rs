use crate::backend::{Backend, ExecutionPolicy};
use crate::config::MODEL_ID;
use crate::loader::{ChecksumPolicy, ModelArtifact};
use crate::wire::{ModelRecord, Request, Response};
use crate::{EngineError, Result};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub struct ServerState {
    _artifact: ModelArtifact,
    backend_profile: &'static str,
    promotion_state: &'static str,
}

impl ServerState {
    pub fn load(artifact: impl AsRef<Path>) -> Result<Self> {
        let artifact = ModelArtifact::open(artifact, ChecksumPolicy::ManifestOnly)?;
        let backend = crate::backend::cpu::CpuBackend::detect(ExecutionPolicy::Production)?;
        let promotion_state = match backend.promotion_state() {
            crate::backend::PromotionState::Unavailable => "unavailable",
            crate::backend::PromotionState::Contract => "contract",
            crate::backend::PromotionState::Verifier => "verifier",
            crate::backend::PromotionState::Experimental => "experimental",
            crate::backend::PromotionState::Optimized => "optimized",
        };
        Ok(Self {
            _artifact: artifact,
            backend_profile: backend.profile(),
            promotion_state,
        })
    }

    fn respond(&self, request: Request) -> Response<'_> {
        match request {
            Request::Health => Response::Health {
                status: "bringup",
                model: MODEL_ID,
                backend: self.backend_profile,
                promotion_state: self.promotion_state,
            },
            Request::Models => Response::Models {
                data: vec![ModelRecord {
                    id: MODEL_ID,
                    object: "model",
                    ready: false,
                }],
            },
            Request::ResponsesCreate(request) => Response::Error {
                code: "engine_not_ready",
                message: format!(
                    "{} request accepted by the transport contract, but the full decoder graph is not promoted (input={}, max_output_tokens={:?}, seed={:?})",
                    request.model,
                    request.input,
                    request.max_output_tokens,
                    request.seed
                ),
            },
        }
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
            let response = match serde_json::from_str::<Request>(&line) {
                Ok(request) => state.respond(request),
                Err(error) => Response::Error {
                    code: "invalid_request",
                    message: error.to_string(),
                },
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
