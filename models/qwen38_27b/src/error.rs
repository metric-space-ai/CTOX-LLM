use thiserror::Error;

pub type Result<T> = std::result::Result<T, EngineError>;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid model artifact: {0}")]
    InvalidArtifact(String),
    #[error("unsupported quantization dtype: {0}")]
    UnsupportedDType(String),
    #[error("backend {backend} cannot execute {operation}: {reason}")]
    UnsupportedOperation {
        backend: &'static str,
        operation: &'static str,
        reason: String,
    },
    #[error("memory budget exceeded: {0}")]
    MemoryBudget(String),
    #[error("invalid tensor shape: {0}")]
    Shape(String),
    #[error("invalid engine state: {0}")]
    InvalidState(String),
    #[error("inference operation cancelled")]
    Cancelled,
}
