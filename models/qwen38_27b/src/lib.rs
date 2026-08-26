//! Qwen3.8-27B model-specific inference engine.
//!
//! General inference frameworks are deliberately absent. The scalar path is a
//! correctness oracle; production dispatch must select a verified hardware
//! backend or return an error.

pub mod backend;
pub mod config;
pub mod decoder;
pub mod engine;
pub mod error;
pub mod fanout;
pub mod format;
pub mod graph;
pub mod kv_cache;
pub mod loader;
pub mod memory;
pub mod quant;
pub mod reference;
pub mod release;
pub mod roofline;
pub mod sampler;
pub mod server;
pub mod tensor_contract;
pub mod wire;

#[cfg(feature = "android-jni")]
pub mod android;

pub use config::Qwen38Config;
pub use error::{EngineError, Result};
