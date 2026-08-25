//! Android JNI control surface. Heavy operations are never implemented here;
//! they are dispatched to the QNN HTP and Vulkan backend.

use jni::objects::{JClass, JString};
use jni::sys::{jboolean, jstring, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;

use crate::loader::{ChecksumPolicy, ModelArtifact};
use crate::memory::{FoldMemoryPlan, FOLD_WEIGHT_LIMIT_BYTES};
use crate::Qwen38Config;

#[no_mangle]
pub extern "system" fn Java_ai_metricspace_ctoxllm_Qwen38Native_validateArtifact(
    mut env: JNIEnv,
    _class: JClass,
    path: JString,
) -> jboolean {
    let path: String = match env.get_string(&path) {
        Ok(value) => value.into(),
        Err(_) => return JNI_FALSE,
    };
    match ModelArtifact::open(path, ChecksumPolicy::ManifestOnly) {
        Ok(_) => JNI_TRUE,
        Err(_) => JNI_FALSE,
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_metricspace_ctoxllm_Qwen38Native_foldMemoryPlan(
    env: JNIEnv,
    _class: JClass,
    context_tokens: i64,
) -> jstring {
    let result = u64::try_from(context_tokens)
        .ok()
        .and_then(|context| {
            FoldMemoryPlan::for_context(&Qwen38Config::default(), context, FOLD_WEIGHT_LIMIT_BYTES)
                .ok()
        })
        .and_then(|plan| {
            plan.verify()
                .ok()
                .and_then(|_| serde_json::to_string(&plan).ok())
        })
        .unwrap_or_else(|| "{\"error\":\"invalid_memory_plan\"}".into());
    env.new_string(result)
        .map(|value| value.into_raw())
        .unwrap_or(std::ptr::null_mut())
}
