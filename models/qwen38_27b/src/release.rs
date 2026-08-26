//! Signed release and backend-pack contract for one canonical logical model.

use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path};

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::backend::BackendKind;
use crate::config::{Qwen38Config, MODEL_ID};
use crate::format::RecoveryMode;
use crate::loader::ModelArtifact;
use crate::memory::{LinearStateDType, SpeculativeStateStrategy};
use crate::tokenizer::{
    Qwen38Tokenizer, CHAT_TEMPLATE_SHA256, END_OF_TEXT_ID, IM_END_ID, IM_START_ID, THINK_END_ID,
    THINK_START_ID, TOKENIZER_SHA256,
};
use crate::{EngineError, Result};

pub const RELEASE_FORMAT: &str = "ctox.model-release.v2";
pub const ED25519_ALGORITHM: &str = "ed25519-sha256";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseManifest {
    pub format: String,
    pub release_id: String,
    pub model: CanonicalModelIdentity,
    pub tokenizer: TokenizerContract,
    pub packages: Vec<ReleasePackage>,
    pub memory_profiles: Vec<MemoryProfile>,
    pub integrity: ManifestIntegrity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalModelIdentity {
    pub model_id: String,
    pub architecture: String,
    pub bf16_repository: String,
    pub bf16_revision: String,
    pub bf16_root_sha256: String,
    pub logical_checkpoint_sha256: String,
    pub logical_tensor_root_sha256: String,
    pub recovery_sha256: String,
    pub fixed_logical_qcodes: bool,
    pub allowed_large_matrix_types: Vec<LogicalQuantType>,
    pub mtp: MtpIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalQuantType {
    Q2B64,
    Q4B64,
    MixedQ2Q4B64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MtpIdentity {
    pub resident_with_text: bool,
    pub layers: u32,
    pub logical_sha256: String,
    pub every_draft_token_verified: bool,
    pub draft_vocabulary: MtpDraftVocabularyIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MtpDraftVocabularyIdentity {
    /// Canonical little-endian u32 token IDs in strictly increasing order.
    pub token_ids: ReleaseFile,
    pub source_teacher_cache_set_sha256: String,
    pub token_count: u32,
    pub observed_output_tokens: u64,
    pub overall_coverage_ppm: u32,
    pub code_coverage_ppm: u32,
    pub minimum_domain_coverage_ppm: u32,
    pub minimum_language_coverage_ppm: u32,
    /// Restricted rows may propose tokens only; target verification remains
    /// full-vocabulary and therefore preserves greedy semantics exactly.
    pub target_verified_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerContract {
    pub tokenizer_root_sha256: String,
    pub chat_template_sha256: String,
    pub reasoning_format: String,
    pub tool_call_format: String,
    pub special_tokens: Vec<SpecialToken>,
    pub files: Vec<ReleaseFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecialToken {
    pub name: String,
    pub id: u32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseFile {
    pub relative_path: String,
    pub bytes: u64,
    pub sha256: String,
    pub chunks: Vec<FileChunk>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileChunk {
    pub index: u32,
    pub offset: u64,
    pub length: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageKind {
    TextMtp,
    Vision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleasePackage {
    pub package_id: String,
    pub kind: PackageKind,
    pub default_download: bool,
    pub packs: Vec<BackendPack>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendPack {
    pub pack_id: String,
    pub backend: BackendKind,
    pub hardware_profile: String,
    pub artifact: ReleaseFile,
    pub artifact_target: String,
    pub artifact_manifest_sha256: String,
    pub logical_checkpoint_sha256: String,
    pub logical_tensor_root_sha256: String,
    pub deterministic_repack: bool,
    pub requantized: bool,
    pub contains_mtp: bool,
    pub loader: LoaderContract,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoaderContract {
    pub strategy: LoaderStrategy,
    pub maximum_resident_full_model_copies: u8,
    pub retains_full_cpu_copy: bool,
    pub retains_full_device_copy: bool,
    pub supports_resumable_download: bool,
    pub supports_atomic_activation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoaderStrategy {
    DirectMmap,
    WindowedUpload,
    SharedBuffer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryProfile {
    pub profile_id: String,
    pub pack_id: String,
    pub context_tokens: u64,
    pub sessions: u32,
    pub resident_model_bytes: u64,
    pub persistent_backend_graph_bytes: u64,
    pub persistent_runtime_bytes: u64,
    pub linear_state_dtype: LinearStateDType,
    pub linear_state_bytes_per_session: u64,
    pub mtp_draft_tokens: u32,
    pub speculative_state_strategy: SpeculativeStateStrategy,
    pub speculative_linear_state_bytes_per_session: u64,
    pub kv: KvMemoryFormula,
    pub mtp_kv: KvMemoryFormula,
    pub prefill_scratch_peak_bytes: u64,
    pub decode_scratch_peak_bytes: u64,
    pub loader_transient_peak_bytes: u64,
    pub accelerator_unattributed_reserve_bytes: u64,
    pub hard_limit_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvMemoryFormula {
    pub fixed_bytes_per_session: u64,
    pub bytes_per_token_per_session: u64,
    pub retained_q4_tokens_per_session: u64,
    pub q4_delta_bytes_per_token: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestIntegrity {
    pub unsigned_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<ManifestSignature>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestSignature {
    pub algorithm: String,
    pub key_id: String,
    pub signature_hex: String,
}

#[derive(Serialize)]
struct UnsignedManifest<'a> {
    format: &'a str,
    release_id: &'a str,
    model: &'a CanonicalModelIdentity,
    tokenizer: &'a TokenizerContract,
    packages: &'a [ReleasePackage],
    memory_profiles: &'a [MemoryProfile],
}

impl ReleaseFile {
    fn validate(&self) -> Result<()> {
        let path = Path::new(&self.relative_path);
        if self.relative_path.is_empty()
            || path.is_absolute()
            || path
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return invalid(format!("unsafe release path {}", self.relative_path));
        }
        if self.bytes == 0 || !valid_sha256(&self.sha256) || self.chunks.is_empty() {
            return invalid(format!(
                "release file {} has invalid size, digest, or chunks",
                self.relative_path
            ));
        }
        let mut next_offset = 0_u64;
        for (expected_index, chunk) in self.chunks.iter().enumerate() {
            if chunk.index as usize != expected_index
                || chunk.offset != next_offset
                || chunk.length == 0
                || !valid_sha256(&chunk.sha256)
            {
                return invalid(format!(
                    "release file {} has invalid chunk {}",
                    self.relative_path, expected_index
                ));
            }
            next_offset = next_offset
                .checked_add(chunk.length)
                .ok_or_else(|| EngineError::InvalidArtifact("chunk range overflows".into()))?;
        }
        if next_offset != self.bytes {
            return invalid(format!(
                "release file {} chunks cover {}, expected {}",
                self.relative_path, next_offset, self.bytes
            ));
        }
        Ok(())
    }

    pub fn read_verified(&self, release_root: &Path) -> Result<Vec<u8>> {
        self.validate()?;
        let canonical_root = release_root.canonicalize()?;
        let path = canonical_root.join(&self.relative_path);
        let canonical_path = path.canonicalize()?;
        if !canonical_path.starts_with(&canonical_root) {
            return invalid(format!(
                "release file escapes installation root: {}",
                self.relative_path
            ));
        }
        let encoded = fs::read(&canonical_path)?;
        if encoded.len() as u64 != self.bytes
            || format!("{:x}", Sha256::digest(&encoded)) != self.sha256
        {
            return invalid(format!(
                "release file size or SHA-256 differs: {}",
                self.relative_path
            ));
        }
        for chunk in &self.chunks {
            let start = usize::try_from(chunk.offset)
                .map_err(|_| EngineError::InvalidArtifact("chunk offset exceeds usize".into()))?;
            let length = usize::try_from(chunk.length)
                .map_err(|_| EngineError::InvalidArtifact("chunk length exceeds usize".into()))?;
            let end = start
                .checked_add(length)
                .ok_or_else(|| EngineError::InvalidArtifact("chunk range overflows".into()))?;
            if format!("{:x}", Sha256::digest(&encoded[start..end])) != chunk.sha256 {
                return invalid(format!(
                    "release chunk {} differs: {}",
                    chunk.index, self.relative_path
                ));
            }
        }
        Ok(encoded)
    }
}

impl KvMemoryFormula {
    pub fn bytes(&self, context_tokens: u64, sessions: u32) -> Result<u64> {
        let per_session = self
            .bytes_per_token_per_session
            .checked_mul(context_tokens)
            .and_then(|value| {
                self.q4_delta_bytes_per_token
                    .checked_mul(self.retained_q4_tokens_per_session.min(context_tokens))
                    .and_then(|q4| value.checked_add(q4))
            })
            .and_then(|value| value.checked_add(self.fixed_bytes_per_session))
            .ok_or_else(|| EngineError::MemoryBudget("KV formula overflows".into()))?;
        per_session
            .checked_mul(sessions as u64)
            .ok_or_else(|| EngineError::MemoryBudget("KV session total overflows".into()))
    }
}

impl MemoryProfile {
    fn checked_sum(values: &[u64], label: &str) -> Result<u64> {
        values.iter().try_fold(0_u64, |sum, value| {
            sum.checked_add(*value)
                .ok_or_else(|| EngineError::MemoryBudget(format!("{label} overflows")))
        })
    }

    pub fn steady_bytes(&self) -> Result<u64> {
        let session_state = self
            .linear_state_bytes_per_session
            .checked_add(self.speculative_linear_state_bytes_per_session)
            .ok_or_else(|| EngineError::MemoryBudget("combined linear state overflows".into()))?
            .checked_mul(self.sessions as u64)
            .ok_or_else(|| EngineError::MemoryBudget("linear state overflows".into()))?;
        Self::checked_sum(
            &[
                self.resident_model_bytes,
                self.persistent_backend_graph_bytes,
                self.persistent_runtime_bytes,
                session_state,
                self.kv.bytes(self.context_tokens, self.sessions)?,
                self.mtp_kv.bytes(self.context_tokens, self.sessions)?,
                self.accelerator_unattributed_reserve_bytes,
            ],
            "steady memory",
        )
    }

    pub fn prefill_peak_bytes(&self) -> Result<u64> {
        Self::checked_sum(
            &[self.steady_bytes()?, self.prefill_scratch_peak_bytes],
            "prefill peak",
        )
    }

    pub fn decode_peak_bytes(&self) -> Result<u64> {
        Self::checked_sum(
            &[self.steady_bytes()?, self.decode_scratch_peak_bytes],
            "decode peak",
        )
    }

    pub fn load_peak_bytes(&self) -> Result<u64> {
        Self::checked_sum(
            &[
                self.resident_model_bytes,
                self.persistent_backend_graph_bytes,
                self.persistent_runtime_bytes,
                self.loader_transient_peak_bytes,
                self.accelerator_unattributed_reserve_bytes,
            ],
            "load peak",
        )
    }

    pub fn maximum_peak_bytes(&self) -> Result<u64> {
        Ok(self
            .load_peak_bytes()?
            .max(self.prefill_peak_bytes()?)
            .max(self.decode_peak_bytes()?))
    }
}

impl ReleaseManifest {
    pub fn load_tokenizer(&self, release_root: &Path) -> Result<Qwen38Tokenizer> {
        self.validate()?;
        let tokenizer_file = self.tokenizer_file("tokenizer.json")?;
        let chat_template_file = self.tokenizer_file("chat_template.jinja")?;
        let tokenizer_bytes = tokenizer_file.read_verified(release_root)?;
        let chat_template_bytes = chat_template_file.read_verified(release_root)?;
        Qwen38Tokenizer::from_release_bytes(
            tokenizer_bytes,
            chat_template_bytes,
            &tokenizer_file.sha256,
            &self.tokenizer.chat_template_sha256,
        )
    }

    fn tokenizer_file(&self, name: &str) -> Result<&ReleaseFile> {
        let mut matches = self.tokenizer.files.iter().filter(|file| {
            Path::new(&file.relative_path)
                .file_name()
                .is_some_and(|file_name| file_name == name)
        });
        let file = matches.next().ok_or_else(|| {
            EngineError::InvalidArtifact(format!("tokenizer release has no {name}"))
        })?;
        if matches.next().is_some() {
            return invalid(format!("tokenizer release has multiple {name} files"));
        }
        Ok(file)
    }

    pub fn load_mtp_draft_token_ids(&self, release_root: &Path) -> Result<Vec<u32>> {
        self.validate()?;
        let identity = &self.model.mtp.draft_vocabulary;
        let encoded = identity.token_ids.read_verified(release_root)?;
        let token_ids = encoded
            .chunks_exact(std::mem::size_of::<u32>())
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("u32 chunk")))
            .collect::<Vec<_>>();
        if token_ids.len() != identity.token_count as usize
            || token_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || token_ids
                .iter()
                .any(|token| *token as usize >= Qwen38Config::default().vocab_size)
        {
            return invalid("MTP draft token file is not canonical for this model");
        }
        Ok(token_ids)
    }

    pub fn backend_pack(&self, pack_id: &str) -> Result<&BackendPack> {
        self.packages
            .iter()
            .flat_map(|package| &package.packs)
            .find(|pack| pack.pack_id == pack_id)
            .ok_or_else(|| {
                EngineError::InvalidArtifact(format!(
                    "backend pack {pack_id} is not in release {}",
                    self.release_id
                ))
            })
    }

    pub fn memory_profile(&self, profile_id: &str) -> Result<&MemoryProfile> {
        self.memory_profiles
            .iter()
            .find(|profile| profile.profile_id == profile_id)
            .ok_or_else(|| {
                EngineError::InvalidArtifact(format!(
                    "memory profile {profile_id} is not in release {}",
                    self.release_id
                ))
            })
    }

    fn unsigned_bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(&UnsignedManifest {
            format: &self.format,
            release_id: &self.release_id,
            model: &self.model,
            tokenizer: &self.tokenizer,
            packages: &self.packages,
            memory_profiles: &self.memory_profiles,
        })?)
    }

    pub fn computed_unsigned_sha256(&self) -> Result<String> {
        Ok(format!("{:x}", Sha256::digest(self.unsigned_bytes()?)))
    }

    pub fn seal_unsigned(&mut self) -> Result<()> {
        self.integrity.unsigned_sha256 = self.computed_unsigned_sha256()?;
        self.integrity.signature = None;
        Ok(())
    }

    pub fn verify_signature(&self, expected_key_id: &str, public_key: &[u8; 32]) -> Result<()> {
        self.validate()?;
        let envelope =
            self.integrity.signature.as_ref().ok_or_else(|| {
                EngineError::InvalidArtifact("release manifest is not signed".into())
            })?;
        if envelope.algorithm != ED25519_ALGORITHM || envelope.key_id != expected_key_id {
            return invalid("release signature algorithm or key id does not match trust policy");
        }
        let signature_bytes = decode_hex::<64>(&envelope.signature_hex, "release signature")?;
        let signature = Signature::from_bytes(&signature_bytes);
        let key = VerifyingKey::from_bytes(public_key)
            .map_err(|error| EngineError::InvalidArtifact(error.to_string()))?;
        let digest = decode_hex::<32>(&self.integrity.unsigned_sha256, "unsigned digest")?;
        key.verify_strict(&digest, &signature).map_err(|_| {
            EngineError::InvalidArtifact("release signature verification failed".into())
        })
    }

    pub fn verify_backend_pack_equivalence(&self, first: &str, second: &str) -> Result<()> {
        self.validate()?;
        let first = self.backend_pack(first)?;
        let second = self.backend_pack(second)?;
        if first.logical_checkpoint_sha256 != second.logical_checkpoint_sha256
            || first.logical_tensor_root_sha256 != second.logical_tensor_root_sha256
        {
            return invalid("backend packs do not represent the same logical checkpoint");
        }
        Ok(())
    }

    pub fn admit_artifact(&self, pack_id: &str, artifact: &ModelArtifact) -> Result<()> {
        self.validate()?;
        let pack = self.backend_pack(pack_id)?;
        let manifest = artifact.manifest();
        if artifact.file_bytes() != pack.artifact.bytes
            || artifact.manifest_sha256() != pack.artifact_manifest_sha256
            || manifest.model != self.model.model_id
            || manifest.revision != self.model.bf16_revision
            || manifest.target != pack.artifact_target
        {
            return invalid("loaded artifact does not match its signed backend-pack identity");
        }
        let recovery = manifest.recovery.as_ref().ok_or_else(|| {
            EngineError::InvalidArtifact("release artifact has no recovery provenance".into())
        })?;
        if recovery.mode != RecoveryMode::Trained
            || recovery.artifact_sha256.as_deref() != Some(&self.model.recovery_sha256)
            || !recovery.fixed_logical_qcodes
        {
            return invalid("release artifact recovery does not match canonical model identity");
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.format != RELEASE_FORMAT || self.release_id.is_empty() {
            return invalid("release format or release id is invalid");
        }
        if self.model.model_id != MODEL_ID
            || self.model.architecture.is_empty()
            || self.model.bf16_repository.is_empty()
            || !valid_lower_hex(&self.model.bf16_revision, 40)
            || !valid_sha256(&self.model.bf16_root_sha256)
            || !valid_sha256(&self.model.logical_checkpoint_sha256)
            || !valid_sha256(&self.model.logical_tensor_root_sha256)
            || !valid_sha256(&self.model.recovery_sha256)
            || !self.model.fixed_logical_qcodes
        {
            return invalid("canonical model identity is incomplete");
        }
        let expected_quant_types = [
            LogicalQuantType::Q2B64,
            LogicalQuantType::Q4B64,
            LogicalQuantType::MixedQ2Q4B64,
        ];
        if self.model.allowed_large_matrix_types != expected_quant_types
            || !self.model.mtp.resident_with_text
            || self.model.mtp.layers == 0
            || !valid_sha256(&self.model.mtp.logical_sha256)
            || !self.model.mtp.every_draft_token_verified
        {
            return invalid("quantization or resident MTP identity is invalid");
        }
        let draft_vocabulary = &self.model.mtp.draft_vocabulary;
        let expected_draft_bytes = u64::from(draft_vocabulary.token_count)
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or_else(|| {
                EngineError::InvalidArtifact("MTP draft vocabulary size overflows".into())
            })?;
        draft_vocabulary.token_ids.validate()?;
        if draft_vocabulary.token_count == 0
            || draft_vocabulary.token_count as usize >= Qwen38Config::default().vocab_size
            || draft_vocabulary.token_ids.bytes != expected_draft_bytes
            || !valid_sha256(&draft_vocabulary.source_teacher_cache_set_sha256)
            || draft_vocabulary.observed_output_tokens == 0
            || ![
                draft_vocabulary.overall_coverage_ppm,
                draft_vocabulary.code_coverage_ppm,
                draft_vocabulary.minimum_domain_coverage_ppm,
                draft_vocabulary.minimum_language_coverage_ppm,
            ]
            .into_iter()
            .all(|coverage| coverage > 0 && coverage <= 1_000_000)
            || !draft_vocabulary.target_verified_only
        {
            return invalid("MTP draft vocabulary identity is invalid");
        }
        if !valid_sha256(&self.tokenizer.tokenizer_root_sha256)
            || self.tokenizer.chat_template_sha256 != CHAT_TEMPLATE_SHA256
            || self.tokenizer.reasoning_format.is_empty()
            || self.tokenizer.tool_call_format.is_empty()
            || self.tokenizer.special_tokens.is_empty()
            || self.tokenizer.files.is_empty()
        {
            return invalid("tokenizer/template contract is incomplete");
        }
        let mut token_names = HashSet::new();
        let mut token_ids = HashSet::new();
        for token in &self.tokenizer.special_tokens {
            if token.name.is_empty()
                || token.text.is_empty()
                || !token_names.insert(&token.name)
                || !token_ids.insert(token.id)
            {
                return invalid("special token names and ids must be unique and non-empty");
            }
        }
        for (name, id, text) in [
            ("end_of_text", END_OF_TEXT_ID, "<|endoftext|>"),
            ("im_start", IM_START_ID, "<|im_start|>"),
            ("im_end", IM_END_ID, "<|im_end|>"),
            ("think_start", THINK_START_ID, "<think>"),
            ("think_end", THINK_END_ID, "</think>"),
        ] {
            if !self
                .tokenizer
                .special_tokens
                .iter()
                .any(|token| token.name == name && token.id == id && token.text == text)
            {
                return invalid(format!("required Qwen special token {name} differs"));
            }
        }
        let mut file_paths = HashSet::new();
        file_paths.insert(&draft_vocabulary.token_ids.relative_path);
        for file in &self.tokenizer.files {
            file.validate()?;
            if !file_paths.insert(&file.relative_path) {
                return invalid("release file paths must be globally unique");
            }
        }
        let tokenizer_file = self.tokenizer_file("tokenizer.json")?;
        let chat_template_file = self.tokenizer_file("chat_template.jinja")?;
        if tokenizer_file.sha256 != TOKENIZER_SHA256
            || chat_template_file.sha256 != CHAT_TEMPLATE_SHA256
            || chat_template_file.sha256 != self.tokenizer.chat_template_sha256
        {
            return invalid("tokenizer or chat-template file differs from the pinned model");
        }

        let mut package_ids = HashSet::new();
        let mut pack_ids = HashSet::new();
        let mut pack_keys = HashSet::new();
        let mut text_packages = 0;
        let mut default_packages = 0;
        let mut text_tensor_root: Option<&str> = None;
        for package in &self.packages {
            if package.package_id.is_empty()
                || package.packs.is_empty()
                || !package_ids.insert(&package.package_id)
            {
                return invalid("release package ids must be unique and non-empty");
            }
            if package.kind == PackageKind::TextMtp {
                text_packages += 1;
            }
            if package.default_download {
                default_packages += 1;
                if package.kind != PackageKind::TextMtp {
                    return invalid("only the text+MTP package may be a default download");
                }
            }
            for pack in &package.packs {
                pack.artifact.validate()?;
                if pack.pack_id.is_empty()
                    || pack.hardware_profile.is_empty()
                    || pack.artifact_target.is_empty()
                    || !pack_ids.insert(&pack.pack_id)
                    || !pack_keys.insert((package.kind, pack.backend, &pack.hardware_profile))
                    || !file_paths.insert(&pack.artifact.relative_path)
                    || !valid_sha256(&pack.artifact_manifest_sha256)
                    || pack.logical_checkpoint_sha256 != self.model.logical_checkpoint_sha256
                    || pack.logical_tensor_root_sha256 != self.model.logical_tensor_root_sha256
                    || !pack.deterministic_repack
                    || pack.requantized
                {
                    return invalid(
                        "backend pack identity or deterministic-repack contract failed",
                    );
                }
                if package.kind == PackageKind::TextMtp {
                    if !pack.contains_mtp {
                        return invalid("every text backend pack must contain resident MTP");
                    }
                    if text_tensor_root
                        .replace(&pack.logical_tensor_root_sha256)
                        .is_some_and(|root| root != pack.logical_tensor_root_sha256)
                    {
                        return invalid("text backend packs have different logical tensor roots");
                    }
                } else if pack.contains_mtp {
                    return invalid("vision package must not duplicate MTP");
                }
                if pack.loader.maximum_resident_full_model_copies != 1
                    || pack.loader.retains_full_cpu_copy
                    || !pack.loader.supports_resumable_download
                    || !pack.loader.supports_atomic_activation
                {
                    return invalid(
                        "loader permits duplicate model residency or lacks install safety",
                    );
                }
            }
        }
        if text_packages != 1 || default_packages != 1 {
            return invalid("release needs exactly one default text+MTP package");
        }

        let mut profile_ids = HashSet::new();
        let mut profiled_packs = HashSet::new();
        for profile in &self.memory_profiles {
            if profile.profile_id.is_empty()
                || !profile_ids.insert(&profile.profile_id)
                || !pack_ids.contains(&profile.pack_id)
                || profile.context_tokens == 0
                || profile.context_tokens > Qwen38Config::default().max_position_embeddings as u64
                || profile.sessions == 0
                || profile.resident_model_bytes == 0
                || profile.hard_limit_bytes == 0
            {
                return invalid("memory profile identity or dimensions are invalid");
            }
            let expected_speculative_state = match profile.speculative_state_strategy {
                SpeculativeStateStrategy::Disabled => 0,
                SpeculativeStateStrategy::ReplayOnReject => profile.linear_state_bytes_per_session,
                SpeculativeStateStrategy::AlignedPages => profile
                    .linear_state_bytes_per_session
                    .checked_mul(profile.mtp_draft_tokens as u64)
                    .ok_or_else(|| {
                        EngineError::MemoryBudget("profile speculative state overflows".into())
                    })?,
            };
            let mtp_kv_bytes = profile.mtp_kv.bytes(profile.context_tokens, 1)?;
            if (profile.mtp_draft_tokens == 0)
                != (profile.speculative_state_strategy == SpeculativeStateStrategy::Disabled)
                || profile.speculative_linear_state_bytes_per_session != expected_speculative_state
                || (profile.mtp_draft_tokens == 0) != (mtp_kv_bytes == 0)
            {
                return invalid("memory profile MTP/state strategy is inconsistent");
            }
            if profile.maximum_peak_bytes()? > profile.hard_limit_bytes {
                return Err(EngineError::MemoryBudget(format!(
                    "memory profile {} exceeds its hard limit",
                    profile.profile_id
                )));
            }
            profiled_packs.insert(&profile.pack_id);
        }
        if profiled_packs.len() != pack_ids.len() {
            return invalid("every backend pack requires at least one admitted memory profile");
        }

        if self.computed_unsigned_sha256()? != self.integrity.unsigned_sha256 {
            return invalid("release manifest unsigned SHA-256 mismatch");
        }
        if let Some(signature) = &self.integrity.signature {
            if signature.algorithm != ED25519_ALGORITHM
                || signature.key_id.is_empty()
                || decode_hex::<64>(&signature.signature_hex, "release signature").is_err()
            {
                return invalid("release signature envelope is invalid");
            }
        }
        Ok(())
    }
}

fn valid_sha256(value: &str) -> bool {
    valid_lower_hex(value, 64)
}

fn valid_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn decode_hex<const N: usize>(value: &str, label: &str) -> Result<[u8; N]> {
    if value.len() != N * 2 {
        return invalid(format!("{label} has the wrong encoded length"));
    }
    let mut decoded = [0_u8; N];
    for (index, output) in decoded.iter_mut().enumerate() {
        *output = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| EngineError::InvalidArtifact(format!("{label} is not lowercase hex")))?;
    }
    if value.bytes().any(|byte| (b'A'..=b'F').contains(&byte)) {
        return invalid(format!("{label} is not lowercase hex"));
    }
    Ok(decoded)
}

fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(EngineError::InvalidArtifact(message.into()))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use ed25519_dalek::{Signer, SigningKey};

    use super::*;
    use crate::format::{
        align_up, FileHeader, ModelManifest, RecoveryProvenance, TensorDType, TensorEntry,
        DEFAULT_ALIGNMENT, HEADER_BYTES,
    };
    use crate::loader::ChecksumPolicy;
    use crate::quant::{Q2Block64, BLOCK_LEN};

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn file(path: &str, bytes: u64, character: char) -> ReleaseFile {
        ReleaseFile {
            relative_path: path.into(),
            bytes,
            sha256: digest(character),
            chunks: vec![FileChunk {
                index: 0,
                offset: 0,
                length: bytes,
                sha256: digest(character),
            }],
        }
    }

    fn pinned_file(path: &str, bytes: u64, sha256: &str) -> ReleaseFile {
        ReleaseFile {
            relative_path: path.into(),
            bytes,
            sha256: sha256.into(),
            chunks: vec![FileChunk {
                index: 0,
                offset: 0,
                length: bytes,
                sha256: sha256.into(),
            }],
        }
    }

    fn manifest() -> ReleaseManifest {
        let pack = BackendPack {
            pack_id: "text-cuda-sm86".into(),
            backend: BackendKind::Cuda,
            hardware_profile: "sm86".into(),
            artifact: file("packs/text-cuda-sm86.ctoxq", 8_000_000_000, 'a'),
            artifact_target: "cuda-sm86".into(),
            artifact_manifest_sha256: digest('b'),
            logical_checkpoint_sha256: digest('c'),
            logical_tensor_root_sha256: digest('d'),
            deterministic_repack: true,
            requantized: false,
            contains_mtp: true,
            loader: LoaderContract {
                strategy: LoaderStrategy::WindowedUpload,
                maximum_resident_full_model_copies: 1,
                retains_full_cpu_copy: false,
                retains_full_device_copy: true,
                supports_resumable_download: true,
                supports_atomic_activation: true,
            },
        };
        let mut manifest = ReleaseManifest {
            format: RELEASE_FORMAT.into(),
            release_id: "qwen38-27b-test.1".into(),
            model: CanonicalModelIdentity {
                model_id: MODEL_ID.into(),
                architecture: "Qwen3_5ForCausalLM".into(),
                bf16_repository: MODEL_ID.into(),
                bf16_revision: "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0".into(),
                bf16_root_sha256: digest('e'),
                logical_checkpoint_sha256: digest('c'),
                logical_tensor_root_sha256: digest('d'),
                recovery_sha256: digest('f'),
                fixed_logical_qcodes: true,
                allowed_large_matrix_types: vec![
                    LogicalQuantType::Q2B64,
                    LogicalQuantType::Q4B64,
                    LogicalQuantType::MixedQ2Q4B64,
                ],
                mtp: MtpIdentity {
                    resident_with_text: true,
                    layers: 1,
                    logical_sha256: digest('1'),
                    every_draft_token_verified: true,
                    draft_vocabulary: MtpDraftVocabularyIdentity {
                        token_ids: file("model/mtp-draft-token-ids.u32le", 160_000, '9'),
                        source_teacher_cache_set_sha256: digest('8'),
                        token_count: 40_000,
                        observed_output_tokens: 5_000_000,
                        overall_coverage_ppm: 975_000,
                        code_coverage_ppm: 960_000,
                        minimum_domain_coverage_ppm: 900_000,
                        minimum_language_coverage_ppm: 900_000,
                        target_verified_only: true,
                    },
                },
            },
            tokenizer: TokenizerContract {
                tokenizer_root_sha256: digest('2'),
                chat_template_sha256: CHAT_TEMPLATE_SHA256.into(),
                reasoning_format: "qwen38_reasoning_v1".into(),
                tool_call_format: "qwen38_tool_call_v1".into(),
                special_tokens: vec![
                    SpecialToken {
                        name: "end_of_text".into(),
                        id: END_OF_TEXT_ID,
                        text: "<|endoftext|>".into(),
                    },
                    SpecialToken {
                        name: "im_start".into(),
                        id: IM_START_ID,
                        text: "<|im_start|>".into(),
                    },
                    SpecialToken {
                        name: "im_end".into(),
                        id: IM_END_ID,
                        text: "<|im_end|>".into(),
                    },
                    SpecialToken {
                        name: "think_start".into(),
                        id: THINK_START_ID,
                        text: "<think>".into(),
                    },
                    SpecialToken {
                        name: "think_end".into(),
                        id: THINK_END_ID,
                        text: "</think>".into(),
                    },
                ],
                files: vec![
                    pinned_file("tokenizer/tokenizer.json", 12_809_320, TOKENIZER_SHA256),
                    pinned_file("tokenizer/chat_template.jinja", 8_952, CHAT_TEMPLATE_SHA256),
                ],
            },
            packages: vec![ReleasePackage {
                package_id: "text-mtp".into(),
                kind: PackageKind::TextMtp,
                default_download: true,
                packs: vec![pack],
            }],
            memory_profiles: vec![MemoryProfile {
                profile_id: "cuda-sm86-16k".into(),
                pack_id: "text-cuda-sm86".into(),
                context_tokens: 16_384,
                sessions: 1,
                resident_model_bytes: 8_000_000_000,
                persistent_backend_graph_bytes: 64 << 20,
                persistent_runtime_bytes: 64 << 20,
                linear_state_dtype: LinearStateDType::F32,
                linear_state_bytes_per_session: 303 << 19,
                mtp_draft_tokens: 0,
                speculative_state_strategy: SpeculativeStateStrategy::Disabled,
                speculative_linear_state_bytes_per_session: 0,
                kv: KvMemoryFormula {
                    fixed_bytes_per_session: 0,
                    bytes_per_token_per_session: 9_216,
                    retained_q4_tokens_per_session: 384,
                    q4_delta_bytes_per_token: 8_192,
                },
                mtp_kv: KvMemoryFormula {
                    fixed_bytes_per_session: 0,
                    bytes_per_token_per_session: 0,
                    retained_q4_tokens_per_session: 0,
                    q4_delta_bytes_per_token: 0,
                },
                prefill_scratch_peak_bytes: 128 << 20,
                decode_scratch_peak_bytes: 32 << 20,
                loader_transient_peak_bytes: 64 << 20,
                accelerator_unattributed_reserve_bytes: 64 << 20,
                hard_limit_bytes: 10 << 30,
            }],
            integrity: ManifestIntegrity {
                unsigned_sha256: String::new(),
                signature: None,
            },
        };
        manifest.seal_unsigned().unwrap();
        manifest
    }

    fn write_admitted_artifact(
        release: &mut ReleaseManifest,
    ) -> (tempfile::TempDir, ModelArtifact) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("text-cuda-sm86.ctoxq");
        let payload = Q2Block64::quantize(&[0.25; BLOCK_LEN])
            .unwrap()
            .encode()
            .to_vec();
        let manifest = ModelManifest {
            format: "ctox.q2q4.v1".into(),
            model: MODEL_ID.into(),
            revision: release.model.bf16_revision.clone(),
            alignment: DEFAULT_ALIGNMENT,
            target: "cuda-sm86".into(),
            recovery: Some(RecoveryProvenance {
                mode: RecoveryMode::Trained,
                format: "ctox.recovery.channel-scales.v2".into(),
                plan_sha256: digest('a'),
                fixed_logical_qcodes: true,
                artifact_sha256: Some(release.model.recovery_sha256.clone()),
                activation_stats_sha256: Some(digest('b')),
                report_sha256: Some(digest('c')),
                fanout_s_in_policy: None,
                fanout_group_sha256: None,
                fanout_group_count: None,
                fanout_logical_s_in_tensors: None,
            }),
            tensors: vec![TensorEntry {
                name: "model.embed_tokens.weight".into(),
                dtype: TensorDType::Q2B64,
                shape: vec![1, BLOCK_LEN as u64],
                offset: 0,
                length: payload.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&payload)),
                segments: Vec::new(),
            }],
        };
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let data_offset = align_up(
            (HEADER_BYTES + manifest_bytes.len()) as u64,
            DEFAULT_ALIGNMENT as u64,
        )
        .unwrap();
        let header = FileHeader {
            version: 1,
            manifest_len: manifest_bytes.len() as u64,
            data_offset,
            tensor_count: 1,
            alignment: DEFAULT_ALIGNMENT,
        };
        let mut file = Vec::from(header.encode());
        file.extend_from_slice(&manifest_bytes);
        file.resize(data_offset as usize, 0);
        file.extend_from_slice(&payload);
        fs::write(&path, &file).unwrap();

        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors).unwrap();
        let pack = &mut release.packages[0].packs[0];
        pack.artifact.bytes = file.len() as u64;
        pack.artifact.sha256 = format!("{:x}", Sha256::digest(&file));
        pack.artifact.chunks = vec![FileChunk {
            index: 0,
            offset: 0,
            length: file.len() as u64,
            sha256: pack.artifact.sha256.clone(),
        }];
        pack.artifact_manifest_sha256 = artifact.manifest_sha256().into();
        release.seal_unsigned().unwrap();
        (directory, artifact)
    }

    #[test]
    fn release_round_trips_and_binds_every_semantic_field() {
        let manifest = manifest();
        manifest.validate().unwrap();
        let json = serde_json::to_vec(&manifest).unwrap();
        let decoded: ReleaseManifest = serde_json::from_slice(&json).unwrap();
        decoded.validate().unwrap();
        let mut changed = decoded;
        changed.tokenizer.chat_template_sha256 = digest('5');
        assert!(changed.validate().is_err());
    }

    #[test]
    fn release_requires_the_pinned_tokenizer_assets_and_special_tokens() {
        let mut wrong_tokenizer = manifest();
        let file = wrong_tokenizer
            .tokenizer
            .files
            .iter_mut()
            .find(|file| file.relative_path.ends_with("tokenizer.json"))
            .unwrap();
        file.sha256 = digest('7');
        file.chunks[0].sha256 = digest('7');
        wrong_tokenizer.seal_unsigned().unwrap();
        assert!(wrong_tokenizer.validate().is_err());

        let mut missing_template = manifest();
        missing_template
            .tokenizer
            .files
            .retain(|file| !file.relative_path.ends_with("chat_template.jinja"));
        missing_template.seal_unsigned().unwrap();
        assert!(missing_template.validate().is_err());

        let mut wrong_special_token = manifest();
        wrong_special_token
            .tokenizer
            .special_tokens
            .iter_mut()
            .find(|token| token.name == "think_end")
            .unwrap()
            .id -= 1;
        wrong_special_token.seal_unsigned().unwrap();
        assert!(wrong_special_token.validate().is_err());
    }

    #[test]
    fn release_binds_canonical_restricted_mtp_vocabulary() {
        let mut wrong_size = manifest();
        wrong_size.model.mtp.draft_vocabulary.token_ids.bytes -= 4;
        wrong_size.seal_unsigned().unwrap();
        assert!(wrong_size.validate().is_err());

        let mut full_vocab = manifest();
        full_vocab.model.mtp.draft_vocabulary.token_count =
            Qwen38Config::default().vocab_size as u32;
        full_vocab.model.mtp.draft_vocabulary.token_ids.bytes =
            full_vocab.model.mtp.draft_vocabulary.token_count as u64 * 4;
        full_vocab.seal_unsigned().unwrap();
        assert!(full_vocab.validate().is_err());

        let mut unverified = manifest();
        unverified.model.mtp.draft_vocabulary.target_verified_only = false;
        unverified.seal_unsigned().unwrap();
        assert!(unverified.validate().is_err());
    }

    #[test]
    fn release_loads_and_rehashes_canonical_mtp_token_ids() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model/mtp-draft-token-ids.u32le");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let encoded = [1_u32, 7, 42]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        fs::write(&path, &encoded).unwrap();
        let digest = format!("{:x}", Sha256::digest(&encoded));
        let mut manifest = manifest();
        manifest.model.mtp.draft_vocabulary.token_count = 3;
        manifest.model.mtp.draft_vocabulary.token_ids = ReleaseFile {
            relative_path: "model/mtp-draft-token-ids.u32le".into(),
            bytes: encoded.len() as u64,
            sha256: digest.clone(),
            chunks: vec![FileChunk {
                index: 0,
                offset: 0,
                length: encoded.len() as u64,
                sha256: digest,
            }],
        };
        manifest.seal_unsigned().unwrap();
        assert_eq!(
            manifest.load_mtp_draft_token_ids(directory.path()).unwrap(),
            vec![1, 7, 42]
        );

        fs::write(&path, [0_u8; 12]).unwrap();
        assert!(manifest.load_mtp_draft_token_ids(directory.path()).is_err());
    }

    #[test]
    fn backend_requantization_and_duplicate_residency_are_rejected() {
        let mut manifest = manifest();
        manifest.packages[0].packs[0].requantized = true;
        manifest.seal_unsigned().unwrap();
        assert!(manifest.validate().is_err());
        manifest.packages[0].packs[0].requantized = false;
        manifest.packages[0].packs[0]
            .loader
            .maximum_resident_full_model_copies = 2;
        manifest.seal_unsigned().unwrap();
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn memory_profiles_use_the_declared_context_formula_and_limit() {
        let mut over_limit = manifest();
        let profile = &over_limit.memory_profiles[0];
        assert_eq!(profile.kv.bytes(16_384, 1).unwrap(), 154_140_672);
        assert!(profile.maximum_peak_bytes().unwrap() < profile.hard_limit_bytes);
        over_limit.memory_profiles[0].hard_limit_bytes = 8_000_000_000;
        over_limit.seal_unsigned().unwrap();
        assert!(over_limit.validate().is_err());

        let mut context_overflow = manifest();
        context_overflow.memory_profiles[0].context_tokens = 262_145;
        context_overflow.seal_unsigned().unwrap();
        assert!(context_overflow.validate().is_err());

        let mut hidden_mtp_memory = manifest();
        hidden_mtp_memory.memory_profiles[0].mtp_draft_tokens = 4;
        hidden_mtp_memory.seal_unsigned().unwrap();
        assert!(hidden_mtp_memory.validate().is_err());

        let mut wrong_replay_bytes = manifest();
        wrong_replay_bytes.memory_profiles[0].mtp_draft_tokens = 4;
        wrong_replay_bytes.memory_profiles[0].speculative_state_strategy =
            SpeculativeStateStrategy::ReplayOnReject;
        wrong_replay_bytes.memory_profiles[0]
            .mtp_kv
            .fixed_bytes_per_session = 1;
        wrong_replay_bytes.memory_profiles[0].speculative_linear_state_bytes_per_session = 1;
        wrong_replay_bytes.seal_unsigned().unwrap();
        assert!(wrong_replay_bytes.validate().is_err());
    }

    #[test]
    fn trusted_ed25519_signature_binds_the_unsigned_digest() {
        let mut manifest = manifest();
        let signing_key = SigningKey::from_bytes(&[7_u8; 32]);
        let digest =
            decode_hex::<32>(&manifest.integrity.unsigned_sha256, "unsigned digest").unwrap();
        let signature = signing_key.sign(&digest);
        manifest.integrity.signature = Some(ManifestSignature {
            algorithm: ED25519_ALGORITHM.into(),
            key_id: "test-key".into(),
            signature_hex: signature
                .to_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        });
        manifest
            .verify_signature("test-key", &signing_key.verifying_key().to_bytes())
            .unwrap();
        manifest.release_id.push_str("-changed");
        assert!(manifest
            .verify_signature("test-key", &signing_key.verifying_key().to_bytes())
            .is_err());
    }

    #[test]
    fn physical_backend_packs_must_share_one_logical_tensor_root() {
        let mut manifest = manifest();
        let mut metal = manifest.packages[0].packs[0].clone();
        metal.pack_id = "text-metal-apple9".into();
        metal.backend = BackendKind::Metal;
        metal.hardware_profile = "apple9".into();
        metal.artifact = file("packs/text-metal-apple9.ctoxq", 8_000_001_024, '5');
        manifest.packages[0].packs.push(metal);
        let mut metal_memory = manifest.memory_profiles[0].clone();
        metal_memory.profile_id = "metal-apple9-16k".into();
        metal_memory.pack_id = "text-metal-apple9".into();
        manifest.memory_profiles.push(metal_memory);
        manifest.seal_unsigned().unwrap();
        manifest
            .verify_backend_pack_equivalence("text-cuda-sm86", "text-metal-apple9")
            .unwrap();

        manifest.packages[0].packs[1].logical_tensor_root_sha256 = digest('6');
        manifest.seal_unsigned().unwrap();
        assert!(manifest
            .verify_backend_pack_equivalence("text-cuda-sm86", "text-metal-apple9")
            .is_err());
    }

    #[test]
    fn signed_pack_identity_admits_only_the_bound_ctoxq_manifest() {
        let mut release = manifest();
        let (_directory, artifact) = write_admitted_artifact(&mut release);
        release.admit_artifact("text-cuda-sm86", &artifact).unwrap();

        release.packages[0].packs[0].artifact_manifest_sha256 = digest('9');
        release.seal_unsigned().unwrap();
        assert!(release.admit_artifact("text-cuda-sm86", &artifact).is_err());
    }
}
