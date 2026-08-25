use std::fs::File;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};

use crate::format::{FileHeader, ModelManifest, HEADER_BYTES};
use crate::{EngineError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumPolicy {
    ManifestOnly,
    AllTensors,
}

pub struct ModelArtifact {
    mmap: Mmap,
    header: FileHeader,
    manifest: ModelManifest,
}

impl ModelArtifact {
    pub fn open(path: impl AsRef<Path>, checksum_policy: ChecksumPolicy) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the mapping is immutable and held for the lifetime of this
        // object. Production packaging treats model files as immutable content-
        // addressed artifacts; callers must not modify the file concurrently.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let header = FileHeader::decode(&mmap)?;
        let manifest_start = HEADER_BYTES;
        let manifest_end = manifest_start
            .checked_add(header.manifest_len as usize)
            .ok_or_else(|| EngineError::InvalidArtifact("manifest range overflows".into()))?;
        if manifest_end > mmap.len() || header.data_offset as usize > mmap.len() {
            return Err(EngineError::InvalidArtifact(
                "manifest or data offset lies beyond file".into(),
            ));
        }
        if header.data_offset < manifest_end as u64 {
            return Err(EngineError::InvalidArtifact(
                "tensor data overlaps manifest".into(),
            ));
        }
        let manifest: ModelManifest = serde_json::from_slice(&mmap[manifest_start..manifest_end])?;
        let expected_format = format!("ctox.q2q4.v{}", header.version);
        if manifest.format != expected_format {
            return Err(EngineError::InvalidArtifact(format!(
                "header version {} requires manifest {}, got {}",
                header.version, expected_format, manifest.format
            )));
        }
        if manifest.tensors.len() != header.tensor_count as usize {
            return Err(EngineError::InvalidArtifact(format!(
                "header declares {} tensors, manifest has {}",
                header.tensor_count,
                manifest.tensors.len()
            )));
        }
        if manifest.alignment != header.alignment {
            return Err(EngineError::InvalidArtifact(
                "header and manifest alignment differ".into(),
            ));
        }
        manifest.validate(mmap.len() as u64 - header.data_offset)?;

        let artifact = Self {
            mmap,
            header,
            manifest,
        };
        if checksum_policy == ChecksumPolicy::AllTensors {
            artifact.verify_checksums()?;
        }
        Ok(artifact)
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let tensor = self
            .manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| EngineError::InvalidArtifact(format!("tensor {name} not found")))?;
        let start = self.header.data_offset as usize + tensor.offset as usize;
        let end = start + tensor.length as usize;
        Ok(&self.mmap[start..end])
    }

    pub fn verify_checksums(&self) -> Result<()> {
        for tensor in &self.manifest.tensors {
            let digest = Sha256::digest(self.tensor_bytes(&tensor.name)?);
            let actual = format!("{digest:x}");
            if actual != tensor.sha256.to_ascii_lowercase() {
                return Err(EngineError::InvalidArtifact(format!(
                    "tensor {} checksum mismatch: expected {}, got {}",
                    tensor.name, tensor.sha256, actual
                )));
            }
        }
        Ok(())
    }
}
