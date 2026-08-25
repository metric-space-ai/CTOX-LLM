use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::quant::{BLOCK_LEN, Q2_BLOCK_BYTES, Q4_BLOCK_BYTES};
use crate::{EngineError, Result};

pub const MAGIC: &[u8; 8] = b"CTOXQ2Q4";
pub const FORMAT_VERSION: u32 = 1;
pub const ENDIAN_MARKER: u32 = 0x0102_0304;
pub const HEADER_BYTES: usize = 64;
pub const DEFAULT_ALIGNMENT: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDType {
    Q2B64,
    Q4B64,
    F16,
    F32,
}

impl TensorDType {
    pub fn expected_bytes(self, elements: u64) -> Result<u64> {
        match self {
            Self::Q2B64 => elements
                .div_ceil(BLOCK_LEN as u64)
                .checked_mul(Q2_BLOCK_BYTES as u64),
            Self::Q4B64 => elements
                .div_ceil(BLOCK_LEN as u64)
                .checked_mul(Q4_BLOCK_BYTES as u64),
            Self::F16 => elements.checked_mul(2),
            Self::F32 => elements.checked_mul(4),
        }
        .ok_or_else(|| EngineError::InvalidArtifact("tensor byte length overflows u64".into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    /// Offset relative to the tensor-data region.
    pub offset: u64,
    pub length: u64,
    pub sha256: String,
}

impl TensorEntry {
    pub fn elements(&self) -> Result<u64> {
        if self.shape.is_empty() || self.shape.contains(&0) {
            return Err(EngineError::Shape(format!(
                "tensor {} has an empty or zero dimension",
                self.name
            )));
        }
        self.shape.iter().try_fold(1_u64, |total, dimension| {
            total.checked_mul(*dimension).ok_or_else(|| {
                EngineError::Shape(format!("tensor {} shape overflows u64", self.name))
            })
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub format: String,
    pub model: String,
    pub revision: String,
    pub alignment: u32,
    pub target: String,
    pub tensors: Vec<TensorEntry>,
}

impl ModelManifest {
    pub fn validate(&self, data_len: u64) -> Result<()> {
        if self.format != "ctox.q2q4.v1" {
            return Err(EngineError::InvalidArtifact(format!(
                "unsupported manifest format {}",
                self.format
            )));
        }
        if self.model.is_empty() || self.revision.len() < 7 {
            return Err(EngineError::InvalidArtifact(
                "model and immutable revision are required".into(),
            ));
        }
        if self.alignment < 64 || !self.alignment.is_power_of_two() {
            return Err(EngineError::InvalidArtifact(format!(
                "alignment {} must be a power of two >= 64",
                self.alignment
            )));
        }

        let mut names = HashSet::with_capacity(self.tensors.len());
        let mut ranges = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            if tensor.name.is_empty() || !names.insert(&tensor.name) {
                return Err(EngineError::InvalidArtifact(format!(
                    "tensor name is empty or duplicated: {}",
                    tensor.name
                )));
            }
            if tensor.offset % self.alignment as u64 != 0 {
                return Err(EngineError::InvalidArtifact(format!(
                    "tensor {} offset {} is not {}-byte aligned",
                    tensor.name, tensor.offset, self.alignment
                )));
            }
            if tensor.sha256.len() != 64
                || !tensor.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(EngineError::InvalidArtifact(format!(
                    "tensor {} has an invalid SHA-256",
                    tensor.name
                )));
            }
            let expected = tensor.dtype.expected_bytes(tensor.elements()?)?;
            if tensor.length != expected {
                return Err(EngineError::InvalidArtifact(format!(
                    "tensor {} has {} bytes, {:?} shape requires {}",
                    tensor.name, tensor.length, tensor.dtype, expected
                )));
            }
            let end = tensor.offset.checked_add(tensor.length).ok_or_else(|| {
                EngineError::InvalidArtifact(format!("tensor {} range overflows", tensor.name))
            })?;
            if end > data_len {
                return Err(EngineError::InvalidArtifact(format!(
                    "tensor {} ends at {}, data region has {} bytes",
                    tensor.name, end, data_len
                )));
            }
            ranges.push((tensor.offset, end, tensor.name.as_str()));
        }

        ranges.sort_unstable_by_key(|range| range.0);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(EngineError::InvalidArtifact(format!(
                    "tensor ranges {} and {} overlap",
                    pair[0].2, pair[1].2
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileHeader {
    pub manifest_len: u64,
    pub data_offset: u64,
    pub tensor_count: u32,
    pub alignment: u32,
}

impl FileHeader {
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_BYTES {
            return Err(EngineError::InvalidArtifact(format!(
                "header is {} bytes, expected at least {HEADER_BYTES}",
                bytes.len()
            )));
        }
        if &bytes[..8] != MAGIC {
            return Err(EngineError::InvalidArtifact("bad CTOX Q2/Q4 magic".into()));
        }
        let u32_at = |offset: usize| {
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("header bounds"))
        };
        let u64_at = |offset: usize| {
            u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("header bounds"))
        };
        let version = u32_at(8);
        if version != FORMAT_VERSION {
            return Err(EngineError::InvalidArtifact(format!(
                "format version {version} is unsupported"
            )));
        }
        if u32_at(12) != ENDIAN_MARKER {
            return Err(EngineError::InvalidArtifact(
                "endianness marker mismatch".into(),
            ));
        }
        Ok(Self {
            manifest_len: u64_at(16),
            data_offset: u64_at(24),
            tensor_count: u32_at(32),
            alignment: u32_at(36),
        })
    }

    pub fn encode(self) -> [u8; HEADER_BYTES] {
        let mut bytes = [0_u8; HEADER_BYTES];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&ENDIAN_MARKER.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.manifest_len.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.data_offset.to_le_bytes());
        bytes[32..36].copy_from_slice(&self.tensor_count.to_le_bytes());
        bytes[36..40].copy_from_slice(&self.alignment.to_le_bytes());
        bytes
    }
}

pub fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(EngineError::InvalidArtifact(format!(
            "invalid alignment {alignment}"
        )));
    }
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| EngineError::InvalidArtifact("aligned offset overflows u64".into()))
}

#[derive(Debug, Clone)]
pub struct PackedTensor {
    pub name: String,
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    pub bytes: Vec<u8>,
}

pub struct ArtifactBuilder {
    pub model: String,
    pub revision: String,
    pub target: String,
    pub alignment: u32,
    pub tensors: Vec<PackedTensor>,
}

impl ArtifactBuilder {
    pub fn write_new(self, path: impl AsRef<Path>) -> Result<ModelManifest> {
        if self.alignment < 64 || !self.alignment.is_power_of_two() {
            return Err(EngineError::InvalidArtifact(format!(
                "alignment {} must be a power of two >= 64",
                self.alignment
            )));
        }
        let mut data_offset = 0_u64;
        let mut entries = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            data_offset = align_up(data_offset, self.alignment as u64)?;
            let elements = tensor.shape.iter().try_fold(1_u64, |total, dimension| {
                total.checked_mul(*dimension).ok_or_else(|| {
                    EngineError::Shape(format!("tensor {} shape overflows", tensor.name))
                })
            })?;
            let expected = tensor.dtype.expected_bytes(elements)?;
            if tensor.bytes.len() as u64 != expected {
                return Err(EngineError::InvalidArtifact(format!(
                    "tensor {} has {} packed bytes, expected {}",
                    tensor.name,
                    tensor.bytes.len(),
                    expected
                )));
            }
            entries.push(TensorEntry {
                name: tensor.name.clone(),
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                offset: data_offset,
                length: expected,
                sha256: format!("{:x}", Sha256::digest(&tensor.bytes)),
            });
            data_offset = data_offset.checked_add(expected).ok_or_else(|| {
                EngineError::InvalidArtifact("tensor data length overflows".into())
            })?;
        }

        let manifest = ModelManifest {
            format: "ctox.q2q4.v1".into(),
            model: self.model,
            revision: self.revision,
            alignment: self.alignment,
            target: self.target,
            tensors: entries,
        };
        manifest.validate(data_offset)?;
        let manifest_bytes = serde_json::to_vec(&manifest)?;
        let file_data_offset = align_up(
            (HEADER_BYTES + manifest_bytes.len()) as u64,
            self.alignment as u64,
        )?;
        let header = FileHeader {
            manifest_len: manifest_bytes.len() as u64,
            data_offset: file_data_offset,
            tensor_count: manifest.tensors.len() as u32,
            alignment: self.alignment,
        };

        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(&header.encode())?;
        file.write_all(&manifest_bytes)?;
        write_zeroes(
            &mut file,
            file_data_offset as usize - HEADER_BYTES - manifest_bytes.len(),
        )?;
        let mut written = 0_u64;
        for tensor in &self.tensors {
            let aligned = align_up(written, self.alignment as u64)?;
            write_zeroes(&mut file, (aligned - written) as usize)?;
            file.write_all(&tensor.bytes)?;
            written = aligned + tensor.bytes.len() as u64;
        }
        file.sync_all()?;
        Ok(manifest)
    }
}

fn write_zeroes(writer: &mut impl Write, mut bytes: usize) -> Result<()> {
    const ZEROES: [u8; 4096] = [0; 4096];
    while bytes > 0 {
        let chunk = bytes.min(ZEROES.len());
        writer.write_all(&ZEROES[..chunk])?;
        bytes -= chunk;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{ChecksumPolicy, ModelArtifact};
    use crate::quant::Q2Block64;

    #[test]
    fn header_round_trip() {
        let header = FileHeader {
            manifest_len: 1234,
            data_offset: 1536,
            tensor_count: 42,
            alignment: 256,
        };
        assert_eq!(FileHeader::decode(&header.encode()).unwrap(), header);
    }

    #[test]
    fn q3_is_not_a_manifest_dtype() {
        let json = r#"{
          "format":"ctox.q2q4.v1","model":"qwen","revision":"1234567",
          "alignment":256,"target":"cpu","tensors":[{
            "name":"bad","dtype":"q3","shape":[64],"offset":0,
            "length":24,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          }]}
        "#;
        assert!(serde_json::from_str::<ModelManifest>(json).is_err());
    }

    #[test]
    fn artifact_builder_round_trips_and_checksums() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tiny.ctoxq");
        let values = [0.25_f32; BLOCK_LEN];
        let bytes = Q2Block64::quantize(&values).unwrap().encode().to_vec();
        ArtifactBuilder {
            model: "test/qwen".into(),
            revision: "0123456789abcdef".into(),
            target: "test".into(),
            alignment: DEFAULT_ALIGNMENT,
            tensors: vec![PackedTensor {
                name: "model.embed_tokens.weight".into(),
                dtype: TensorDType::Q2B64,
                shape: vec![1, BLOCK_LEN as u64],
                bytes,
            }],
        }
        .write_new(&path)
        .unwrap();
        let artifact = ModelArtifact::open(path, ChecksumPolicy::AllTensors).unwrap();
        assert_eq!(artifact.manifest().tensors.len(), 1);
    }
}
