use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};

use crate::backend::{Activation, FusedMatVec, RecoveredRow, ScaleSlice};
use crate::fanout::{qwen38_fanout_groups, FanoutGroup, QWEN38_FANOUT_POLICY};
use crate::format::{
    FileHeader, ModelManifest, QuantSegment, TensorDType, TensorEntry, HEADER_BYTES,
};
use crate::{EngineError, Qwen38Config, Result};
use half::f16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumPolicy {
    ManifestOnly,
    AllTensors,
}

#[derive(Clone)]
pub struct ModelArtifact {
    mmap: Arc<Mmap>,
    header: FileHeader,
    manifest: ModelManifest,
    manifest_sha256: String,
    tensor_indices: HashMap<String, usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub entry: &'a TensorEntry,
    pub bytes: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizedMatrixView<'a> {
    pub dtype: TensorDType,
    pub weights: &'a [u8],
    pub segments: &'a [QuantSegment],
    pub rows: usize,
    pub columns: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizedRowView<'a> {
    pub dtype: TensorDType,
    pub weights: &'a [u8],
    pub columns: usize,
}

impl<'a> QuantizedMatrixView<'a> {
    pub fn row(self, row: usize) -> Result<QuantizedRowView<'a>> {
        if row >= self.rows {
            return Err(EngineError::Shape(format!(
                "quantized row {row} exceeds {} rows",
                self.rows
            )));
        }
        let (dtype, payload, local_row) = match self.dtype {
            TensorDType::Q2B64 | TensorDType::Q4B64 => (self.dtype, self.weights, row),
            TensorDType::MixedQ2Q4B64 => {
                let segment = self
                    .segments
                    .iter()
                    .find(|segment| {
                        usize::try_from(segment.row_start).is_ok_and(|start| row >= start)
                            && usize::try_from(segment.row_end).is_ok_and(|end| row < end)
                    })
                    .ok_or_else(|| {
                        EngineError::InvalidArtifact(format!(
                            "mixed quantized row {row} has no segment"
                        ))
                    })?;
                let start = usize::try_from(segment.row_start).map_err(|_| {
                    EngineError::InvalidArtifact("mixed row start overflows usize".into())
                })?;
                let offset = usize::try_from(segment.offset).map_err(|_| {
                    EngineError::InvalidArtifact("mixed row offset overflows usize".into())
                })?;
                let length = usize::try_from(segment.length).map_err(|_| {
                    EngineError::InvalidArtifact("mixed row length overflows usize".into())
                })?;
                let end = offset.checked_add(length).ok_or_else(|| {
                    EngineError::InvalidArtifact("mixed row payload range overflows".into())
                })?;
                if end > self.weights.len() {
                    return Err(EngineError::InvalidArtifact(
                        "mixed row segment exceeds matrix payload".into(),
                    ));
                }
                (segment.dtype, &self.weights[offset..end], row - start)
            }
            other => return Err(EngineError::UnsupportedDType(format!("{other:?}"))),
        };
        let block_bytes = match dtype {
            TensorDType::Q2B64 => crate::quant::Q2_BLOCK_BYTES,
            TensorDType::Q4B64 => crate::quant::Q4_BLOCK_BYTES,
            other => return Err(EngineError::UnsupportedDType(format!("{other:?}"))),
        };
        let row_bytes = self
            .columns
            .checked_div(crate::quant::BLOCK_LEN)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or_else(|| EngineError::Shape("quantized row byte size overflows".into()))?;
        let start = local_row
            .checked_mul(row_bytes)
            .ok_or_else(|| EngineError::Shape("quantized row offset overflows".into()))?;
        let end = start
            .checked_add(row_bytes)
            .ok_or_else(|| EngineError::Shape("quantized row range overflows".into()))?;
        if end > payload.len() {
            return Err(EngineError::InvalidArtifact(
                "quantized row exceeds its packed payload".into(),
            ));
        }
        Ok(QuantizedRowView {
            dtype,
            weights: &payload[start..end],
            columns: self.columns,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FloatTensorView<'a> {
    F16Le(&'a [u8]),
    F32Le(&'a [u8]),
}

impl<'a> FloatTensorView<'a> {
    pub fn len(self) -> usize {
        match self {
            Self::F16Le(bytes) => bytes.len() / 2,
            Self::F32Le(bytes) => bytes.len() / 4,
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn value(self, index: usize) -> Result<f32> {
        if index >= self.len() {
            return Err(EngineError::Shape(format!(
                "float tensor index {index} exceeds {} values",
                self.len()
            )));
        }
        let value = match self {
            Self::F16Le(bytes) => {
                let offset = index * 2;
                f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
            }
            Self::F32Le(bytes) => {
                let offset = index * 4;
                f32::from_le_bytes(
                    bytes[offset..offset + 4]
                        .try_into()
                        .expect("checked bounds"),
                )
            }
        };
        if !value.is_finite() {
            return Err(EngineError::InvalidArtifact(format!(
                "float tensor value {index} is non-finite"
            )));
        }
        Ok(value)
    }

    pub fn to_f32_vec(self) -> Result<Vec<f32>> {
        (0..self.len()).map(|index| self.value(index)).collect()
    }

    /// Borrow recovery scales in the exact on-disk FP16 representation used
    /// by fused CPU/Metal/CUDA kernels. F32 tensors are deliberately rejected:
    /// converting them would not be a zero-copy recovery-scale binding.
    pub fn as_recovery_scales(self) -> Result<ScaleSlice<'a>> {
        match self {
            Self::F16Le(bytes) => Ok(ScaleSlice::F16Le(bytes)),
            Self::F32Le(_) => Err(EngineError::UnsupportedDType(
                "recovery scales must be packed FP16".into(),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RecoveredMatrixView<'a> {
    pub matrix: QuantizedMatrixView<'a>,
    pub s_in: FloatTensorView<'a>,
    pub s_out: FloatTensorView<'a>,
}

impl<'a> RecoveredMatrixView<'a> {
    /// Construct one fused projection directly from mmap-backed quant codes
    /// and FP16 recovery scales. The input may have a shorter lifetime than
    /// the artifact; no model tensor is copied or repacked.
    pub fn operation<'b>(self, input: &'b [f32], activation: Activation) -> Result<FusedMatVec<'b>>
    where
        'a: 'b,
    {
        Ok(FusedMatVec {
            dtype: self.matrix.dtype,
            weights: self.matrix.weights,
            segments: self.matrix.segments,
            rows: self.matrix.rows,
            columns: self.matrix.columns,
            input,
            s_in: Some(self.s_in.as_recovery_scales()?),
            s_out: Some(self.s_out.as_recovery_scales()?),
            bias: None,
            activation,
        })
    }

    pub fn row_operation(self, row: usize) -> Result<RecoveredRow<'a>> {
        let packed = self.matrix.row(row)?;
        Ok(RecoveredRow {
            dtype: packed.dtype,
            weights: packed.weights,
            columns: packed.columns,
            s_in: self.s_in.as_recovery_scales()?,
            s_out: self.s_out.value(row)?,
        })
    }
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
        let manifest_bytes = &mmap[manifest_start..manifest_end];
        let manifest_sha256 = format!("{:x}", Sha256::digest(manifest_bytes));
        let manifest: ModelManifest = serde_json::from_slice(manifest_bytes)?;
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
        let tensor_indices = manifest
            .tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.name.clone(), index))
            .collect();

        let artifact = Self {
            mmap: Arc::new(mmap),
            header,
            manifest,
            manifest_sha256,
            tensor_indices,
        };
        artifact.verify_recovery_fanout()?;
        if checksum_policy == ChecksumPolicy::AllTensors {
            artifact.verify_checksums()?;
        }
        Ok(artifact)
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    /// A bound shared-A8 policy is accepted only when every mmap-backed FP16
    /// input correction in each frozen fan-out group is byte-identical.
    pub fn verify_recovery_fanout(&self) -> Result<()> {
        let Some(recovery) = &self.manifest.recovery else {
            return Ok(());
        };
        if recovery.fanout_s_in_policy.as_deref() != Some(QWEN38_FANOUT_POLICY) {
            return Ok(());
        }
        verify_fanout_scale_bytes(&qwen38_fanout_groups(&Qwen38Config::default()), |name| {
            self.tensor_bytes(name)
        })
    }

    pub fn file_bytes(&self) -> u64 {
        self.mmap.len() as u64
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        Ok(self.tensor(name)?.bytes)
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let index = self
            .tensor_indices
            .get(name)
            .ok_or_else(|| EngineError::InvalidArtifact(format!("tensor {name} not found")))?;
        let tensor = &self.manifest.tensors[*index];
        let start = self.header.data_offset as usize + tensor.offset as usize;
        let end = start + tensor.length as usize;
        Ok(TensorView {
            entry: tensor,
            bytes: &self.mmap[start..end],
        })
    }

    pub fn float_tensor(&self, name: &str) -> Result<FloatTensorView<'_>> {
        let tensor = self.tensor(name)?;
        match tensor.entry.dtype {
            TensorDType::F16 => Ok(FloatTensorView::F16Le(tensor.bytes)),
            TensorDType::F32 => Ok(FloatTensorView::F32Le(tensor.bytes)),
            other => Err(EngineError::UnsupportedDType(format!(
                "tensor {name} is {other:?}, expected F16/F32"
            ))),
        }
    }

    pub fn quantized_matrix(&self, name: &str) -> Result<QuantizedMatrixView<'_>> {
        let tensor = self.tensor(name)?;
        if tensor.entry.shape.len() != 2 {
            return Err(EngineError::Shape(format!(
                "quantized tensor {name} is not a matrix"
            )));
        }
        if !matches!(
            tensor.entry.dtype,
            TensorDType::Q2B64 | TensorDType::Q4B64 | TensorDType::MixedQ2Q4B64
        ) {
            return Err(EngineError::UnsupportedDType(format!(
                "tensor {name} is {:?}, expected quantized matrix",
                tensor.entry.dtype
            )));
        }
        let rows = usize::try_from(tensor.entry.shape[0])
            .map_err(|_| EngineError::Shape(format!("tensor {name} rows overflow usize")))?;
        let columns = usize::try_from(tensor.entry.shape[1])
            .map_err(|_| EngineError::Shape(format!("tensor {name} columns overflow usize")))?;
        Ok(QuantizedMatrixView {
            dtype: tensor.entry.dtype,
            weights: tensor.bytes,
            segments: &tensor.entry.segments,
            rows,
            columns,
        })
    }

    pub fn recovered_matrix(&self, name: &str) -> Result<RecoveredMatrixView<'_>> {
        let matrix = self.quantized_matrix(name)?;
        let s_in = self.float_tensor(&format!("{name}.s_in"))?;
        let s_out = self.float_tensor(&format!("{name}.s_out"))?;
        if s_in.len() != matrix.columns || s_out.len() != matrix.rows {
            return Err(EngineError::Shape(format!(
                "recovery scales for {name} differ from {}x{} matrix",
                matrix.rows, matrix.columns
            )));
        }
        Ok(RecoveredMatrixView {
            matrix,
            s_in,
            s_out,
        })
    }

    pub fn verify_checksums(&self) -> Result<()> {
        for tensor in &self.manifest.tensors {
            let digest = Sha256::digest(self.tensor(&tensor.name)?.bytes);
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

fn verify_fanout_scale_bytes<'a, F>(groups: &[FanoutGroup], mut resolve: F) -> Result<()>
where
    F: FnMut(&str) -> Result<&'a [u8]>,
{
    for group in groups {
        let mut names = group.scale_names.iter();
        let first_name = names.next().ok_or_else(|| {
            EngineError::InvalidArtifact("recovery fan-out group is empty".into())
        })?;
        let reference = resolve(first_name)?;
        for name in names {
            if resolve(name)? != reference {
                return Err(EngineError::InvalidArtifact(format!(
                    "recovery fan-out scales differ at {}",
                    group.prefix
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::Backend;
    use crate::format::{ArtifactBuilder, PackedTensor, DEFAULT_ALIGNMENT};
    use crate::quant::{Q2Block64, BLOCK_LEN};

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn indexed_views_bind_matrix_and_recovery_without_repacking() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("views.ctoxq");
        let source = [0.5_f32; BLOCK_LEN];
        let weights = Q2Block64::quantize(&source).unwrap().encode().to_vec();
        let f32_values = [0.25_f32, -0.75_f32];
        ArtifactBuilder {
            model: "test/qwen".into(),
            revision: "0123456789abcdef".into(),
            target: "test".into(),
            alignment: DEFAULT_ALIGNMENT,
            tensors: vec![
                PackedTensor {
                    name: "linear.weight".into(),
                    dtype: TensorDType::Q2B64,
                    shape: vec![1, BLOCK_LEN as u64],
                    bytes: weights,
                },
                PackedTensor {
                    name: "linear.weight.s_in".into(),
                    dtype: TensorDType::F16,
                    shape: vec![BLOCK_LEN as u64],
                    bytes: f16_bytes(&vec![1.0; BLOCK_LEN]),
                },
                PackedTensor {
                    name: "linear.weight.s_out".into(),
                    dtype: TensorDType::F16,
                    shape: vec![1],
                    bytes: f16_bytes(&[1.5]),
                },
                PackedTensor {
                    name: "A_log".into(),
                    dtype: TensorDType::F32,
                    shape: vec![2],
                    bytes: f32_values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                },
            ],
        }
        .write_new(&path)
        .unwrap();

        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors).unwrap();
        let recovered = artifact.recovered_matrix("linear.weight").unwrap();
        assert_eq!(
            (recovered.matrix.rows, recovered.matrix.columns),
            (1, BLOCK_LEN)
        );
        assert_eq!(recovered.matrix.dtype, TensorDType::Q2B64);
        assert!(recovered.matrix.segments.is_empty());
        assert_eq!(recovered.s_in.len(), BLOCK_LEN);
        assert_eq!(recovered.s_in.value(BLOCK_LEN - 1).unwrap(), 1.0);
        assert_eq!(recovered.s_out.value(0).unwrap(), 1.5);
        assert!(matches!(
            recovered.s_in.as_recovery_scales().unwrap(),
            ScaleSlice::F16Le(_)
        ));
        assert_eq!(
            artifact
                .float_tensor("A_log")
                .unwrap()
                .to_f32_vec()
                .unwrap(),
            f32_values
        );
        assert!(artifact.tensor("missing").is_err());
        assert!(artifact.quantized_matrix("A_log").is_err());
        assert!(artifact.float_tensor("linear.weight").is_err());

        let input = [1.0_f32; BLOCK_LEN];
        let operation = recovered.operation(&input, Activation::Identity).unwrap();
        let output = CpuBackend::scalar_verifier()
            .fused_matvec(&operation)
            .unwrap();
        assert!((output[0] - 48.0).abs() < 1e-6);

        let row = recovered.row_operation(0).unwrap();
        let embedding = CpuBackend::scalar_verifier().recovered_row(&row).unwrap();
        assert_eq!(embedding.len(), BLOCK_LEN);
        assert!(embedding.iter().all(|value| (*value - 0.75).abs() < 1e-6));
    }

    #[test]
    fn float_view_rejects_bounds_and_nonfinite_values() {
        let finite = 1.0_f32.to_le_bytes();
        let view = FloatTensorView::F32Le(&finite);
        assert_eq!(view.value(0).unwrap(), 1.0);
        assert!(view.value(1).is_err());
        assert!(view.as_recovery_scales().is_err());
        let nonfinite = f32::NAN.to_le_bytes();
        assert!(FloatTensorView::F32Le(&nonfinite).value(0).is_err());
    }

    #[test]
    fn fanout_scale_verifier_rejects_one_different_byte() {
        let groups = [FanoutGroup {
            kind: "test",
            prefix: "layer.attn".into(),
            scale_names: vec!["q.s_in".into(), "k.s_in".into(), "v.s_in".into()],
        }];
        let equal = [1_u8, 2, 3, 4];
        let different = [1_u8, 2, 3, 5];
        assert!(verify_fanout_scale_bytes(&groups, |_name| Ok(&equal)).is_ok());
        assert!(verify_fanout_scale_bytes(&groups, |name| {
            Ok(if name == "v.s_in" { &different } else { &equal })
        })
        .is_err());
    }

    #[test]
    fn mixed_matrix_resolves_only_the_requested_row_group() {
        let q2 = Q2Block64::quantize(&[0.25; BLOCK_LEN])
            .unwrap()
            .encode()
            .to_vec();
        let q4 = crate::quant::Q4Block64::quantize(&[-0.5; BLOCK_LEN])
            .unwrap()
            .encode()
            .to_vec();
        let mut weights = q2.clone();
        weights.extend_from_slice(&q4);
        let q2_length = u64::try_from(q2.len()).unwrap();
        let segments = [
            QuantSegment {
                group_index: 0,
                row_start: 0,
                row_end: 1,
                dtype: TensorDType::Q2B64,
                offset: 0,
                length: q2_length,
            },
            QuantSegment {
                group_index: 1,
                row_start: 1,
                row_end: 2,
                dtype: TensorDType::Q4B64,
                offset: q2_length,
                length: u64::try_from(q4.len()).unwrap(),
            },
        ];
        let matrix = QuantizedMatrixView {
            dtype: TensorDType::MixedQ2Q4B64,
            weights: &weights,
            segments: &segments,
            rows: 2,
            columns: BLOCK_LEN,
        };
        let first = matrix.row(0).unwrap();
        let second = matrix.row(1).unwrap();
        assert_eq!(first.dtype, TensorDType::Q2B64);
        assert_eq!(first.weights, q2);
        assert_eq!(second.dtype, TensorDType::Q4B64);
        assert_eq!(second.weights, q4);
        assert!(matrix.row(2).is_err());
    }
}
