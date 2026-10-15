//! Validate calibrated head scales using the exact native projection and packed weights.
use anyhow::ensure;
use ctox_qwen38_27b::{backend::cuda_runtime::CudaCandidateRuntime,loader::{ChecksumPolicy,FloatTensorView,ModelArtifact,RecoveredMatrixView},backend::Activation};
use std::{fs,io::{Read,Write}};
#[allow(dead_code)] #[path="../sampling_tv/student.rs"] mod sampling;
fn main()->anyhow::Result<()> {
    let a:Vec<String>=std::env::args().collect();ensure!(a.len()==6,"head ARTIFACT CUBIN HIDDEN_FILE SCALE_PREFIX OUTPUT");
    let runtime=CudaCandidateRuntime::new(&fs::read(&a[2])?,0)?;
    let artifact=ModelArtifact::open(&a[1],ChecksumPolicy::AllTensors)?;
    let old=artifact.recovered_matrix("lm_head.weight")?;
    let si=fs::read(format!("{}.s_in.f16",a[4]))?;let so=fs::read(format!("{}.s_out.f16",a[4]))?;
    ensure!(si.len()==old.matrix.columns*2 && so.len()==old.matrix.rows*2,"scale size mismatch");
    let recovered=RecoveredMatrixView{matrix:old.matrix,s_in:FloatTensorView::F16Le(&si),s_out:FloatTensorView::F16Le(&so)};
    let zero=vec![0.;old.matrix.columns];let op=recovered.operation(&zero,Activation::Identity)?;
    let projection=runtime.prepare_shared_a8_projection(&op)?;
    let mut source=fs::File::open(&a[3])?;let bytes=source.metadata()?.len() as usize;let size=old.matrix.columns*4;
    ensure!(bytes%size==0,"hidden file shape");
    let mut output=std::io::BufWriter::new(fs::OpenOptions::new().create_new(true).write(true).open(&a[5])?);
    let mut buf=vec![0u8;size];
    for row in 0..bytes/size {
        source.read_exact(&mut buf)?;
        let h:Vec<f32>=buf.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        let op=recovered.operation(&h,Activation::Identity)?;let activation=runtime.prepare_shared_a8_activation(&op)?;
        let values=runtime.dispatch_shared_a8_fanout(&activation,&[&projection])?;
        serde_json::to_writer(&mut output,&serde_json::json!({"row":row,"distribution":sampling::sample_distribution(&values[0])?}))?;
        output.write_all(b"\n")?;
    }
    Ok(())
}
