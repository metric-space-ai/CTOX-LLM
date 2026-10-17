//! Export native residual boundaries and FFN terms; no kernel or weight changes.
use anyhow::ensure;
use ctox_qwen38_27b::{backend::{cuda_graph::PreparedCudaProjectionGraph,cuda_runtime::CudaCandidateRuntime},loader::{ChecksumPolicy,ModelArtifact},Qwen38Config};
use serde::{Deserialize,Serialize};
use std::{fs,io::{self,BufRead,Write}};
#[allow(dead_code)] #[path="../sampling_tv/student.rs"] mod sampling;
#[derive(Deserialize)] struct Request { id:String, token_ids:Vec<u32>, positions:Vec<usize> }
#[derive(Serialize)] struct Row { id:String, position:usize, row:usize, distribution:Vec<(usize,f64)> }
fn write_row(file:&mut impl Write,x:Vec<f32>)->anyhow::Result<()> {
    ensure!(x.len()==5120 && x.iter().all(|v|v.is_finite()),"invalid state");
    for v in x {file.write_all(&v.to_le_bytes())?;} Ok(())
}
fn main()->anyhow::Result<()> {
    let a:Vec<String>=std::env::args().collect();ensure!(a.len()==4,"block-student ARTIFACT CUBIN OUTPUT_PREFIX");
    let runtime=CudaCandidateRuntime::new(&fs::read(&a[2])?,0)?;
    let artifact=ModelArtifact::open(&a[1],ChecksumPolicy::AllTensors)?;
    let config=Qwen38Config::default();
    let mut graph=PreparedCudaProjectionGraph::prepare_for_calibration(&runtime,&artifact,&config,2048)?;
    let create=|suffix:&str|->anyhow::Result<io::BufWriter<fs::File>>{Ok(io::BufWriter::new(fs::OpenOptions::new().write(true).create_new(true).open(format!("{}{}",a[3],suffix))?))};
    let mut hidden=create(".f32")?;let mut terms=create("-ffn.f32")?;let mut out=create(".jsonl")?;let mut row=0;
    for line in io::stdin().lock().lines() {
        let s:Request=serde_json::from_str(&line?)?;
        ensure!(!s.positions.is_empty() && s.token_ids.len()<=2048 && s.positions.windows(2).all(|w|w[0]<w[1]) && s.positions.last().unwrap()+1<s.token_ids.len(),"invalid positions");
        graph.reset_session()?;let mut next=0;
        for (position,&token) in s.token_ids[..=*s.positions.last().unwrap()].iter().enumerate() {
            let view=graph.dispatch_target_token_device(&runtime,&config,token as usize,position)?;
            if position==s.positions[next] {
                let logits=runtime.verifier_read_f32_device(view)?;
                write_row(&mut hidden,runtime.verifier_read_f32_device(graph.embedding().device_output()?)?)?;
                for layer in 0..64 {
                    let key=if layer==63 {"target:63:post_ffn:final".to_string()} else {format!("target:{layer}:post_ffn:layer_{}",layer+1)};
                    let norm=graph.norms().residual(&key)?;
                    let h=if layer==63 {norm.normalized_output()?} else {norm.residual_output()?};
                    write_row(&mut hidden,runtime.verifier_read_f32_device(h)?)?;
                    let before=graph.norms().residual(&format!("target:{layer}:post_attention"))?.residual_output()?;
                    let down=graph.projection(&format!("model.language_model.layers.{layer}.mlp.down_proj.weight"))?.device_output()?;
                    write_row(&mut terms,runtime.verifier_read_f32_device(before)?)?;
                    write_row(&mut terms,runtime.verifier_read_f32_device(down)?)?;
                }
                serde_json::to_writer(&mut out,&Row{id:s.id.clone(),position,row,distribution:sampling::sample_distribution(&logits)?})?;out.write_all(b"\n")?;
                row+=1;next+=1;if next==s.positions.len(){break}
            }
        }
        hidden.flush()?;terms.flush()?;out.flush()?;eprintln!("block-student {} {} states",s.id,s.positions.len());
    }
    Ok(())
}
