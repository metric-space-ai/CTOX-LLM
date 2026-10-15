//! Complete-context native replay with selected assistant-only hidden states.
use anyhow::ensure;
use ctox_qwen38_27b::{backend::{cuda_graph::PreparedCudaProjectionGraph,cuda_runtime::CudaCandidateRuntime},loader::{ChecksumPolicy,ModelArtifact},Qwen38Config};
use serde::{Deserialize,Serialize};
use std::{fs,io::{self,BufRead,Write}};
#[allow(dead_code)] #[path="../sampling_tv/student.rs"] mod sampling;
#[derive(Deserialize)] struct Request { id:String, token_ids:Vec<u32>, positions:Vec<usize> }
#[derive(Serialize)] struct Row { id:String, position:usize, row:usize, distribution:Vec<(usize,f64)> }
fn main()->anyhow::Result<()> {
    let a:Vec<String>=std::env::args().collect();ensure!(a.len()==4,"student ARTIFACT CUBIN OUTPUT_PREFIX");
    let runtime=CudaCandidateRuntime::new(&fs::read(&a[2])?,0)?;
    let artifact=ModelArtifact::open(&a[1],ChecksumPolicy::AllTensors)?;
    let config=Qwen38Config::default();
    let mut graph=PreparedCudaProjectionGraph::prepare_for_calibration(&runtime,&artifact,&config,2048)?;
    let mut hidden=io::BufWriter::new(fs::OpenOptions::new().write(true).create_new(true).open(format!("{}.f32",a[3]))?);
    let mut out=io::BufWriter::new(fs::OpenOptions::new().write(true).create_new(true).open(format!("{}.jsonl",a[3]))?);
    let mut row=0;
    for line in io::stdin().lock().lines() {
        let s:Request=serde_json::from_str(&line?)?;
        ensure!(!s.positions.is_empty() && s.token_ids.len()<=2048 && s.positions.windows(2).all(|w|w[0]<w[1]) && s.positions.last().unwrap()+1<s.token_ids.len(),"invalid positions");
        graph.reset_session()?;let mut next=0;
        for (position,&token) in s.token_ids[..=*s.positions.last().unwrap()].iter().enumerate() {
            let view=graph.dispatch_target_token_device(&runtime,&config,token as usize,position)?;
            if position==s.positions[next] {
                let logits=runtime.verifier_read_f32_device(view)?;
                let h=runtime.verifier_read_f32_device(graph.norms().residual("target:63:post_ffn:final")?.normalized_output()?)?;
                ensure!(h.len()==5120 && h.iter().all(|x|x.is_finite()),"bad head input");
                for x in h {hidden.write_all(&x.to_le_bytes())?;}
                serde_json::to_writer(&mut out,&Row{id:s.id.clone(),position,row,distribution:sampling::sample_distribution(&logits)?})?;out.write_all(b"\n")?;
                row+=1;next+=1;if next==s.positions.len(){break}
            }
        }
        hidden.flush()?;out.flush()?;eprintln!("student {} {} positions",s.id,s.positions.len());
    }
    Ok(())
}
