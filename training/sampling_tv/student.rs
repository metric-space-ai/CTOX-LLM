//! Replay generated Q8 prefixes, emitting only post-sampler distributions.
use anyhow::{ensure, Context};
use ctox_qwen38_27b::{backend::{cuda_graph::PreparedCudaProjectionGraph, cuda_runtime::CudaCandidateRuntime},loader::{ChecksumPolicy,ModelArtifact},Qwen38Config};
use serde::{Deserialize,Serialize};
use std::{fs,io::{self,BufRead,Write}};
#[derive(Deserialize)]
struct Request { id:String, token_ids:Vec<u32>, prompt_tokens:usize, generated_tokens:usize, vocab_size:usize }
#[derive(Serialize)]
struct Row { id:String, offset:usize, distribution:Vec<(usize,f64)> }
fn sample_distribution(logits:&[f32]) -> anyhow::Result<Vec<(usize,f64)>> {
    ensure!(logits.len() >= 20 && logits.iter().all(|x|x.is_finite()), "invalid logits");
    let mut ids:Vec<usize>=(0..logits.len()).collect();
    // Fixed measured profile: temperature=1, top-k=20, then top-p=.95.
    ids.select_nth_unstable_by(19, |a,b| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
    ids.truncate(20);
    ids.sort_by(|a,b|logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
    let max=logits[ids[0]] as f64;
    let weights:Vec<f64>=ids.iter().map(|i|(logits[*i] as f64-max).exp()).collect();
    let z:f64=weights.iter().sum(); let mut mass=0.; let mut n=0;
    for w in &weights {mass+=w/z; n+=1; if mass>=0.95 {break}}
    let z:f64=weights[..n].iter().sum();
    Ok(ids[..n].iter().zip(&weights).map(|(i,w)|(*i,w/z)).collect())
}
fn main()->anyhow::Result<()> {
    let a:Vec<String>=std::env::args().collect(); ensure!(a.len()==3,"usage: student ARTIFACT CUBIN < requests.jsonl");
    let runtime=CudaCandidateRuntime::new(&fs::read(&a[2])?,0)?;
    let artifact=ModelArtifact::open(&a[1],ChecksumPolicy::AllTensors)?;
    let config=Qwen38Config::default();
    let mut graph=PreparedCudaProjectionGraph::prepare_for_calibration(&runtime,&artifact,&config,2048)?;
    let mut out=io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let s:Request=serde_json::from_str(&line?)?;
        ensure!(s.prompt_tokens>0 && s.generated_tokens>0 && s.token_ids.len()==s.prompt_tokens+s.generated_tokens && s.token_ids.len()<=2048,"invalid generation boundaries");
        graph.reset_session()?;
        // Feeding token at prefix_len-1 predicts the FIRST generated token.
        // Never score prompt processing or a prediction after the last output.
        for (position,&token) in s.token_ids[..s.token_ids.len()-1].iter().enumerate() {
            let view=graph.dispatch_target_token_device(&runtime,&config,token as usize,position)?;
            if position+1>=s.prompt_tokens {
                let logits=runtime.verifier_read_f32_device(view)?;
                ensure!(logits.len()==s.vocab_size,"vocabulary mismatch");
                let row=Row{id:s.id.clone(),offset:position+1-s.prompt_tokens,distribution:sample_distribution(&logits)?};
                serde_json::to_writer(&mut out,&row).context("writing sampling row")?; out.write_all(b"\n")?; out.flush()?;
            }
        }
        eprintln!("completed {}: {} generated positions",s.id,s.generated_tokens);
    }
    Ok(())
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn sampler_contract() {
        let mut x=vec![-1000.;24]; x[0]=0.51_f32.ln(); x[1]=0.49_f32.ln();
        let p=sample_distribution(&x).unwrap(); assert_eq!(p.len(),2); assert!((p[0].1-0.51).abs()<1e-7);
        x[0]=0.96_f32.ln(); x[1]=0.04_f32.ln(); assert_eq!(sample_distribution(&x).unwrap(),vec![(0,1.)]);
        assert!(sample_distribution(&[f32::NAN;24]).is_err());
    }
}
