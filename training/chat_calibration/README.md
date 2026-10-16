# Complete-chat calibration experiments

These tools investigate whether existing Q2/Q4 correction scales can preserve
Q8 output distributions better when calibration retains real conversation
context. They do not declare an accepted model or a TV threshold achieved.

`prepare.py` selects complete conversations from an existing licensed recovery
corpus, preserving the official chat template and tool schemas. Missing user
questions, undeclared tools, absent schemas and oversized chats are rejected.
Source identity, full conversation and first-user-message hashes separate
training, tuning and held-out records. The initial bounded screen balances
chats with tool calls and general chats, and accepts at most 2048 tokens per
chat. This selection is not a representative long-context benchmark.

Assistant target positions are shifted by one: the last input token predicts
the first assistant token. User/system messages, tool results, role headers and
the automatically inserted empty reasoning prefix are excluded from the loss.
Selected assistant positions are uniform and independent of model outputs.
The complete preceding context is always evaluated. The tool-focused refinement
adds up to 16 actual tool-call positions per training conversation, because
uniform positions can otherwise be dominated by reasoning text. Tuning and
held-out positions are unchanged. Captures are merged only with identical
full contexts, runtime bindings and disjoint positions.

`capture.py` runs the pinned Q8 teacher on the source continuations (teacher
forcing). Official HF token IDs are given to both engines. Every complete
conversation must decode byte-for-byte through the GGUF vocabulary; native GGUF
tokenization differences are recorded instead of dropping multilingual cases.
Maximum-centered FP16 logits are a disposable, bounded training cache. Exact
post-sampler reference distributions remain in compact JSONL.

`student.rs` replays those same IDs through the native Q2/Q4 graph and records
selected normalized head inputs and distributions. `fit.py` is an **initial
head-only calibration screen**: only existing LM-head input/output scales can
change. Packed codes, body scales and model size stay fixed. The differentiable
surrogate applies FP16 scale rounding and A8 activation rounding; it must first
match native baseline distributions. Full-vocabulary Q8 KL supplies gradients;
only tuning TV selects a checkpoint. An optional, independently checked TV loss
adds gradients for the exact fixed sampler within its selected support. The
predeclared refinement compares KL-only and KL+TV (weight 1), choosing only by
tuning TV before native and original-benchmark validation. A straight-through derivative is used for
rounding. This is scale fitting, not full-weight fine-tuning.

`head.rs` verifies the selected scales with the actual native packed projection
and corrected CUDA kernel. `verify.py` reports all held-out positions, including
regressions. `legacy_check.py` additionally retains the original sampled-Q8
benchmark, without adapting the sampler or selecting easy reference positions.
Sampling remains temperature 1, top-k 20 followed by top-p .95. A new result on
teacher-forced source chats is not interchangeable with that sampled-generation
benchmark or with practical answer correctness. MTP is not calibrated here.

Build `build.sh ENGINE_SNAPSHOT OUTPUT` through the shared Linux build lane,
using a pinned engine snapshot and a task-specific `CARGO_TARGET_DIR`. The
explicit GPU1 runner uses the existing admitted-container/native-CUDA handoff,
a Qwen runtime lock, process-group deadlines, and separate teacher/student
processes. Hardware paths in that runner describe the existing research host;
these are not product runtime dependencies. Do not run it outside admission.
No weights, dense target caches or conversation contents belong in Git.

Contract verification: run `check_contracts.py` through the admitted lane. It
also mutates the assistant-role check and requires the masking contract to fail.

`head_diagnosis.py` asks whether substituting a higher-precision output projection
can recover the Q8 distributions while retaining the same native body states.
It evaluates training and tuning positions only, with Q8 head weights and either
FP32 inputs or native-style A8 rounding. It removes learned Q2/Q4 head corrections.
This diagnostic is neither a native Q8 replay nor an additive decomposition of
quantization error, and does not produce a deployable mixed-precision model.
