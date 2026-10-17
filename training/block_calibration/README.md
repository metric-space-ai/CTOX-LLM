# Complete-chat block calibration

This experiment localizes Q8/native residual drift and fits existing FFN output
channel scales. It does not retrain weights, change quantized codes, enlarge the
model format, or establish the below-5% TV goal by itself.

`prepare.py` selects eight training and four tuning conversations from the
existing disjoint complete-chat corpus. It retains each full preceding context,
but scores only assistant-generation positions. Eight uniformly spaced answer
positions and up to eight actual tool-call positions are selected per chat.
Prompt, tool-result, role-header, and forced empty-thinking tokens are excluded.
Actual tool-call TV is reported separately from the broader tool-enabled bucket.

`capture.py` exports Q8 raw inputs to blocks 0–63 and final normalized hidden
states. `student.rs` exports matching native states plus each FFN's residual and
down-projection output. Their sum must reproduce the native next-block input.
The final normalized boundary is diagnostic only: it is not a linear fit target.

`analyze.py` fits training rows only, using per-channel ridge regression for the
existing down-projection `s_out`. Gains are bounded to ±10%, and stored FP16
scales to [0.25, 4]. At most four blocks are selected by training improvement;
each must also improve local tuning residual error. These local errors are
not an additive decomposition of output TV and do not prove better answers.

`replay.py` applies the selected scales at the fixed strengths 0.25, 0.5, and 1.0
to a sealed RAM-backed artifact. Tensor offsets and quantized payloads stay
unchanged. Tensor checksums are updated and checked by the native loader.
The artifact disappears when the bounded replay process finishes; no extra
8 GB model copy is left on disk.

`verify.py` compares the native post-sampler distributions at identical prefixes.
The sampler remains temperature 1, top-k 20, top-p 0.95. TV is half the L1 distance;
it is not an answer-error rate. Tuning results never mark the goal achieved.
Any selected improvement still requires the untouched complete-chat holdout and
the original Q8-generated-answer cohort before promotion. These cohorts measure
different prefix populations and must be reported separately.

Build the bridge/student through the admitted Linux build lane with `build.sh`.
`run_gpu1.sh` expects the pinned teacher, original pack, corrected cubin, runtime
binaries, and private input chats. Use the existing paired lane admission and
bounded native supervisor; never launch it alongside an unowned GPU workload.
`check_contracts.py` also verifies a mutation: omitting the residual subtraction
must break known-scale recovery. The native run is the integration check.

`prepare_checkpoint.py` recovers a completed historical fit only when its scale
metadata binds the fit report, original artifact and pinned Q8 teacher. All
non-scale tensor identities must match the current artifact, and evaluation
conversation identities must be absent from historical calibration. It exports
two fixed variants: the 64 main blocks only, and all saved scales. These are
historical short-window fits, not new training on complete chats.

The default replay scope remains FFN output scales. `--scale-scope recovery`
explicitly permits existing FP16 input/output scales; tensor shapes, bounds and
all quantized payloads stay fixed. `checkpoint_evaluate.py` selects between the
two variants on full-chat tuning and then uses the original generation guard.
It does not promote a model or replace independent holdout acceptance.
