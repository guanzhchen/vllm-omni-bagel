# BAGEL Force-Interleaved vLLM Omni Evaluation

This branch contains the BAGEL force-interleaved two-stage path used to compare
vLLM Omni against the local VLMEvalKit BAGEL force-interleaved reference.

The goal was not only to make vLLM Omni faster. The force-interleaved eval is
accuracy-sensitive, so the vLLM Omni path was first aligned with the local BAGEL
behavior on a judged 32-sample probe, then measured on a 100-sample speed-only
run.

## Branch and Version Context

- Fork repo: `guanzhchen/vllm-omni-bagel`
- Fork base used for the PR branch: `b6f29ee6`
- Force-interleaved feature commit: `a704d8f6`
- Dev branch package seen during the 2026-05-31 fair gates:
  `vllm-omni 0.1.dev1594+ga704d8f64.d20260531`
- vLLM package seen during the same gates: `vllm 0.20.0`

## What Changed

The vLLM Omni implementation was changed to preserve local BAGEL
force-interleaved semantics while keeping the serving path efficient:

- Added BAGEL reference-style VQA prompt layout helpers, image span tracking,
  and logical RoPE handling.
- Added img2img VAE plus ViT preprocessing support for the generated image
  feedback path.
- Added non-causal image-block recompute hooks so image blocks update cache
  state like local BAGEL.
- Added Stage 0 to Stage 1 continuation through transferred KV for the
  think -> image -> answer force-interleaved flow.
- Added CFG branch handling, including optional reuse of the main prompt prefix
  for CFG text conditioning.
- Added deterministic per-item seeding with `FORCE_SEED_BY_INDEX=1`, so async
  request ordering in vLLM Omni does not change stochastic generations.
- Added request-mode diffusion batching and StagePool least-loaded replica
  routing for throughput.

## Why Accuracy Was Previously Off

The earlier vLLM Omni path was fast, but it differed from local BAGEL in several
places that matter for the force-interleaved judge:

- Image feedback could miss the same VAE plus ViT conditioning used by local
  BAGEL.
- Image-block cache behavior differed because local BAGEL performs non-causal
  image updates.
- CFG text conditioning could use a different prefix from the main branch.
- Async serving changed request/completion order. If sampling used a global RNG,
  the same samples consumed random numbers in a different order and produced
  different outputs.
- Stage 1 text batching changed think/answer generation behavior relative to
  the local reference path.
- Output formatting had to match the local judge payload.

The final aligned configs kept the speed-oriented vLLM Omni two-stage serving
path. The conservative local-equivalent path keeps Stage 1 text per request;
the fastest accuracy-aligned gate uses batched Stage 1 text only when decoded
think text is re-encoded before image generation.

## Current MMVet Accuracy and Inference Time

The headline accuracy comparison should be read as the current MMVet judged
performance, with the measured inference-only wall time kept alongside it. The
older 2026-05-28 speed notes are historical because the local timing was not
recorded in the same artifact as the vLLM Omni timing.

Reference local artifact:

```text
bagel_local_serve/outputs/eval100_omni_rep4batch4_balanced_fair20_gate32_20260531_184043/bagel_local
```

Reference local result:

| System | Score | Valid | Inference Time | Throughput |
|---|---:|---:|---:|---:|
| BAGEL local force-interleaved, 4-rank torchrun | 69.062 | 32/32 | 302.109s | 0.106 samples/s |

vLLM Omni gates tried against that same local baseline:

| Candidate | Key knobs | Score | Inference Time | Speed ratio vLLM/local | Status |
|---|---|---:|---:|---:|---|
| Two-stage rep4 batch4 | `FORCE_STAGE1_BATCH_TEXT=0`, `max_num_seqs=4`, `batch_balanced` | 64.38 | 323.319s | 1.07 | Accuracy gap and slower |
| Two-stage rep4 batch8 | `max_num_seqs=8`, `request_batch_wait_ms=60000`, dataset order | 73.12 | 353.585s | 1.17 | Accuracy aligned, slower |
| Single-stage rep4 | `OMNI_BACKEND=vllm_force`, `bagel_force_single_stage_rep4.yaml` | 19.06 | 310.45s | 1.028 | Accuracy collapse |
| Two-stage rep4 batch4 batched text | `FORCE_STAGE1_BATCH_TEXT=1` | 62.5 | 306.752s | 1.015 | Accuracy gap and not faster |
| Two-stage rep4 batch8 batched text + reencode | `FORCE_STAGE1_BATCH_TEXT=1`, `FORCE_STAGE1_REENCODE_TEXT=1` | 71.88 | 325.874s | 1.079 | Accuracy aligned, slower |

No 2026-05-31 candidate met the `>3x` target. The fastest accuracy-aligned run
was the two-stage rep4 batch8 batched-text + reencode gate, but it was still
slower than the BAGEL local baseline. The closest speed runs were
approximately equal to local and had large accuracy drops.

The current bottleneck is inside Stage 1, not the HTTP driver. On the fastest
accuracy-aligned gate, each Stage 1 replica spent about 77-85s on batched think,
149-168s on the image batch, and 15-18s on batched answer. Since the
BAGEL local 32-sample baseline is 302.109s, a `>3x` vLLM result would need
to finish below about 100.7s; the Stage 1 image batch alone already exceeds that
limit. Hitting `>3x` therefore requires a materially different Stage 1 image
kernel/topology, not just higher request concurrency.

Representative artifacts:

```text
bagel_local_serve/outputs/eval100_omni_rep4batch4_balanced_fair20_gate32_20260531_184043
bagel_local_serve/outputs/eval100_omni_rep4batch8_wait60_fair20_gate32_20260531_192132
bagel_local_serve/outputs/eval100_omni_single_rep4_fair20_gate32_20260531_194012
bagel_local_serve/outputs/eval100_omni_rep4batch4_batchtext_fair20_gate32_20260531_195246
bagel_local_serve/outputs/eval100_omni_rep4batch8_batchtext_reencode_fair20_gate32_20260531_201140
```

## Current Eval Config

The fastest accuracy-aligned two-stage gate used:

```bash
N_SAMPLES=32
CONCURRENCY=96
MLAUNCH_GPUS=4
OMNI_BACKEND=vllm_force_twostage
FORCE_TWOSTAGE_DEPLOY_CFG=/mnt/moonfs/chenguanzheng-m4/Projects/ThinkMorph-Pro/vllm_omni_serving/bagel_force_twostage_stage1rep4_batch8.yaml
FORCE_TWOSTAGE_DIFFUSION_BATCH_SIZE=8
FORCE_TWOSTAGE_SCHEDULE_ORDER=dataset
FORCE_TWOSTAGE_SCHEDULE_REPLICAS=4
```

Accuracy-sensitive settings:

```bash
BAGEL_VQA_REFERENCE_PREFILL=1
BAGEL_VQA_REFERENCE_LAYOUT=1
BAGEL_VQA_LOGICAL_ROPE=1
BAGEL_FORCE_IMG2IMG_VIT=1
BAGEL_CFG_TEXT_FROM_MAIN_PREFIX=1
BAGEL_IMG2IMG_VIT_SEPARATOR=0
BAGEL_IMG2IMG_NONCAUSAL_RECOMPUTE=1
FORCE_SEED=0
FORCE_SEED_BY_INDEX=1
FORCE_DO_SAMPLE=1
FORCE_STAGE1_BATCH_TEXT=1
FORCE_STAGE1_REENCODE_TEXT=1
```

For a more conservative local-equivalent diagnostic, set
`FORCE_STAGE1_BATCH_TEXT=0` and `FORCE_STAGE1_REENCODE_TEXT=0`. That path keeps
think/answer generation per request and matched accuracy in the batch8 gate, but
was slower at 353.585s for the same 32 samples.

The key point is `FORCE_SEED_BY_INDEX=1`. vLLM Omni is async, so request order
and completion order can differ from the local reference. Seeding by MMVet item
index makes each item deterministic independent of scheduling order.

Stage 0 uses vLLM CUDA graph capture for the AR understanding prefill. Stage 1
is intentionally eager (`enforce_eager: true`): compiling the BAGEL DiT path on
2026-05-31 corrupted outputs and was slower (`~72.7 -> 51.9` MMVet, `23.3` vs
`16.6` s/sample in the compile gate). The current code keeps Stage 1 eager.

## How to Re-run

The common two-stage runner is:

```text
bagel_local_serve/mlaunch_eval100_force_twostage.sh
```

A current 32-sample judged gate uses:

```bash
N_SAMPLES=32 \
MLAUNCH_GPUS=4 \
CONCURRENCY=96 \
JUDGE=gpt-5.4 \
JUDGE_NPROC=12 \
JUDGE_REASONING_EFFORT=medium \
OMNI_BACKEND=vllm_force_twostage \
FORCE_TWOSTAGE_DEPLOY_CFG=/mnt/moonfs/chenguanzheng-m4/Projects/ThinkMorph-Pro/vllm_omni_serving/bagel_force_twostage_stage1rep4_batch8.yaml \
FORCE_TWOSTAGE_DIFFUSION_BATCH_SIZE=8 \
FORCE_TWOSTAGE_SCHEDULE_ORDER=dataset \
FORCE_TWOSTAGE_SCHEDULE_REPLICAS=4 \
BAGEL_CFG_TEXT_FROM_MAIN_PREFIX=1 \
FORCE_STAGE1_BATCH_TEXT=1 \
FORCE_STAGE1_REENCODE_TEXT=1 \
FORCE_SEED=0 \
FORCE_SEED_BY_INDEX=1 \
FORCE_DO_SAMPLE=1 \
BAGEL_FORCE_IMG2IMG_VIT=1 \
BAGEL_IMG2IMG_VIT_SEPARATOR=0 \
BAGEL_IMG2IMG_NONCAUSAL_RECOMPUTE=1 \
BAGEL_LOCAL_MODEL=bagel_local_force_interleaved \
bash bagel_local_serve/mlaunch_eval100_force_twostage.sh
```

Use `BAGEL_LOCAL_REUSE_DIR=/path/to/.../bagel_local` when only re-gating a new
vLLM Omni topology against an already judged local baseline. Do not scale a
candidate to 100 samples unless the 32-sample gate is both accuracy-aligned and
faster on inference-only wall time.

