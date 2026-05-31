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
- Installed package seen during testing: `vllm-omni 0.1.dev1593+gb6f29ee61.d20260513`
- A separate `vllm` package was not installed in the default shell used for
  validation.

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

The final aligned config kept the speed-oriented vLLM Omni two-stage serving
path, but disabled the text-batching choices that changed the judged outputs.

## Accuracy Alignment Gate

The accuracy gate was a judged 32-sample MMVet probe with the same 20-step force
configuration used for the later speed run.

Artifact:

```text
bagel_local_serve/outputs/eval100_eval32_steps20_cfgprefix_nobatchtext_reusefix_20260528_154426
```

Result:

| System | Score | Valid | Inference Time |
|---|---:|---:|---:|
| vLLM Omni two-stage force, 20 steps | 71.88 | 32/32 | 504.419s |
| BAGEL local force-interleaved | 59.375 | 32/32 | 1071.252s |

The local run reused this reference output directory:

```text
bagel_local_serve/outputs/eval100_eval32_batchtext_noreencode_20260528_103927/bagel_local
```

The comparison markdown is:

```text
bagel_local_serve/outputs/eval100_eval32_steps20_cfgprefix_nobatchtext_reusefix_20260528_154426/comparison/comparison.md
```

## Speed Comparison

After the judged 32-sample gate, speed was measured on 100 samples with judge
disabled.

Artifact:

```text
bagel_local_serve/outputs/eval100_speed100_steps20_cfgprefix_nobatchtext_20260528_155916
```

Result:

| System | Samples | Valid | Generated Images | Inference Time | Throughput |
|---|---:|---:|---:|---:|---:|
| vLLM Omni two-stage force, 20 steps | 100 | 100 | 100 | 1489.254s | 0.067 samples/s |
| BAGEL local force-interleaved | 100 | 100 | - | 3242.676s | 0.03084 samples/s |

Inference-only speedup:

```text
3242.676 / 1489.254 = 2.177x
```

Including vLLM engine initialization, total vLLM time was `1693.357s`, still
`1.915x` versus local inference-only time.

The comparison markdown is:

```text
bagel_local_serve/outputs/eval100_speed100_steps20_cfgprefix_nobatchtext_20260528_155916/comparison/comparison.md
```

## Final Eval Config

The final accuracy-aligned speed run used:

```bash
N_SAMPLES=100
CONCURRENCY=64
SKIP_JUDGE=1
MLAUNCH_GPUS=2
OMNI_BACKEND=vllm_force_twostage
FORCE_TWOSTAGE_DEPLOY_CFG_OVERRIDE=/mnt/moonfs/chenguanzheng-m4/Projects/ThinkMorph-Pro/vllm_omni_serving/bagel_force_twostage_dualgpu_stage1rep2.yaml
FORCE_TWOSTAGE_DIFFUSION_BATCH_SIZE=4
FORCE_NUM_TIMESTEPS=20
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
FORCE_STAGE1_BATCH_TEXT=0
FORCE_STAGE1_REENCODE_TEXT=0
```

The key point is `FORCE_SEED_BY_INDEX=1`. vLLM Omni is async, so request order
and completion order can differ from the local reference. Seeding by MMVet item
index makes each item deterministic independent of scheduling order.

Stage 1 CUDA graph was not enabled in the measured config because
`enforce_eager: true` was kept in the Stage 1 deployment config. The measured
speedup came from two-stage KV reuse, Stage 1 replica scaling, request
concurrency, image batching, and the 20-step diffusion config.

## How to Re-run

The historical launch scripts generated for the two final runs are:

```text
bagel_local_serve/logs/eval100_eval32_steps20_cfgprefix_nobatchtext_reusefix_20260528_154426_remote.sh
bagel_local_serve/logs/eval100_speed100_steps20_cfgprefix_nobatchtext_20260528_155916_remote.sh
```

The common runner is:

```text
bagel_local_serve/mlaunch_eval100_force_twostage.sh
```

A typical 32-sample judged alignment run uses:

```bash
N_SAMPLES=32 \
SKIP_JUDGE=0 \
CONCURRENCY=64 \
OMNI_BACKEND=vllm_force_twostage \
FORCE_NUM_TIMESTEPS=20 \
FORCE_TWOSTAGE_DIFFUSION_BATCH_SIZE=4 \
BAGEL_CFG_TEXT_FROM_MAIN_PREFIX=1 \
FORCE_STAGE1_BATCH_TEXT=0 \
FORCE_STAGE1_REENCODE_TEXT=0 \
FORCE_SEED=0 \
FORCE_SEED_BY_INDEX=1 \
BAGEL_FORCE_IMG2IMG_VIT=1 \
BAGEL_IMG2IMG_VIT_SEPARATOR=0 \
BAGEL_IMG2IMG_NONCAUSAL_RECOMPUTE=1 \
bash bagel_local_serve/mlaunch_eval100_force_twostage.sh
```

A typical 100-sample speed-only run changes only the sample count and judge flag:

```bash
N_SAMPLES=100 \
SKIP_JUDGE=1 \
CONCURRENCY=64 \
OMNI_BACKEND=vllm_force_twostage \
FORCE_NUM_TIMESTEPS=20 \
FORCE_TWOSTAGE_DIFFUSION_BATCH_SIZE=4 \
BAGEL_CFG_TEXT_FROM_MAIN_PREFIX=1 \
FORCE_STAGE1_BATCH_TEXT=0 \
FORCE_STAGE1_REENCODE_TEXT=0 \
FORCE_SEED=0 \
FORCE_SEED_BY_INDEX=1 \
BAGEL_FORCE_IMG2IMG_VIT=1 \
BAGEL_IMG2IMG_VIT_SEPARATOR=0 \
BAGEL_IMG2IMG_NONCAUSAL_RECOMPUTE=1 \
bash bagel_local_serve/mlaunch_eval100_force_twostage.sh
```

## Code Validation

Checks that passed on the force-interleaved patch:

```bash
git diff --check
python -m py_compile \
  vllm_omni/diffusion/data.py \
  vllm_omni/diffusion/diffusion_engine.py \
  vllm_omni/diffusion/executor/multiproc_executor.py \
  vllm_omni/diffusion/models/bagel/bagel_transformer.py \
  vllm_omni/diffusion/models/bagel/pipeline_bagel.py \
  vllm_omni/diffusion/request.py \
  vllm_omni/diffusion/sched/base_scheduler.py \
  vllm_omni/diffusion/worker/diffusion_model_runner.py \
  vllm_omni/diffusion/worker/diffusion_worker.py \
  vllm_omni/engine/async_omni_engine.py \
  vllm_omni/engine/stage_pool.py \
  vllm_omni/entrypoints/openai/serving_chat.py \
  vllm_omni/model_executor/models/bagel/bagel.py \
  vllm_omni/model_executor/stage_input_processors/bagel.py \
  tests/diffusion/test_diffusion_batch_wait.py \
  tests/engine/test_stage_pool.py \
  tests/model_executor/test_bagel_vqa_reference_layout.py \
  vllm_omni/utils/bagel_vqa.py
```

Unit tests were attempted with:

```bash
pytest -q tests/diffusion/test_diffusion_batch_wait.py \
  tests/engine/test_stage_pool.py \
  tests/model_executor/test_bagel_vqa_reference_layout.py
```

In the default shell environment, collection failed before reaching these tests
because dependency `aenum` was missing.
