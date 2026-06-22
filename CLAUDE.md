# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Manifests and patches for deploying a NVIDIA NIM inference service (`llama-3.1-nemotron-nano-4b-v1.1`) on RHOAI 3.4 with a T4 GPU (16 GB VRAM) in the `nvidia-nim` namespace on cluster `zenek-hqxqx` (us-east-2).

## Cluster context

- 3 GPU worker nodes (T4), each in a different AZ — no non-GPU worker nodes
- The NIM PVC is AZ-pinned to `us-east-2b` → the predictor pod can only schedule on `ip-10-0-59-146`
- GPU nodes are CPU-constrained from operator workloads; `kserve-container` CPU request is intentionally `200m` (limit `1`) — do not raise it without checking available headroom first

## Apply order for a fresh deployment

```bash
oc apply -f manifests/00-namespace.yaml
# create secrets (see README Step 2 — do not apply 01-ngc-secrets.yaml with real keys)
oc apply -f manifests/02-pvc.yaml
oc apply -f manifests/03-servingruntime.yaml
oc apply -f manifests/04-inferenceservice.yaml
```

## Patch files (for existing UI-created InferenceServices)

| File | When to use |
|---|---|
| `fix-t4-patch.yaml` | UI-deployed ISVC missing T4 env vars |
| `fix-playground.yaml` | ISVC not visible in RHOAI Playground/GenAI Studio |
| `fix-cpu-request.yaml` | ISVC stuck Pending due to CPU pressure on GPU nodes |

Apply pattern:
```bash
oc patch inferenceservice llama-31-nemotron-nano-4b -n nvidia-nim \
  --type=merge --patch-file manifests/<fix-file>.yaml
```

## Key design decisions

**`NIM_MODEL_PROFILE`**: Forces the vLLM backend profile. T4 auto-selects `tensorrt_llm-trtllm_buildable`, which compiles CUDA kernels on first start (~10 min). The KServe readiness probe times out and kills the container first, looping forever and wiping the build cache on each restart. The vLLM profile starts in ~2 min with no build step.

**`NIM_MAX_MODEL_LEN: 16384`**: T4 has 16 GB VRAM; weights use 8.44 GB (FP16 — T4 doesn't support BF16), leaving ~6 GB for KV cache (~27K tokens max). The model's default context of 131K tokens exceeds this and vLLM refuses to start. 16K fits comfortably.

**`NIM_SERVED_MODEL_NAME: llama-31-nemotron-nano-4b`**: Without this, NIM serves the model as `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1`. Llama Stack constructs the external model ID as `{provider_id}/{served_name}` → 3-segment path → RHOAI Playground URL routing breaks (model appears grayed out). A slash-free name produces a 2-segment ID that routes correctly.

**`cpu: 200m` request, `1` limit**: NIM/vLLM is GPU-bound. The only GPU node in `us-east-2b` (where the PVC lives) had ~412m CPU free after DAS operator testing caused operator pods to reschedule there. 200m request (+ 100m rbac-proxy + 100m agent = 400m total) fits within that headroom. The 1-core limit allows CPU bursting during request processing.

**Playground Llama Stack config**: The `llama-stack-config` ConfigMap must NOT include `provider_model_id` in the model entry — when present, Llama Stack uses it instead of `model_id` to construct the external ID, re-introducing the 3-segment slash problem. Also requires `tls_verify: false` because NIM's kube-rbac-proxy sidecar uses a self-signed cert.

## DAS operator warning

The DAS (Dynamic Accelerator Scheduling) operator was previously installed on this cluster and fully removed on 2026-06-22. If it is reinstalled, its `MutatingWebhookConfiguration` (`das-mutating-webhook-configuration`) uses `failurePolicy: Fail` and will block all pod creation cluster-wide if the operator is later deleted without removing the webhook first.
