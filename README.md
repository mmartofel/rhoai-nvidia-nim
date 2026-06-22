# RHOAI 3.4 + NVIDIA NIM — Llama-3.1-Nemotron-Nano-4B on T4 GPU

Reproduces the NIM inference service deployment on Red Hat OpenShift AI 3.4 with an NVIDIA T4 GPU (16 GB VRAM).

Reference article: https://developers.redhat.com/articles/2025/05/08/how-set-nvidia-nim-red-hat-openshift-ai

---

## Prerequisites

| Requirement | Notes |
|---|---|
| OpenShift 4.14+ | Tested on sandbox cluster `api.zenek.sandbox5515-opentlc-com:6443` |
| RHOAI 3.4 operator installed | Must be running in `redhat-ods-applications` namespace |
| NVIDIA GPU Operator | Manages the T4 node; `nvidia.com/gpu` resource must be schedulable |
| NGC account + API key | https://org.ngc.nvidia.com/setup/api-keys — use a **Personal API Key** |

---

## Step 1 — Enable NIM in RHOAI (UI, one-time per cluster)

1. Open the RHOAI dashboard → **Applications → Explore**
2. Find the **NVIDIA NIM** tile and click **Enable**
3. Confirm it appears under **Applications → Enabled**

---

## Step 2 — Create NGC secrets

Create both secrets imperatively (never commit real keys). Set your key first:

```bash
export NGC_API_KEY="<your-key>"
```

Create the namespace:

```bash
oc apply -f manifests/00-namespace.yaml
```

Runtime secret — injects `NGC_API_KEY` env var into the NIM container:

```bash
oc create secret generic nvidia-nim-secrets \
  --from-literal=NGC_API_KEY="${NGC_API_KEY}" \
  -n nvidia-nim
```

Image pull secret — used by the ServingRuntime to pull from `nvcr.io`:

```bash
oc create secret docker-registry ngc-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="${NGC_API_KEY}" \
  -n nvidia-nim
```

---

## Step 3 — Apply manifests

```bash
# Create PVC for model cache (30 Gi, gp3-csi)
oc apply -f manifests/01-pvc.yaml

# Create NIM ServingRuntime
oc apply -f manifests/02-servingruntime.yaml

# Create InferenceService (includes T4 fixes — see below)
oc apply -f manifests/03-inferenceservice.yaml

# Ensure RHOAI Playground (GenAI Studio) visibility labels are set.
# Labels are already embedded in 03-inferenceservice.yaml so this is idempotent;
# it also works as a standalone fix for UI-deployed ISVCs.
oc patch inferenceservice llama-31-nemotron-nano-4b -n nvidia-nim \
  --type=merge --patch-file manifests/fix-playground.yaml
```

Watch the pod start up:

```bash
oc get pods -n nvidia-nim -w
```

The predictor pod goes through these phases:
1. `Pending` — PVC attaching, image pulling (~2–3 min if not cached)
2. `Running 1/3` — NIM container loading model weights (~2 min)
3. `Running 3/3` — all containers ready, service live

---

## Step 4 — Verify

```bash
oc get inferenceservice -n nvidia-nim
# READY column should show: True

URL=$(oc get inferenceservice llama-31-nemotron-nano-4b -n nvidia-nim \
  -o jsonpath='{.status.url}')

curl -sk "${URL}/v1/chat/completions" \
  -H "Authorization: Bearer $(oc whoami -t)" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-31-nemotron-nano-4b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
  }'
```

Expected: JSON response with `choices[0].message.content`.

---

## If you deployed via RHOAI UI instead

The UI creates the ServingRuntime and PVC automatically. You still need to apply the T4 patch:

```bash
oc patch inferenceservice llama-31-nemotron-nano-4b -n nvidia-nim \
  --type=merge --patch-file manifests/fix-t4-patch.yaml
```

Then delete the current pod so it restarts with the new config:

```bash
oc delete pod -n nvidia-nim -l serving.kserve.io/inferenceservice=llama-31-nemotron-nano-4b
```

---

## T4 fixes explained

Two bugs hit when deploying any NIM chat model on a T4 (Turing, compute 7.5, 16 GB):

### 1. TensorRT-LLM build loop (`NIM_MODEL_PROFILE`)

NIM auto-selects the `tensorrt_llm-trtllm_buildable` profile on T4. This profile must compile GPU-specific CUDA kernels from scratch on first start — which takes **10+ minutes**. The KServe readiness probe times out and kills the container before the build finishes. Each restart wipes the EmptyDir cache and starts the build over, creating an infinite crash loop.

Fix: force the `vllm` profile via `NIM_MODEL_PROFILE`. vLLM has no build step and starts in ~2 minutes.

The profile ID `4f904d571fe60ff...` is for `nvidia/llama3.1-nemotron-nano-4b-v1.1` v1.8.5. To find the ID for a different model/version, read the NIM container logs — it lists all valid profile IDs on startup.

### 2. KV cache too small for default context (`NIM_MAX_MODEL_LEN`)

The model's default context length is 131 072 tokens. On T4:
- Weights occupy 8.44 GB (FP16, T4 doesn't support BF16)
- ~6 GB remain for KV cache → capacity ≈ 27 000 tokens
- vLLM refuses to start if `max_model_len > KV cache capacity`

Fix: set `NIM_MAX_MODEL_LEN=16384` to cap at 16 K context, which comfortably fits.

---

## Enabling the RHOAI Playground

Steps A (visibility labels) and B (slash-free model name) are already handled by
`03-inferenceservice.yaml` and the `fix-playground.yaml` patch applied in Step 3.

The one remaining manual action is registering the model in the Llama Stack config after
the `lsd-genai-playground` pod is running.

After the `lsd-genai-playground` pod is running, patch the auto-generated `llama-stack-config`
ConfigMap to register the NIM model. Use the Python snippet below (requires `pyyaml` — on RHEL9
nodes it is available under `python3.10 -m pip`):

```bash
oc get configmap llama-stack-config -n nvidia-nim -o jsonpath='{.data.config\.yaml}' \
  > /tmp/llama-stack-config.yaml

python3.10 -c "
import yaml, json

with open('/tmp/llama-stack-config.yaml') as f:
    cfg = yaml.safe_load(f)

# Register NIM model (no provider_model_id — keep model_id slash-free)
models = cfg.setdefault('registered_resources', {}).setdefault('models', [])
entry = next((m for m in models if m.get('provider_id') == 'vllm-inference-1'
              and m.get('model_type') == 'llm'), None)
if entry is None:
    models.append({
        'provider_id': 'vllm-inference-1',
        'model_id': 'llama-31-nemotron-nano-4b',
        'model_type': 'llm',
        'metadata': {'display_name': 'Llama-3.1-Nemotron-Nano-4B'},
    })
else:
    entry['model_id'] = 'llama-31-nemotron-nano-4b'
    entry.pop('provider_model_id', None)

with open('/tmp/llama-stack-config-new.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
"

oc patch configmap llama-stack-config -n nvidia-nim --type=merge \
  -p \"{\\\"data\\\":{\\\"config.yaml\\\":\$(python3.10 -c \\\"import json; print(json.dumps(open('/tmp/llama-stack-config-new.yaml').read()))\\\")}}\""

oc rollout restart deployment/lsd-genai-playground -n nvidia-nim
```

**Why `provider_model_id` must NOT be set:** When present, Llama Stack uses it (not `model_id`) to
form the external ID, producing the 3-segment slash problem described in Step B.

**TLS verification:** The lsd-genai-playground Deployment sets `VLLM_TLS_VERIFY=false` as an env
var. The ConfigMap template `${env.VLLM_TLS_VERIFY:=true}` resolves to `false` at runtime — no
ConfigMap edit needed. Patching it directly would be overwritten on every `oc rollout restart`.

### Verify Playground is working

```bash
# Model should appear as 2-segment ID
LS_POD=$(oc get pods -n nvidia-nim | grep lsd-genai-playground | awk '{print $1}')
oc exec $LS_POD -n nvidia-nim -- curl -s http://localhost:8321/v1/models | \
  python3 -m json.tool | grep '"id"'
# Expected: "id": "vllm-inference-1/llama-31-nemotron-nano-4b"
```

Then open RHOAI dashboard → **AI & ML → Playground**. The model `Llama-3.1-Nemotron-Nano-4B`
should be selectable (not grayed out). Send a message to confirm end-to-end chat.

---

## File index

| File | Purpose |
|---|---|
| `manifests/00-namespace.yaml` | Namespace with RHOAI NIM annotation |
| `manifests/01-pvc.yaml` | 30 Gi PVC for model cache |
| `manifests/02-servingruntime.yaml` | NIM ServingRuntime |
| `manifests/03-inferenceservice.yaml` | InferenceService with T4 fixes + Playground labels + served model name |
| `manifests/04-llamastack-config-patch.yaml` | Documents Llama Stack ConfigMap changes required for Playground |
| `manifests/fix-t4-patch.yaml` | Patch-only: T4 vLLM profile + context cap + served model name |
| `manifests/fix-playground.yaml` | Patch-only: adds Playground visibility to any InferenceService |
| `manifests/fix-cpu-request.yaml` | Patch-only: lowers kserve-container CPU request to 200m (limit stays 1); needed when GPU nodes have limited CPU headroom |

