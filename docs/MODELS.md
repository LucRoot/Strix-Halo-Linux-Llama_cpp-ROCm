# Per-model runbooks (earned configs)

Every config below is reproduced from a live systemd unit on the reference
box (Strix Halo, ROCm 7.2.2, July 2026) — not from memory or a sizing
spreadsheet. Unit files: [../units/](../units/). Ports are arbitrary localhost
assignments; keep them consistent so your router/monitoring doesn't care which
model answers.

Fleet at a glance (5 GPU seats + 2 CPU seats, all resident):

| Seat (unit) | Model | Quant | ngl | ctx | parallel | KV K/V | FA | MemoryMax / Low |
|---|---|---|---|---|---|---|---|---|
| `llama-gemma4` | Gemma 4-26B-A4B QAT | q4_0 | all | 32768 | 2 | q8_0/q8_0 | on | 16G / 4G |
| `llama-lfm` | LFM2.5-8B-A1B | IQ4_NL | (hybrid) | 16384 | 1 | q8_0/q8_0 | on | 8G / 4G |
| `llama-qwen-35b-moe` | Qwen-35B-A3B (post-trained) | Q4_K_M | all | 16384 | 2 | q8_0/q8_0 | on | 16G / 1G |
| `llama-glm-flash` | GLM-4.7-Flash | Q3_K_M | all | 32768 | 2 | q8_0/q8_0 | on | 12G / 1G |
| `llama-cascade2` | Nemotron Cascade 2 30B-A3B | Q4_K_M | all | 16384 | 1 | q8_0/q8_0 | on | 28G / 1G |
| `llama-llama3b-cpu` | Llama 3.2 3B fine-tune | Q6_K | 0 (CPU) | 32768 | 2 | f16 | off | 14G / 5G |
| `llama-gpt-oss-20b-cpu` | GPT OSS 20B | MXFP4 | 0 (CPU) | 8192 | 1 | q5_1/q4_0 | off | 22G / 7G |

Measured sequential cold-load times (same box, quiet system):
llama3b-cpu 10 s, gemma4 20 s, glm-flash 20 s, qwen-35b-moe 40 s,
cascade2 40 s. VRAM after all five GPU seats: 85.3 GB / 103 GB (83%).

---

## Gemma 4-26B-A4B QAT (q4_0) — `llama-gemma4.service`

```
-m /opt/models/gemma-4-26B_q4_0-it.gguf -ngl all -c 32768 \
--threads 16 --threads-batch 16 --parallel 2 --cont-batching \
--cache-ram 512 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
```

### Control-token warnings at load are expected, not fatal

Every load of this GGUF prints:

```
W load: override 'tokenizer.ggml.add_bos_token' to 'true' for Gemma4
W load: control-looking token:    212 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
W load: control-looking token:     50 '<|tool_response>' was not control-type; this is probably a bug in the model. its type will be overridden
W load: special_eog_ids contains '<|tool_response>', removing '</s>' token from EOG list
```

These are llama.cpp normalizing the QAT export's tokenizer metadata. The model
loads and serves fine — a load that prints these and reaches `model loaded`
in ~40 s is healthy. Don't chase them; do note that `</s>` is dropped from the
EOG list, so if your client stops on `</s>` it will run until
`<|tool_response>` or max tokens.

### `-ngl all` + mmap: the accidental-hybrid-tensor-override

Gemma 4 is a hybrid MoE; llama.cpp auto-selects CPU overrides for some
tensors when it decides they don't pay off on the GPU. With default mmap you
get this warning:

```
W llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --no-mmap for better performance
```

On this box that combination still loads (37.7 s post-fix), so the unit keeps
mmap. **The pitfall is assuming the warning is cosmetic under pressure:** the
same warning on the Qwen MoE seat (below) preceded a pathological
no-tokens-for-120s hang. If gemma4 ever degrades to minutes-per-token, first
check is `journalctl -u llama-gemma4 | grep 'tensor overrides'` and flip to
`--no-mmap`.

Also note: with `-ngl all` this build logs
`common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to -2, abort`
and proceeds — `-ngl all` is a user-set value, so the auto-fit aborts instead
of shrinking. That line is benign *if* the model actually fits; if VRAM is
already full, "fit" means silent partial CPU offload. Check `rocm-smi`.

### KV and context

32k context × `--parallel 2` ⇒ each slot gets 16k (`n_ctx_slot = 16384` in
the load log). q8_0 KV at 32k is the single biggest VRAM line item for this
seat; see the [KV rationale](#kv-cache-quantization-rationale) below.

---

## LFM2.5-8B-A1B (Liquid hybrid) — `llama-lfm.service`

Model: `unsloth/LFM2.5-8B-A1B-GGUF`, `LFM2.5-8B-A1B-UD-IQ4_NL.gguf`
(4,358,844,000 bytes).

```
-m /opt/models/LFM2.5-8B-A1B-UD-IQ4_NL.gguf --no-mmap -c 16384 \
--parallel 1 --cont-batching --cache-ram 512 \
--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
```

- **q8_0/q8_0 KV at 16k ctx** — the point of this seat is a different
  training base at nearly no VRAM cost; IQ4_NL weights are ~4.1 GiB, and the
  q8 KV is cheap at 16k.
- **`--no-mmap` is load-bearing here.** LFM2.5 is a hybrid arch: it keeps
  conv/SSM state on the CPU side. Under system memory pressure the kernel
  reclaims that state, and decode collapses to zero tokens for minutes while
  everything looks "up". The unit's own comment says it best:

  > Hybrid arch keeps CPU-side conv/SSM state that the kernel swaps out under
  > system memory pressure, collapsing decode to zero tokens for minutes.
  > Bias reclaim away from this seat so its working set stays resident.

  That's why this seat gets `MemoryLow=4G` — cgroup protection, not a limit
  (see docs/PITFALLS.md for the memory-budgeting rationale).
- `--parallel 1`: one slot, full 16k context. Measured 13.8 tok/s under fleet
  load after the MemoryLow fix.

---

## Qwen-35B-A3B (post-trained, Q4_K_M) — `llama-qwen-35b-moe.service`

```
-m /opt/models/Qwen-35B-A3B-Q4_K_M.gguf -ngl all --no-mmap -c 16384 \
--parallel 2 --cont-batching --cache-ram 2048 \
--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
```

### The mmap + partial-offload hang (why `--no-mmap` and 16k)

This 20 GB-class MoE originally ran at `-c 32768` with mmap. Symptoms under
fleet load: `No tokens received for 120.0s` on the API, the warning
`tensor overrides to CPU are used with mmap enabled`, and `rocm-smi` showing
only ~13.8 GB VRAM for this seat while a comparable 30B seat held 24.8 GB —
i.e. silent partial CPU offload with mmap thrashing between RAM and GPU.
Fix that held: `--no-mmap`, context halved 32768 → 16384, `--cache-ram`
raised 512 → 2048 to reduce slot-eviction churn. First-token latency on a
trivial prompt dropped ~97 s → ~16 s.

Lesson for MoE seats on a shared APU GPU: **verify the offload with
`rocm-smi`, not with `-ngl all`.** `-ngl all` is a request, not a receipt.

### The Unsloth IQ2 finding

We evaluated replacing this seat with Qwen-Coder-Next, pulling Unsloth's
`UD-IQ4_XS` (36 GB) and queueing `UD-IQ2_M` / `UD-IQ3_S`. The evaluation was
stopped and the swap rejected: the incumbent Q4_K_M post-trained model was
better, faster, and already fits the budget — and at 35B, IQ2-class quants
buy you a few GiB at a visible quality cost you feel immediately on
orchestration/tool-use prompts. Standing rule from that evaluation: on a
96 GiB APU GPU, **don't quant a 30–35B MoE below ~Q4_K_M to "make room"**;
you have room. Spend quants on the KV cache instead (below).

---

## GLM-4.7-Flash (Q3_K_M) — `llama-glm-flash.service`

```
-m /opt/models/GLM-4.7-Flash-Q3_K_M.gguf -ngl all --no-mmap -c 32768 \
--parallel 2 --cont-batching --cache-ram 1024 \
--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
```

### Q3 vs Q4 sizing

History on this box: every tested GLM-4.7-Flash GGUF (including Unsloth's
`UD-Q4_K_XL`, ~17 GB) **segfaulted llama.cpp's ROCm path during GPU offload**
early on, so the seat spent time as a CPU model at Q4. The current build
loads `Q3_K_M` on the GPU cleanly at 32k/parallel-2 inside a 12 GB
`MemoryMax`. The sizing logic:

- Q4_K_XL (~17 GB weights) + q8_0 KV at 32k × 2 slots does not fit this
  seat's share of the GPU alongside four other residents.
- Q3_K_M fits with headroom, and for a coder/tool-use seat the Q3→Q4 quality
  delta is smaller than the latency delta of falling back to CPU.

So: **Q3 on GPU beats Q4 on CPU.** If you have the box to yourself, run Q4;
in a fleet, fit beats precision.

---

## Nemotron Cascade 2 30B-A3B (Q4_K_M) — `llama-cascade2.service`

```
-m /opt/models/Nemotron-Cascade-2-30B-A3B.i1-Q4_K_M.gguf -ngl all --no-mmap \
-c 16384 --parallel 1 --cont-batching --cache-ram 512 \
--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
```

Hybrid `nemotron_h_moe` arch: all 52 layers carry attention tensors *and* 23
of them carry SSM/Mamba tensors (checked via GGUF metadata). That's why this
seat stays at 16k: with 52 attention layers, KV at 32k would dominate the
budget. `--parallel 1` — a reasoner/math seat wants long single contexts more
than concurrency. Largest seat budget: `MemoryMax=28G`.

---

## CPU-resident seats

`-ngl 0` — these run on the 16 Zen 5 cores and leave the GPU to the big
models. CPU seats are where you *cut* KV precision, because their memory is
your system RAM:

- `llama-llama3b-cpu` (3B, Q6_K): `-c 32768 --parallel 2`, **no KV quant**,
  `--cache-ram 4096`. Small enough that full-precision KV is free; this is
  the high-throughput small seat. `MemorySwapMax=2G` gives it a pressure
  release valve instead of an OOM.
- `llama-gpt-oss-20b-cpu` (20B, MXFP4): `-c 8192 --parallel 1`,
  `--cache-type-k q5_1 --cache-type-v q4_0`, `--cache-ram 1024`. Context was
  cut 16384 → 8192 and KV dropped to q5_1/q4_0 specifically to shrink its
  system-RAM footprint; that plus `MemoryMax=22G` keeps it from starting the
  swap storms that take down the GPU seats (see PITFALLS).

---

## KV cache quantization rationale

All five GPU seats run `--cache-type-k q8_0 --cache-type-v q8_0`:

- KV at q8_0 halves the KV memory vs f16 with no measurable quality loss on
  our workloads, and on a unified-memory APU the KV cache is the budget line
  you can actually control — weights are fixed by the quant you downloaded.
- **q8_0 V-cache requires flash attention on this build:**
  `V cache quantization requires flash_attn` — `--flash-attn off` with
  `--cache-type-v q8_0` is rejected. Keep `--flash-attn on` for every
  KV-quantized seat.
- CPU seats go lower (q5_1/q4_0, or q4_0/q4_0 on the voice-class 4B) because
  there the cache competes with the OS, not with other models.

## `--parallel` / continuous batching tuning

- `--parallel N` splits `-c` into N equal slots (`n_ctx_slot = c / N` in the
  load log). `--parallel 2 -c 32768` = two 16k slots. Decide slot count by
  the context your *workload* needs, then set `-c` to cover it.
- `--cont-batching` is on for every seat; with `--parallel 1` it still earns
  its keep on chunked prefill.
- Raising fleet-wide concurrency has a cliff: pushing all specialists to high
  parallelism at once made the 35B MoE the critical path (the hang story
  above). Big MoE seats stay at `--parallel 1–2`; small CPU seats can take 2+.
- `--threads 16 --threads-batch 16` on a 16-core box; the CPU seats share
  those cores, so don't oversubscribe `--parallel` on CPU seats when GPU
  seats are also prompt-processing.

## Flash attention on ROCm

`--flash-attn on` on every GPU seat: required for q8_0 V-cache (above) and
measurably cheaper on memory at 32k. One caveat from this box: under severe
memory pressure we captured crash backtraces inside the HIP flash-attn kernel
(`launch_fattn<256, 8, 2>` / `ggml_cuda_flash_attn_ext_tile_case`). The root
cause was the memory storm, not flash-attn — the same seats with the same
flag run clean once memory is budgeted — but if you see fattn crashes, check
swap and KFD state *first* ([PITFALLS.md](PITFALLS.md)) before blaming the
flag.
