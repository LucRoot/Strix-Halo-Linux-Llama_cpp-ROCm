# strix-halo-llama-cpp-rocm

Production llama.cpp (ROCm/HIP) serving on the AMD Strix Halo APU (gfx1151):
a build recipe that actually links against a modern ROCm, plus per-model
serving runbooks earned by running seven models concurrently on one box.

---

## TL;DR

This repo documents a working ROCm/HIP build of llama.cpp for the AMD Strix
Halo APU (`gfx1151`) and a set of systemd units that run a 7-model fleet on
one 128 GB unified-memory box. The key finding: with ROCm 7.2.2, the
`HSA_OVERRIDE_GFX_VERSION=11.5.1` override, cgroup memory budgets, and
`--no-mmap` for MoE/hybrid seats, you can keep five GPU-resident models and
two CPU-resident models resident and serving on a single APU GPU. All
configs trace to live systemd units and captured logs from the reference
machine.

Read the upstream [llama.cpp HIP build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip)
first for the generic flow; everything here is what upstream docs don't tell
you about this specific APU.

---

## Environment

| Component | Detail |
|---|---|
| APU | AMD Ryzen AI Max+ 395 ("Strix Halo") |
| GPU | Radeon 8060S, gfx1151, 40 CUs |
| Memory | 128 GB unified LPDDR5X; GPU-visible VRAM ≈ 96 GiB (`rocm-smi` reports 103,079,215,104 B) |
| OS | Ubuntu 24.04 |
| ROCm | 7.2.2 (pinned — see [docs/BUILD.md](docs/BUILD.md)) |
| Compiler | GCC 15.2.0 |
| llama.cpp | build `1 (4f31eed)`, HIP backend |
| CUDA | None (AMD GPU only) |

The whole point of this APU: "VRAM" is carved-out system RAM, so a 26B model
at q4_0 and a 35B MoE at Q4_K_M coexist on one GPU — *if* you budget memory
deliberately. Seven seats serving at once, ~78 GiB VRAM in use:

```
GPU[0] : VRAM Total Memory (B): 103079215104
GPU[0] : VRAM Total Used Memory (B): 83395117056
```

---

## What's in here

- [docs/BUILD.md](docs/BUILD.md) — full ROCm 7.2.2 llama.cpp build recipe for
  gfx1151 (`HSA_OVERRIDE_GFX_VERSION=11.5.1`, cmake flags, runtime env).
- [docs/MODELS.md](docs/MODELS.md) — per-model runbooks: Gemma 4-26B QAT,
  LFM2.5-8B-A1B, Qwen-35B-A3B, GLM-4.7-Flash, Nemotron Cascade 2, plus
  CPU-resident seats. KV-cache quantization rationale, flash-attn notes,
  `--parallel` / continuous-batching tuning.
- [docs/PITFALLS.md](docs/PITFALLS.md) — the failures that cost real hours:
  KFD driver thrash (loads that spin forever), mmap vs `--no-mmap` under
  memory pressure, llama-server ignoring SIGTERM.
- [units/](units/) — sanitized systemd units for each seat, including the
  cgroup memory budgets that keep a 7-model fleet from eating itself.

---

## Step-by-step instructions

### 1. Install prerequisites

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build git curl libcurl4-openssl-dev
```

Install ROCm 7.2.2 to `/opt/rocm-7.2.2` per AMD's installer, then verify the
GPU is visible:

```bash
/opt/rocm-7.2.2/bin/rocminfo | grep -m1 -A2 gfx1151
/opt/rocm-7.2.2/bin/rocm-smi --showmeminfo vram | head
```

The runtime user must be in the `render` and `video` groups.

### 2. Build llama.cpp

```bash
export ROCM_HOME=/opt/rocm-7.2.2
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ROCM_ARCH=gfx1151

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build -G Ninja \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DLLAMA_CURL=ON
cmake --build build -j --target llama-server
```

### 3. Smoke-test on GPU

```bash
export ROCM_HOME=/opt/rocm-7.2.2
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export AMD_SERIALIZE_KERNEL=3

./build/bin/llama-server \
  -m /opt/models/some-model-q4_0.gguf \
  -ngl all -c 8192 --port 8080 --host 127.0.0.1 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
```

### 4. Install fleet units (optional)

Copy the units, edit model paths and memory budgets to your fleet, then:

```bash
sudo cp units/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-gemma4.service
```

---

## Verification / testing

Run these commands to prove the build and a single seat are healthy:

```bash
# 1. Binary links against the expected ROCm version
./build/bin/llama-server --version
ldd ./build/bin/llama-server | grep -c rocm-7.2.2   # expect ~10 ROCm libs

# 2. Server starts and /health returns 200
export ROCM_HOME=/opt/rocm-7.2.2 HSA_OVERRIDE_GFX_VERSION=11.5.1 AMD_SERIALIZE_KERNEL=3
./build/bin/llama-server \
  -m /opt/models/some-model-q4_0.gguf \
  -ngl all -c 8192 --port 8080 --host 127.0.0.1 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 &

sleep 45   # cold load on a large model can take 20-40 s
curl -s http://127.0.0.1:8080/health
kill %1
```

A healthy cold load of a 26B q4_0 model takes ~20-40 s on this box (measured:
Gemma 4-26B loaded in 37.7 s). If it takes *minutes* with the process pinned
at 100% single-core and zero syscalls, see
[docs/PITFALLS.md](docs/PITFALLS.md#kfd-driver-thrash-loads-that-spin-forever).

To verify a fleet seat:

```bash
sudo systemctl start llama-gemma4.service
sudo systemctl status llama-gemma4.service
journalctl -u llama-gemma4 -n 50 --no-pager
curl -s http://127.0.0.1:8013/health
```

---

## Known limitations

- **ROCm 7.2.2 is pinned for a reason.** Newer ROCm releases may ship broken
  or missing device libraries for gfx1151. Build against 7.2.2 first, then
  experiment.
- **Single GPU, no spillover.** `-ngl all` on a one-GPU APU means "everything
  on the iGPU". If a model doesn't fit, llama.cpp silently keeps tensors on
  CPU. Check `rocm-smi` and the load log for `tensor overrides to CPU`.
- **Unified memory is system memory.** Every GiB given to a model is a GiB the
  OS cannot use. Without cgroup budgets, a swap storm can wedge the KFD
  driver and make cold loads spin forever.
- **`--flash-attn on` is required for q8_0 V-cache.** The build rejects
  `--cache-type-v q8_0` without it.
- **Measurements are from one box on one day.** Re-run the verification
  commands on your own hardware.
- **All configs are localhost-only.** The units bind `127.0.0.1`; put auth in
  front (reverse proxy or API gateway) before exposing to a network.

---

## Reproduction notes

Every config value in this repo traces to a live systemd unit or a captured
log on the reference box (Strix Halo, ROCm 7.2.2, July 2026). Shell commands
were run on that box unless explicitly tagged
`[unverified — from build notes]`.

To reproduce the full fleet:

1. Build llama.cpp per [docs/BUILD.md](docs/BUILD.md).
2. Download the models referenced in [docs/MODELS.md](docs/MODELS.md) to a
   local path (e.g., `/opt/models/`) and update the unit files to match.
3. Create the service user and directories referenced in
   [units/README.md](units/README.md).
4. Install and start the units one at a time; starting all large seats at
   once causes GPU memory contention and OOM/SIGKILL thrashing.

---

## License

PolyForm Noncommercial 1.0.0 — see [LICENSE](LICENSE).

---

**Author:** Dr. Lucas Root, Ph.D. — [info@lucasroot.com](mailto:info@lucasroot.com)
