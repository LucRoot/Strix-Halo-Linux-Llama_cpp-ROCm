# Building llama.cpp with ROCm/HIP for gfx1151 (Strix Halo)

Upstream reference: [llama.cpp HIP build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip).
This page is the gfx1151-specific recipe that produced the binary our fleet
runs, with the ROCm version pinned from the deployed build's linkage.

## Why a pinned ROCm matters

Strix Halo (gfx1151) is a new target. Not every ROCm release ships working
device libraries for it, and llama.cpp's HIP backend compiles kernels per
`AMDGPU_TARGETS`. The deployed, load-tested build links against **ROCm 7.2.2**:

```
libhipblas.so.3    => /opt/rocm-7.2.2/lib/libhipblas.so.3
libamdhip64.so.7   => /opt/rocm-7.2.2/lib/libamdhip64.so.7
librocsolver.so.0  => /opt/rocm-7.2.2/lib/librocsolver.so.0
librocblas.so.5    => /opt/rocm-7.2.2/lib/librocblas.so.5
libhipblaslt.so.1  => /opt/rocm-7.2.2/lib/libhipblaslt.so.1
libhsa-runtime64.so.1 => /opt/rocm-7.2.2/lib/libhsa-runtime64.so.1
librocroller.so.1  => /opt/rocm-7.2.2/lib/librocroller.so.1
```

(`ldd` output of the deployed `llama-server-rocm`, trimmed to ROCm libs.)

The deployed binary reports:

```
version: 1 (4f31eed)
built with GNU 15.2.0 for Linux x86_64
```

## Prerequisites

```bash
# Ubuntu 24.04 baseline
sudo apt install build-essential cmake ninja-build git curl libcurl4-openssl-dev
```

Install ROCm 7.2.2 to `/opt/rocm-7.2.2` per AMD's installer, then verify the
GPU is visible:

```bash
/opt/rocm-7.2.2/bin/rocminfo | grep -m1 -A2 gfx1151
/opt/rocm-7.2.2/bin/rocm-smi --showmeminfo vram | head
```

Expected on a 128 GB Strix Halo box:

```
GPU[0] : VRAM Total Memory (B): 103079215104     # ≈ 96 GiB
```

The runtime user must be in the `render` and `video` groups (see the units).

## The one non-negotiable env var

gfx1151 is not in every ROCm component's supported-target list yet. Without
this override, some HIP kernels fail to load or fall back silently:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```

`11.5.1` is the gfx version encoded in `gfx1151`. Set it for **both** the
build and every runtime process — the systemd units carry it in
`Environment=`.

## Build

```bash
export ROCM_HOME=/opt/rocm-7.2.2
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ROCM_ARCH=gfx1151   # harmless for llama.cpp, keeps one env for the whole stack

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

`[unverified — from build notes]` — the deployed binary predates these notes
and its exact cmake command line was not recorded; the flags above are the
standard upstream HIP recipe narrowed to `gfx1151`, matching the deployed
binary's linkage (`ldd` above) and runtime env. `-DGGML_HIP_ROCWMMA_FATTN=ON`
enables the rocWMMA flash-attention path; if your ROCm lacks rocWMMA, drop it —
flash attention still works via the standard HIP path.

Verify:

```bash
./build/bin/llama-server --version
ldd ./build/bin/llama-server | grep -c rocm-7.2.2   # expect ~10 ROCm libs
```

## Runtime environment (what the fleet units set)

```ini
Environment="ROCM_HOME=/opt/rocm-7.2.2"
Environment="PYTORCH_ROCM_ARCH=gfx1151"
Environment="HSA_OVERRIDE_GFX_VERSION=11.5.1"
Environment="AMD_SERIALIZE_KERNEL=3"
```

- `AMD_SERIALIZE_KERNEL=3` serializes kernel launches. It costs a little
  peak throughput but makes multi-process GPU sharing (7 servers on one GPU)
  far more predictable — worth it the moment you run more than one model.
- Bind to `127.0.0.1` and put auth in front (reverse proxy or API gateway).
  The fleet units bind localhost only.

## Smoke test

```bash
export ROCM_HOME=/opt/rocm-7.2.2 HSA_OVERRIDE_GFX_VERSION=11.5.1 AMD_SERIALIZE_KERNEL=3
./build/bin/llama-server \
  -m /opt/models/some-model-q4_0.gguf \
  -ngl all -c 8192 --port 8080 --host 127.0.0.1 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
```

Watch the load line and VRAM:

```bash
rocm-smi --showmemuse | head
curl -s http://127.0.0.1:8080/health
```

A healthy cold load of a 26B q4_0 model takes ~20–40 s on this box (measured:
Gemma 4-26B loaded in 37.7 s). If it takes *minutes* with the process pinned
at 100% single-core and zero syscalls, you are not looking at a slow load —
see [PITFALLS.md](PITFALLS.md#kfd-driver-thrash-loads-that-spin-forever).

## Notes specific to unified memory

- There is no dedicated VRAM. `rocm-smi` VRAM is a carve-out of the 128 GB
  system pool; every GiB a model takes is a GiB the OS doesn't have. Budget
  with cgroups (see the `MemoryMax=`/`MemoryLow=` lines in [../units/](../units/)).
- `-ngl all` on a single-GPU APU means "everything on the iGPU" — there is no
  second device to spill to, so if it doesn't fit, llama.cpp silently keeps
  tensors on CPU. Check the log for `tensor overrides to CPU` (see
  [MODELS.md](MODELS.md#the-accidental-hybrid-tensor-override)).
