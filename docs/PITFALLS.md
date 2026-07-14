# Pitfalls (things that actually bit us)

All evidence below is from the reference box (Strix Halo gfx1151, ROCm 7.2.2,
July 2026). Kernel log lines are verbatim.

## KFD driver thrash: loads that spin forever

**Symptom:** a model that loaded in ~70 s yesterday now parses metadata,
slowly uploads ~13 GB to VRAM over ~10 minutes (I/O-starved), then enters a
100% userspace single-core spin with near-zero syscalls — and never finishes.
One instance consumed 51 min 14 s of CPU time over 52 min 10 s of wall clock
before being killed. Four consecutive failed loads, same binary, same model,
same unit.

**Not the cause:** VRAM, RAM, the model file, or the unit config. The same
combination loaded in 71 s the day before.

**The cause:** on gfx1151, VRAM is carved-out system RAM. A 26 GB swap storm
drove continuous GPU BO eviction/restore inside the kernel's KFD driver.
Kernel log, the week leading up to it:

```
Jul 04 18:18:52 kernel: amdgpu: SVM mapping failed, exceeds resident system memory limit
Jul 04 23:05:35 workqueue: amdgpu_amdkfd_restore_userptr_worker [amdgpu] hogged CPU for >10000us 16387 times
Jul 05 04:15:47 ... 32771 times
Jul 05 13:04:41 ... 65539 times
Jul 06 06:06:58 ... 131075 times
Jul 07 17:49:09 ... 262147 times
```

The restore-worker hog count **doubling daily** is the fingerprint: driver
state degrading across 12 days of uptime. A fresh ROCm process's HSA init
(queues, signals, module load) spins in userspace waiting on completions
stuck behind the restore thrash. Seats that loaded *before* the storm and
held their mappings resident were unaffected — only a cold-loading seat hit
the wall.

**Diagnosis method** (tells "KFD thrash" apart from "slow load"):

```bash
# 1. Process pinned 100% single-core, userspace, almost no syscalls/faults?
top -H -p $(pgrep -f llama-server-rocm | head -1)
pidstat -p <pid> 5 2

# 2. Kernel fingerprints:
journalctl -k --since "7 days ago" | grep -E 'SVM mapping failed|restore_userptr_worker'

# 3. Swap storm in progress or recently?
free -g
```

**Remediation:** nothing userspace can do — HSA init is stuck in the driver.
Reboot. Post-reboot proof on this box: identical binary/model/unit loaded in
**37.7 s** (vs the 71 s pre-storm baseline), health 200, swap 26 GB → 0.

**Prevention:** the swap storm was started by unbudgeted system-RAM pressure.
Per-seat cgroup budgets (`MemoryLow`/`MemoryMax`, in [../units/](../units/))
are the structural fix — see the cgroup memory budgeting discussion in
[units/README.md](../units/README.md) for the budgeting method.

## mmap vs `--no-mmap` under memory pressure

llama.cpp defaults to mmap for model files. On a memory-constrained unified
APU this is a live grenade in two ways:

1. **mmap + CPU tensor overrides = pathological.** Hybrid/MoE models get
   auto-selected CPU tensor overrides; with mmap enabled the loader warns:
   ```
   W llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --no-mmap for better performance
   ```
   On the 35B MoE seat this preceded minutes-long zero-token stalls (partial
   CPU offload thrashing between RAM and GPU). `--no-mmap` + a context size
   that actually fits fixed it; first-token latency dropped ~97 s → ~16 s.
2. **mmap lets the kernel reclaim your weights.** Under pressure, mmap'd
   pages are the first thing the kernel drops — and on this box "dropped
   model pages" are read back from *swap or disk* mid-decode. The LFM hybrid
   seat's decode collapsed to zero tokens for exactly this reason. Fix:
   `--no-mmap` (weights become anonymous memory you can protect) plus
   `MemoryLow=` so reclaim is biased away from the seat.

Rule of thumb for this fleet: `--no-mmap` everywhere **except** seats proven
stable with mmap (gemma4 keeps mmap — it loads in 37.7 s and serves fine;
its warning is logged and watched, not acted on).

## llama-server ignores SIGTERM: budget the SIGKILL wall

`llama-server` does not shut down promptly on SIGTERM — under load (or stuck
in a GPU call) it ignores it entirely. systemd's default `TimeoutStopSec=90`
then ends in SIGKILL. Captured on this box (when the timeout was still 10 s):

```
Jul 03 08:21:31 systemd[1]: <unit>: Killing process ... (llama-server-ro) with signal SIGKILL.
Jul 03 09:57:48 systemd[1]: <unit>: Killing process ... (llama-server-ro) with signal SIGKILL.
```

SIGKILL mid-inference is how you lose slot state, hot caches, and
occasionally wedge GPU memory held by the dead process's context (orphan
processes kept holding ports and VRAM after stop, in our case).

Operational note: every unit sets **`TimeoutStopSec=120`**. That is not
"shutdown takes 2 minutes" — it's the wall before systemd gives up on a
clean stop. Expect stops of a loaded GPU seat to take tens of seconds (slow
GPU teardown is normal); only after 120 s does systemd SIGKILL. If you
orchestrate restarts (deploys, config reloads), sequence them: stop one
seat, wait for the port to free, start the next. Starting all large seats at
once causes GPU memory contention, slow loads, and OOM/SIGKILL thrashing.

## Circular start dependencies (a systemd footgun)

Don't put `Wants=` between model units (e.g. every seat wanting a shared
"warm-up" oneshot that in turn wants all seats). We did exactly that; the
result was that starting *any* seat pulled up *all four* large models
concurrently → GPU contention → SIGKILL/OOM thrash. Use `After=` for
ordering; never `Wants=` across seats. Each unit here stands alone.
