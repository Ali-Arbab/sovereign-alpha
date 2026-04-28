# Hardware Arrival Runbook

Per master directive §0.5.1.H — point-and-shoot procedure for taking the system from **bootstrap state** (synthetic everything) to **full Sovereign state** (DeepSeek-R1 32B at q6 over the real 1.8M-document corpus). When hardware lands, the work is "swap the model and point at real data" — not "build the system."

Every step has a **verification command** that must succeed before proceeding. If a verification fails, **do not skip ahead** — diagnose and fix at the failing step.

---

## §0 — Pre-arrival checklist (run before hardware arrives)

| # | Item | Verify |
|---|---|---|
| 0.1 | Repo cloned at `G:\Sovereign Alpha`, on `main`, working tree clean | `git status` (no unstaged changes) |
| 0.2 | Python 3.12 installed, `uv` on PATH | `python --version` && `uv --version` |
| 0.3 | `UV_CACHE_DIR=G:\.uv-cache` persistent | `[Environment]::GetEnvironmentVariable("UV_CACHE_DIR", "User")` (PowerShell) |
| 0.4 | `make bootstrap-test` returns 0 in <5min on current hardware | `make bootstrap-test` |
| 0.5 | At least 30% of target corpus storage populated | size of `data/corpus/` |
| 0.6 | Persona library version-locked, all six personas load | `uv run pytest tests/test_personas.py` |

If any of 0.1–0.6 fails, fix before hardware arrives. None require the new hardware.

---

## §1 — Day-of-arrival physical install (T+0 to T+90 min)

1. Unbox: RTX 5090 (32GB VRAM), Ryzen 9 9950X3D (16C/3D V-Cache), 128GB DDR5, 4TB PCIe Gen5 NVMe, X870E Creator motherboard.
2. Build the system per ATX standard. Two notes specific to this stack:
   - Connect the 5090 to a dedicated 16-pin 12V-2x6 PSU rail (1000W+ PSU recommended).
   - Mount the Gen5 NVMe in the **CPU-direct** M.2 slot (typically M.2_1) for the full 14GB/s bandwidth — chipset slots cap at PCIe Gen4.
3. First boot to BIOS:
   - Confirm CPU detected, all 16 cores listed.
   - Confirm 128GB DDR5 detected at the rated speed (e.g. 6000 MT/s).
   - Confirm both NVMe drives detected.
   - Enable XMP/EXPO for full-speed RAM.
   - Enable Resizable BAR (required for full GPU bandwidth utilization).
4. Install Windows 11 Pro to the **new Gen5 NVMe**. The old C: SSD becomes a secondary drive (use it for OS-only files; project moves to the Gen5).
5. **Verify:**
   - `wmic cpu get name,numberofcores` → Ryzen 9 9950X3D, 16 cores.
   - `wmic memorychip get capacity` → four 32GB sticks (or two 64GB) totaling 131072 MB ≈ 128 GB.
   - `nvidia-smi` (after driver install in §2) → `NVIDIA GeForce RTX 5090, 32760 MiB`.

---

## §2 — Software stack install (T+90 min to T+4 hr)

In order:

1. **NVIDIA driver** — latest stable Studio/Game Ready supporting Blackwell. Reboot.
2. **CUDA Toolkit** — version matching the driver (typically 12.4+). Add `bin/` to PATH.
3. **cuDNN** — drop into the CUDA install dir.
4. **Python 3.12** — pin via `pyenv-win` or system installer.
5. **uv** — `pip install uv` or scoop/winget.
6. **git, gh, make** — winget if not present.
7. **Visual C++ Build Tools** — required to compile some Python wheels (e.g. `llama-cpp-python` if not using prebuilt).

**Verify:**
- `nvidia-smi` shows the 5090 with the expected driver version.
- `nvcc --version` returns 12.4+.
- `python --version` returns 3.12.x.
- `uv --version` works.

---

## §3 — Project relocation (T+4 hr to T+5 hr)

The project currently lives on `G:` (HDD, 212 GB). Move it to the Gen5 NVMe. Letter the Gen5 drive `G:` to keep all `G:\Sovereign Alpha\…` paths working without code changes.

```powershell
# Stop any running Python / IDE processes that might hold file locks.
# Re-letter the new Gen5 drive to G: in Disk Management; re-letter the
# old HDD partition to a different letter (e.g. H:).
robocopy "H:\Sovereign Alpha" "G:\Sovereign Alpha" /E /MT:32 /R:3 /W:5
```

Repoint the uv cache to the Gen5:

```powershell
[Environment]::SetEnvironmentVariable("UV_CACHE_DIR", "G:\.uv-cache", "User")
# Restart shell.
```

**Verify:**
- `Get-PhysicalDisk | Format-Table FriendlyName, MediaType` confirms the drive at G: is `SSD`.
- `git status` in the new location works (no `.git` corruption from the copy).
- `uv sync --all-extras` succeeds; `make ci` returns 0.

---

## §4 — Inference engine install (T+5 hr to T+7 hr)

Pick **one** of the two backends. vLLM is faster for batched throughput; llama.cpp is more flexible for quantized models on a single GPU.

### Option A: vLLM
```powershell
uv pip install vllm
```

### Option B: llama-cpp-python (CUDA build)
```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
uv pip install llama-cpp-python --upgrade --no-cache-dir
```

**Verify with the smoke-test harness (§5):**
```powershell
make inference-smoke-test
```

If the smoke-test succeeds, the inference plumbing is wired correctly.

---

## §5 — Replace the synthetic Alpha Ledger with real inference

The bootstrap-phase pipeline currently reads from `synthetic_ledger.py`. Two-step migration:

1. **Smoke test with a 7B / 13B model on a 1k-doc subset.** This is plumbing validation — tag every output `phase=bootstrap, discard_for_research=True`. Per directive §0.5.1.A bullet 13.
   - Sample ~1000 SEC EDGAR 10-Ks at random across the corpus.
   - Run inference with **Qwen 2.5-7B-Instruct** or **Llama 3.1-13B-Instruct** at q6 quantization.
   - Persona: pick one (e.g. `supply_chain_analyst_v1`).
   - Validate output: every record passes `AlphaLedgerRecord` Pydantic validation; `pandera.AlphaLedgerFrame` accepts the full frame.
   - **Verify:** `make bootstrap-test` still returns 0 with the new ledger as input (replacing synthetic).

2. **Full-scale 32B inference.** Once the 7B/13B run is clean.
   - Download **DeepSeek-R1 32B q6_k** (≈26-28GB resident in VRAM).
   - Tokenize the full 1.8M-document corpus (cached, content-addressed; tokenization runs once per tokenizer version).
   - Run inference under each of the six bootstrap personas plus any project-specific additions.
   - Persist each persona's output to `data/alpha_ledger/<persona_id>/year=YYYY/month=MM/part-N.parquet`.
   - Compute the run manifest: `(corpus_hash, persona_hash, model_hash, seed, lockfile_hash)`.
   - **Verify:**
     - `pytest tests/test_temporal_firewall.py` passes.
     - Re-running the same manifest produces a byte-identical output (determinism check).
     - The full Module II backtest runs against the new ledger and produces friction-adjusted Sharpe / Sortino / drawdown reports.

---

## §6 — Validation gates (must pass before declaring "Sovereign state")

| # | Gate | Command |
|---|---|---|
| 6.1 | All unit + property + leak tests | `make test` |
| 6.2 | Lint clean | `make lint` |
| 6.3 | Bootstrap end-to-end | `make bootstrap-test` |
| 6.4 | Inference smoke (when wired) | `make inference-smoke-test` |
| 6.5 | Determinism: re-run with identical manifest yields byte-identical Alpha Ledger | manual diff of two runs' Parquet files |
| 6.6 | Module II backtest produces a friction-adjusted Sharpe with PSR > 0.5 vs SPX buy-hold over 2015-2025 | `scripts/run_full_backtest.py` (post-arrival) |
| 6.7 | UE5 client renders the procedural city from the live Module III bus | open the UE5 project, Play in Editor |

When all seven pass, the system has transitioned from **bootstrap state** to **Sovereign state**. The iterative research loop (directive §7) is now a one-line operation: change a persona prompt → re-run inference overnight → re-fuse → re-backtest → replay in UE5.

---

## §7 — Rollback

If §5 step 2 produces invalid output (schema breach, leak-test failure, non-deterministic re-runs), **do not retry blindly**. Roll back to the synthetic-ledger path:

```powershell
# Restore the bootstrap-test as the canonical entry point
git checkout main -- scripts/bootstrap_test.py
make bootstrap-test  # should still return 0
```

Then diagnose the inference-stage failure in isolation (typically: tokenizer mismatch, persona prompt corruption, or model file corruption). The bootstrap synthetic ledger remains a fully-functional substitute until the real-inference path is repaired.
