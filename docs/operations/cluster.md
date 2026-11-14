# Training on the GPU cluster (first run)

This guide takes the sea-state experiment
([currituck_sea_state](../experiments/currituck_sea_state.md)) from the desktop
to an H200 and back. It assumes no earlier cluster experience. Everything in
`code blocks` is typed as shown; `<angle brackets>` are values you fill in.

**What goes to the cluster:** the code (from GitHub) and one 3.3 GB package of
resized stills with their labels. The 294 GB archive stays on the desktop.

## How a cluster is organised

A cluster has **login nodes**, where you arrive by SSH, edit files and submit
work, and **compute nodes**, where the GPUs are. A scheduler called **Slurm**
decides which job runs on which compute node and when. You describe a job in a
short script (`deploy/slurm/train_sea_state.sbatch`), hand it to Slurm with
`sbatch`, and it runs when a GPU is free, even after you log out.

**Never train on the login node.** It is shared by everyone logged in and has
no GPU for you; administrators stop long processes there. Installing software
and copying data on the login node is fine.

## 0. Questions for the cluster's administrators (Jose)

Their answers fill the `CHANGE_ME` lines of the job script and the paths below.

1. How do I get an account, and can I connect from off campus (VPN)?
2. Is it Slurm? Which partition, account and GPU request line should I use
   (`--partition`, `--account`, `--gres`), and what is the longest job allowed?
3. Docker, Apptainer, or a plain Python environment (conda or uv)? This
   guide uses uv; if containers are required, tell me and an Apptainer recipe
   will be added.
4. Where should datasets live (home, project or scratch space), what are the
   quotas, and is scratch purged?
5. What is the best way to bring in 5–300 GB: a data-transfer node, Globus, or
   can the cluster download from the internet (Cloudflare R2)? Does the login
   node have internet (needed once, in step 3)?
6. Is it a whole H200 per job, or a share of one (MIG)?
7. Is there an example job script for this cluster?

## 1. Log in

From the desktop (on the VPN if you are off campus):

```bash
ssh <user>@<login-host>
```

The first time, SSH shows the host's key fingerprint and asks
`Are you sure you want to continue connecting (yes/no)?`: type `yes`. Then the
password prompt shows nothing while you type; that is normal. You arrive at a
prompt like `[<user>@login1 ~]$`. `exit` logs out.

## 2. Get the code

On the login node:

```bash
git clone https://github.com/StevenFAU/Wave_Analysis.git
cd Wave_Analysis
```

Expect `Cloning into 'Wave_Analysis'...` and a few progress lines. Later,
`git pull` brings in new commits.

## 3. Set up the environment (once)

Still on the login node, in `Wave_Analysis`:

```bash
bash deploy/slurm/setup_env.sh
```

It installs uv in `~/.local/bin`, then the exact package versions of
`uv.lock` with PyTorch into `Wave_Analysis/.venv` (several GB, a few minutes),
and downloads the ImageNet weights into `Wave_Analysis/.cache/torch` (compute
nodes often have no internet). The last lines look like:

```text
torch 2.14.0, built for CUDA <version>; weights in /home/<user>/Wave_Analysis/.cache/torch
Environment ready. On a GPU node, 'nvidia-smi' must show a 'CUDA Version' at least the one above.
```

(The exact versions may differ.) If the home folder is small (question 4), ask
where a project folder should go and clone there instead.

## 4. Copy the package

On the **desktop**, in the repository (the package was built there with
`scripts/sea_state_package.py`):

```bash
rsync -a --partial --info=progress2 data/processed/packages/currituck_sea_state_v0/ <user>@<login-host>:<data-folder>/currituck_sea_state_v0/
```

`<data-folder>` is the dataset space from question 4 (for example
`/scratch/<user>`). The trailing slashes matter: they copy the folder's
contents into a folder of the same name. A progress line counts up to about
3.3 GB; if the connection drops, the same command resumes. Then, **on the
cluster**, check that every file arrived intact:

```bash
cd <data-folder>/currituck_sea_state_v0 && sha256sum -c --quiet SHA256SUMS && echo intact
```

It prints `intact` after a few seconds. Any other output names the files to
copy again.

## 5. A smoke test (5 minutes)

Fill the four `CHANGE_ME` lines in `deploy/slurm/train_sea_state.sbatch`
(`nano deploy/slurm/train_sea_state.sbatch`; save with Ctrl+O, Enter, leave
with Ctrl+X), then, in `Wave_Analysis`:

```bash
mkdir -p logs
sbatch --export=ALL,PACKAGE=<data-folder>/currituck_sea_state_v0,LIMIT=64,EPOCHS=1 deploy/slurm/train_sea_state.sbatch
```

Slurm answers `Submitted batch job <job-id>`. This run uses 64 stills per
partition and one epoch: it checks the GPU, the environment and the paths, not
the model.

## 6. Watch, stop, and check a job

```bash
squeue -u $USER                     # your jobs: ST is PD (waiting) or R (running)
tail -f logs/sea-state-<job-id>.out # the job's output as it is written; Ctrl+C stops watching, not the job
scancel <job-id>                    # stop a job
sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS   # after it ends: COMPLETED, FAILED, TIMEOUT ...
```

A smoke test's log starts with the node name and the GPU
(`NVIDIA H200, 143771 MiB, ...`), then one line per epoch, the metrics of each
partition, and ends with `finished at ...; results in .../runs/<split>_sea_crop_<job-id>`.
If the job fails at once, the log's last lines say why; common causes are a
`CHANGE_ME` left in the script, a wrong `PACKAGE` path, or a driver older than
the CUDA version of step 3.

## 7. The real runs

Each takes well under an hour on one H200. Submit them together; Slurm runs
them as GPUs free up:

```bash
P=<data-folder>/currituck_sea_state_v0
sbatch --export=ALL,PACKAGE=$P,SPLIT=year deploy/slurm/train_sea_state.sbatch
sbatch --export=ALL,PACKAGE=$P,SPLIT=day deploy/slurm/train_sea_state.sbatch
sbatch --export=ALL,PACKAGE=$P,SPLIT=year,INPUT=full_frame deploy/slurm/train_sea_state.sbatch
```

The first two are the experiment (held-out year, held-out days); the third is
the ablation that shows what the beach adds (people, umbrellas, the season).

## 8. Bring the results back

Each run writes a folder under `Wave_Analysis/runs/`. On the desktop:

```bash
rsync -a <user>@<login-host>:Wave_Analysis/runs/ runs/
```

In each run folder, `report.md` lists every partition's metrics with their
95 % intervals, next to each baseline on the same stills;
`confusion_test.svg` is the confusion matrix of the test partition (open it in
a browser); `metrics.json` holds all numbers, `predictions.csv` every still's
prediction, and `provenance.yaml` the commit, inputs and GPU.

## Settings at a glance

| Variable (`--export=ALL,NAME=value`) | Default | Meaning |
|---|---|---|
| `PACKAGE` | `/scratch/$USER/currituck_sea_state_v0` | the copied package |
| `SPLIT` | `year` | `year`: train 2022–2024, test 2025 and 2026; `day`: held-out days |
| `INPUT` | the config's `sea_crop` | `full_frame` for the ablation |
| `BACKBONE` | the config's `convnext_tiny` | `resnet50` also works |
| `EPOCHS`, `LIMIT` | config, all stills | smoke tests |
| `STAGE` | `1` | copy the package to the node's local disk (`$TMPDIR`) before training |

## Training on the desktop GPU (ROCm)

> **Do not train on the desktop GPU until its cooling has been checked
> (2026-11-14).** At 16:12:14 UTC, three minutes into a timing test, the
> driver logged `GPU over temperature range(SW CTF) detected` and powered the
> desktop off (an orderly shutdown). After the restart, a load test that
> stopped itself at 90 °C saw the card go from 58 to 80 °C at the edge sensor
> and from 69 to 98 °C at the hottest point within half a second of full load
> (262 W), with the fan turning (1,000-1,400 rpm). That speed points to poor
> contact between the chip and its heatsink (a cooler shifted, for example in
> the move of 2026-11-13, or dried thermal paste), not to the fan curve. The
> Currituck runs of 2026-11-13/14 (9 h) had finished without a trip. Idle
> limits read by the driver: edge 100 °C, junction 110 °C (critical).
> **After cleaning (2026-11-14 16:40 UTC) the card is unchanged:** the same
> load took the hottest point from 53 to 105 °C and the edge from 46 to 85 °C
> in about one second (229-262 W), and when the power dipped to 162 W it fell
> 20 °C as fast. A chip that heats and cools in fractions of a second is
> barely in contact with its heatsink: the cooler needs to come off and be
> refitted with new paste (and pads), or the card repaired (Sapphire,
> subsystem `1da2:439e`). The test script, which stops itself at set limits:
> `.venv-rocm/bin/python .claude/bench/gpu_heat.py 180 100 85` (seconds,
> junction °C, edge °C; local, not in the repository).
> Check temperatures without root:
> `cat /sys/class/drm/card*/device/hwmon/hwmon*/temp{1,2}_input` (milli-°C).

The same three runs fit on the desktop's AMD card (Radeon RX 6800 XT class,
gfx1030, 17.2 GB), without waiting for a cluster account. Set up and timed on
2026-11-13; the three Currituck runs ran on it on 2026-11-13/14.

**Environment (once, no sudo):** `bash deploy/local/setup_rocm_env.sh` builds
`.venv-rocm` with uv.lock's versions and PyTorch 2.14.0 for ROCm 7.2 (a 6.2 GB
download, 16 GB installed). The project's `.venv` stays CPU-only. The GPU is
reachable because the logged-in user has access to `/dev/kfd` and
`/dev/dri/renderD128`.

**Measured speeds** (`convnext_tiny`, training steps on batches of 64;
loading and augmenting stills on the CPU gives ~1,000 sea crops/s and ~400
full frames/s with 8 workers, so the GPU sets the pace):

| Input | Precision | Pieces of | Train | Predict | GPU memory |
|---|---|---|---|---|---|
| sea crop 168 × 900 | bf16 (the config) | 64 | 12.9 img/s | | 12.3 GB |
| sea crop | fp16 | 64 | 70 img/s | | 12.3 GB |
| sea crop | **fp32** | **32** | **53 img/s** | 171 img/s | 10.9 GB (64 at once: out of memory) |
| full frame 509 × 900 | fp16 | 16 | 22.9 img/s | | 9.5 GB |
| full frame | **fp32** | **8** | **17.4 img/s** | 53 img/s | 8.4 GB (16: out of memory) |

- **fp32, not bf16:** the card has no bf16 hardware, and bf16 runs five times
  slower. fp16 is 25–35 % faster than fp32 but would need loss scaling, which
  the training code does not have; fp32 needs nothing new.
- **Pieces (`--micro-batch`):** each batch of 64 is computed in pieces whose
  gradients are summed before the optimizer step, the loss divided by the
  whole batch's class weight, so the update is the full batch's (test
  `test_micro_batches_give_the_same_update`; ConvNeXt has no batch
  normalization).
- **No cuDNN benchmark mode on ROCm:** with it, MIOpen compiles kernels at run
  time and fails (`'type_traits' file not found`); the script turns it off
  when `torch.version.hip` is set.
- A 1,000-still, one-epoch check on the real package ran at the speeds
  above plus ~6 s per epoch for starting the loader workers.

**Expected time**, 15 epochs each, run one after the other:

| Run | Per epoch | Whole run |
|---|---|---|
| year split, sea crop (the main result) | ~6.6 min | ~1 h 50 min |
| day split, sea crop | ~8.8 min | ~2 h 20 min |
| year split, full frame (ablation) | ~20 min | ~5 h 15 min |
| all three | | **~9.5 h** |

**Start** (the package is on the internal disk, so the WebCOOS drive does not
matter; the desktop may feel slower while the GPU is busy):

```bash
systemd-run --user --collect --unit wave-analysis-train -p Nice=5 --working-directory="$PWD" bash deploy/local/train_sea_state.sh
```

```bash
journalctl --user -u wave-analysis-train -f    # follow; Ctrl+C stops following, not the runs
```

```bash
systemctl --user stop wave-analysis-train      # stop
```

Each run writes `runs/desktop_<split>_<input>_<UTC time>/`, with the same
files as a cluster run (section 8). The log prints `(null): No such file or
directory` from the ROCm libraries at start-up; the launcher filters it out,
and it is harmless.
