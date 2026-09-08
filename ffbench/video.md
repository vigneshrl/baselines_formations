# FastFunnels presentation videos — template and catalogue

Audience: stakeholders and the paper. Every video must be understandable without a narrator:
what is being compared, what happened, and *why* it happened, measured from the run itself.
This file is the specification. It does not generate anything.

---

## 1. Global template (applies to every video)
At the begining of the video show all the maps in the start of the video showing this is what we are going to work with. In one slide. 
### 1.1 Layout
| element | spec |
|---|---|
| panels | one panel (solo run) or two panels side by side (comparison). Panel = the environment render, 900x700 px (patch-only renderer) or 800x700 px (follower runner). 30 px gutter between panels. |
| header bar (top, 78-80 px) | line 1: **what this panel is** (policy name + checkpoint, or "PATCH CAR ALONE" / "PATCH CAR + N FOLLOWERS"). line 2: live HUD, see 1.3. |
| narration bar (bottom, 150-210 px) | the last 4 timed events, one per line, wrapped; colour-coded (1.4). The per-scene verdict line appears at the bottom once the run ends. |
| inset (optional, top-right of a panel) | full-map overview: track in grey, **narrow regions in amber dots**, trajectory in blue, current position as a red dot. Use whenever the course has narrow regions or is longer than one screen. |
| cards | intro card 5-7 s (what left/right are, colour legend, "one shared clock, real speed"); per-scene title card 2.5 s (map label, N, course length, obstacle count); end-of-scene hold 1.5 s; summary card 10-12 s with one line per scene. |

### 1.2 What must be visible in the panel
- the **patch car body** (rectangle at the car pose, 0.58 x 0.31 m) — not just a marker;
- the **funnel ellipse** and its **deformation** (a, b changing along the course);
- follower cars, colour-coded, with the ones outside the funnel outlined red (follower videos);
- a metre **grid** with labelled axes (5 m squares at normal scale, 25 m squares at 5x scale). On scaled maps the view window scales with the map so the corridor looks the same size on screen and the grid is what shows the scale;
- obstacles and walls (map render).

### 1.3 Live HUD fields (header line 2)
`t = <s>   progress <%>   v <m/s>   funnel <2a> x <2b> m   corridor <width> m   <NARROW REGION | open section>`
plus, for follower panels, `inside <k>/<N>` (green when k = N, red otherwise) and, when a run has ended, its verdict text in its colour.

### 1.4 Colour code (strict)
| colour | meaning |
|---|---|
| green | success: obstacle passed, all followers back inside, ARRIVED, reached the course end |
| amber | behaviour worth noting, not a failure: enters the narrow region, drops to 2.0 m/s briefly, PARTIAL verdicts |
| red | failure **with the measured reason**: follower left the funnel (and why), car-to-car contact, COLLISION (and what it clipped), STUCK, loiter |
| grey | neutral context ("start", course facts) |

### 1.5 Narration events (all derived from the logged trajectory, never hand-written)
Patch events:
- `passes obstacle k with <d> m to spare, funnel <w> m wide, <v> m/s` (closest approach < 6 m, while still progressing)
- `enters the narrow region: corridor closes to <w> m, funnel shrinks to <w> m minimum`
- `brakes to the 2.0 m/s floor for <t> s to get past obstacle k (the crawl local optimum)` / `... at the narrow-region entrance` — contiguous spans of v < 2.5 m/s lasting >= 0.4 s, attributed by context (obstacle within 4 m / corridor narrowing / neither)
- `inside the <w> m tunnel it runs at <v> m/s with the funnel at its minimum` (green; say explicitly when narrowness is *not* what slows it)
- `stops making progress at <%>: oscillates in place (the loiter local optimum)` — 200 steps with < 0.5 m net progress at v > 0.5
- end: `ARRIVED` (green) · `REACHED THE COURSE END (<%>): the final contact is the closed end wall, not an obstacle` (green; open_narrow-family maps end on a wall and cannot be "arrived" — see 1.7) · `COLLISION at <%>: the funnel clipped <a wall obstacle (d m) | the wall in a bend of radius r m | a narrow section (half-width w m) | the corridor wall>` (red) · `STUCK at <%>` (red)

Follower events (follower videos):
- `follower i leaves the funnel at <%>: <reason>` where reason is one of: obstacle squeeze (obstacle < 3 m: "the funnel narrows to b=.. and the outer slot has no room"), bend (|curvature| > 0.03: "the slot on the outside of the turn has farther to travel"), narrow section (half-width < 3.5 m), lag ("the patch is at v m/s and follower i lags behind its slot")
- `all N followers back inside the funnel` (green, after >= 25 steps out)
- `two followers touch (car bodies overlap) at <%>: their planned paths crossed while both dodged inward` (red)

Verdicts: SUCCESS (green) = arrived/reached the end, all followers inside >= 75 % of the time, no contact · PARTIAL (amber) = reached the end but formation held < 75 % or contact · FAILURE (red) = patch did not reach the end. Summary card counts SUCCESS only.

### 1.6 Playback and synchronisation
- **Real time is the default**: one rendered frame every 4 simulation steps (0.04 s) played at 25 fps. Always also export a **2x** cut (every other frame) with `_2x` in the name.
- Side-by-side panels share **one clock from t = 0**. A panel that ends first holds its last frame (with its verdict) while the other continues. Never time-stretch a panel.
- On uniformly scaled maps the course is k times longer at the same car speed; say so on the title card ("1 frame = 0.5 s" if a larger stride is used) rather than hiding it.
- Length guidance: solo 20-40 s; 2-map comparisons ~2 min; 12-map comparisons ~6 min real time (ship the 2x cut alongside).

### 1.7 Evaluation rules that decide what the video shows
- Run the checkpoint **through the path it is known to work in**, and say which in the title:
  - legacy raw-action checkpoints (June ck15M, ck18510000): the *shim* path (absolute steer -> slew-limited steering rate, centred actions). ck18510000 gives 99 % on open_narrow_obs this way and 38 % with raw steering.
  - v1 / v2 / v3 / v4 checkpoints: **raw actions**, the env copy they were trained with (`versions/<name>/envs`), their own config yaml. v3 needs `versions/v3_x5` (width estimator 40 m, lidar 150 m) and the 5x maps.
- Pair the checkpoint with **its own** vecnormalize file (`checkpoint_N_vecnormalize.pkl`; for `final_model` use `final_vecnormalize.pkl`). Never `best_vecnormalize.pkl` / `best_model.zip` without checking timestamps: the two "best" callbacks disagree and in several runs "best" is an early crawl-phase checkpoint.
- Seed 0, spawn at the course start, deterministic actions, 8000-step cap at normal scale (40000 at 5x).
- open_narrow-family maps (`open_narrow`, `open_narrow_obs`, `on_ext`, `on_obs_ext`, their 5x copies) end on a closed wall and pinch below the funnel minimum in the last metres: progress >= 97 % is "reached the course end", never "collision". State this on screen.
- Held-out vs training layouts must be labelled as such on the title card (e.g. "held-out layout 2", "one of v2's TRAINING layouts", "family v2 never saw").

### 1.8 Naming and location
`paper_vids/<subject>_<comparison>_<maps>_<realtime|2x>.mp4`, one `_summary.json` next to each with per-scene outcome, progress, steps and (followers) all-inside %. Working files for the stakeholder series live in `videos/stakeholder_*.mp4`.

### 1.9 QA before sharing
1. every scene's verdict matches its `_summary.json`;
2. no red event on a run that reached the course end unless it is a follower/contact event;
3. the car body and the funnel are visible in the first frame of every scene;
4. the shared clock is identical on both panels in the first frame after the title card;
5. real-time file plays at 1x (a 22 s run is 22 s of video).

---

## 2. Video catalogue

### 2.1 Delivered (2026-09-06 / 07)
| id | file | content |
|---|---|---|
| P1 | `paper_vids/legacy_patch_open_narrow_obs_realtime.mp4` | ck18510000 alone on open_narrow_obs, real time, car body + deformation + narrow-region inset; slow spans attributed to obstacle passes (it runs 8-9 m/s inside the tunnel) |
| P2 | `paper_vids/legacy_patch_vs_patch_plus_N_dmpc_{realtime,2x}.mp4` | ck18510000 alone vs + N DMPC followers (adaptive slots), N = 1..4, on sw_lshape001, open_narrow_obs, on_obs_ext_obs |
| P8 | `paper_vids/v2_patch_vs_patch_plus_N_dmpc_3_obstacle_maps_{realtime,2x}.mp4` | v2 (final_model, raw actions) alone vs + N DMPC followers (adaptive slots), N = 1..4, on sw_lshape001 (held-out L-shape), mf_slalom_g7_5 (family never seen), sw_zigzag001 (family never seen). All 12 arrive; SUCCESS 4/12 (N=1 everywhere + slalom N=2), the rest PARTIAL on formation < 75 % or car-car contact. Opens with the maps slide (template 1). |
| P9 | `paper_vids/working_patch_vs_patch_plus_N_nmpc_open_narrow_obs_{realtime,2x}.mp4` | template remake of `videos/working_patch_open_narrow_obs_nmpc_N1-6.mp4` at real speed (the old cut rendered every 10 steps = 2.5x): ck18510000 (shim) alone vs + N decentralised NMPC followers (ring slots), N = 1..6. All reach the course end (99 %); formation 93/16/21/5/3/2 %, contact from N = 2 up -> SUCCESS 1/6 |
| P11 | `paper_vids/working_patch_vs_patch_plus_N_dmpc_adaptive_open_narrow_obs_{realtime,2x}.mp4` | P9 with DISTRIBUTED MPC + adaptive slots instead of decentralised NMPC + ring: ck18510000 (shim) alone vs + N DMPC followers, N = 1..6, open_narrow_obs, real speed. All reach the end (99 %); formation 96/27/12/12/10/8 %, contact from N = 2 up -> SUCCESS 1/6 (N = 1 at 96 % beats NMPC's 93 %; N >= 2 collapses at the first obstacle (19 %), where the funnel narrows to b = 1.0 m and the outer slots have no room) |
| P12 | `paper_vids/original_N4_vs_x5_N10_dmpc_{realtime,2x}.mp4` | slide "Original scale vs scaled version to hold more No. of agents with 1 + N DMPC agents": LEFT = the N = 4 panel cropped out of P11 (ck18510000 + 4 DMPC, open_narrow_obs, 21.8 s, PARTIAL: 3 of 4 followers held); RIGHT = the user's pre-rendered `videos/v3_x5_dmpc_N10_on_obs_x5.mp4` (v3 + 10 DMPC on on_obs_x5, no trace) paced to real time (1 clip frame = 0.57 s of sim, 87.9 s; ends at 95 % on the last obstacle, 9/10 inside). Built with `paper_crop_pair.py`; the clip's own caption is relabelled DMPC (the runner's frame title still said NMPC). The user asked for the 2x cut with a "2x speed" badge instead of a re-simulation; the right panel still steps 0.57 s of sim per clip frame, so it cannot be as smooth as the left panel without a rerun (`versions/v3_x5_dmpc` = v3 env + dmpc.py is prepared for that) |
| P10 | `paper_vids/legacy_original_vs_v3_x5_scaled_patch_only_{realtime,2x}.mp4` | patch only, side by side: ck18510000 (shim) on open_narrow_obs vs v3 ck17500000 (raw, `versions/v3_x5` env) on on_obs_x5, one shared clock, view 20 m / grid 5 m vs view 100 m / grid 25 m. Message on the cards: the corridor is scaled 5x to hold 10-12 followers, not because the original fails. Legacy REACHED THE END (99 %, 21.8 s); v3 COLLISION at 95 % on the last obstacle pair (87.9 s, funnel at its 10 m floor) |
| S1 | `videos/stakeholder_patch_vs_patch_plus_N_{realtime,2x}.mp4` | v2 alone vs v2 + N followers, N = 1..6, sw_lshape001 + sw_zigzag001 (all 12 arrive; formation 16-92 %) |
| SA | `videos/stakeholder_A_legacy_vs_v2_12_wall_maps_{realtime,2x}.mp4` | legacy (June ck15M) vs v2, patch only, the 12 held-out wall-obstacle maps (legacy 1/12, v2 4/12) |
| SB | `videos/stakeholder_B_legacy_vs_v2_dmpc_maps_{realtime,2x}.mp4` | same on sw_lshape001 + sw_zigzag001 |
| X1 | `videos/ck18510000_side_by_side_open_narrow_obs_vs_uniform_x5.mp4` | original vs uniformly 5x-scaled open_narrow_obs, same checkpoint, view zoomed 5x, grid 5 m vs 25 m (the 5x run fails: car dynamics do not scale) |
| X2 | `videos/v3_x5_checkpoint_17500000_vs_ck18510000_original_side_by_side.mp4` | v3 (trained at 5x) on the 5x map vs ck18510000 on the original |

### 2.2 To create
| id | content | policies | maps | status / notes |
|---|---|---|---|---|
| P3 | legacy vs new policy, patch only, side by side | June ck15M (shim) vs v2 final (raw) | the v2 reel courses minus open_narrow: mf_lshape, sw_lshape000/001/002, lw_lshape150, mf_zigzag, sw_zigzag001, mf_slalom_g7_5 | runs exist for the 4 sw_ maps (from SA); 4 map pairs still to run; composer `paper_pair3_compose.py` written |
| P4 | **one policy for all 13 scenes**: legacy vs the winning checkpoint, patch only | ck18510000 (shim) vs the best of v4a / v4b / v4c | open_narrow_obs + the 12 sw_ layouts | waits for training; candidate must arrive on all 13 first (v4c is the only run whose pool contains open_narrow) |
| P5 | the same winning checkpoint + N followers | winner + DMPC, N = 1..4 (and 6 if it holds) | open_narrow_obs, sw_lshape001, sw_narrowing001, sw_slalom_g7_5002 | after P4; this is the "faster than legacy on the same map" comparison the paper needs |
| P6 | large-formation demo | v3 (5x) or a v4c successor at 5x + N = 10-12 DMPC followers | on_obs_x5 | funnel floor must be raised (b_min ~7 m) for 12 cars; the user's own N = 10 run is the reference |
| P7 | speed/deformation study | ck18510000 vs winner | open_narrow_obs only, 3 panels: alone, +2, +4 | optional; reuse P1 layout with the inset |

### 2.3 Checkpoints referenced
| name | path | run through |
|---|---|---|
| legacy June ck15M | `patch_policy_models/run_20260602_233933/checkpoint_15000000` | shim |
| legacy "working" ck18510000 | `patch_policy_models/run_20260518_151612/checkpoint_18510000` | shim |
| v1 (zig-zag) | `patch_policy_models/run_20260906_181907/final_model` | raw, `versions/v1_zigzag` |
| v2 (L-shape) | `patch_policy_models/run_20260906_181910/final_model` | raw, `versions/v2_lshape` |
| v3 (5x open_narrow_obs) | `patch_policy_models/run_20260906_234432/checkpoint_17500000` | raw, `versions/v3_x5`, 5x maps |
| v4a / v4b / v4c | `run_20260907_105827` / `run_20260907_105831` / `run_20260907_121319` | raw, `versions/v4_multi`; pick late checkpoints by evaluation, not `best_model` |

---

## 3. Tooling — `presentation_code/paper_vids_tools/` (in the repo since 2026-09-07)
| script | role |
|---|---|
| `patch_runner.py` | generates `dmpc_trace.py` / `nmpc_trace.py` from the ring-formation worktree's `dmpc_follower_n.py` / `mpc_follower_native_n.py` (re-run after editing those). Adds `--patch-mode raw|shim`, `--env-config yaml`, `--trace pkl`, `--render none`; the generated scripts chdir to the worktree (`FF_RUNNER_ROOT`) because `envs/dmpc.py` lives only there |
| `run_follower_scenes.py` | launches (map x N) trace runs in parallel: `--tool dmpc|nmpc --policy --patch-mode --env-config --slots --maps --ns --out-dir`; skips existing traces |
| `patch_trace.py` | patch-only trace runner (N = 0) in the same pickle format; `FF_RUNNER_ROOT` picks the env copy (e.g. `versions/v3_x5` for v3's 40 m width estimator) |
| `paper_pair_compose.py` | two INDEPENDENT patch runs side by side (legacy original vs v3 5x), one shared clock, hold-last-frame, per-panel scale (view and grid x scale, event thresholds x scale, car-motion rules unscaled) |
| `paper_crop_pair.py` | one panel cropped out of a template video (left) beside a pre-rendered clip with no trace (right), the clip paced to real time from its known run length; cards and verdicts passed on the command line |
| `paper_dmpc_compose.py` | patch-alone (left) vs patch + N (right) from ONE trace per scene (the patch never sees the followers): HUD, derived narration, verdicts, intro / maps / title / summary cards, realtime + 2x cuts, `_summary.json`. Input: a `scenes.json` list of `{trace, label, family}` |
Traces: `videos/_traces/v2_dmpc/` (v2, 6 maps x N = 1..4 — the 3-map video uses a subset), `videos/_traces/ck18510000_nmpc_on_obs/`, `videos/_traces/ck18510000_dmpc_adaptive_on_obs/`, `videos/_traces/scale_pair/`.
Note: the policies and the MPC solvers run on CPU (SB3 on cpu, IPOPT); the GPUs are idle during video runs, so parallelism = cores.

### 3.1 Older scripts (were in a session scratchpad; not recovered)
| script | role |
|---|---|
| `patch_pair_run.py` | one (policy, map) patch-only run -> mp4 (1 frame / 4 steps) + trace pkl (s, ey, v, a, b, x, y per step) |
| `patch_pair_compose.py`, `paper_pair3_compose.py` | legacy-vs-new side-by-side composer (events, verdicts, cards, streaming writer) |
| `paper_legacy_on_obs.py` | solo video with car body, deformation, narrow-region inset, slow-span attribution |
| `dmpc_trace.py` | patched copy of the worktree's `dmpc_follower_n.py`: `--every 4`, `--out-left` (followers hidden), `--trace` pkl. Needs `PYTHONPATH=<repo>/f1tenth_gym` (the repo-local gym; the test_PPO gym has a different `vehicle_dynamics_st` signature) and cwd = the `ring-formation` worktree |
| `compose_stakeholder.py`, `paper_dmpc_compose.py` | patch-alone vs patch+N composer (follower exit reasons, contact, verdicts) |
These were lost with the scratchpad of the 2026-09-06/07 session; P1/P2/SA/SB cannot be regenerated without rewriting them (the composer above covers the follower comparisons).
