# Feature Tracker -- Super Mario Bot V3

A prioritized list of work items. Items are ranked by impact on getting the bot
to actually learn. Work top-down -- each section builds on the one above it.

---

## Phase 0: Critical Bugfix Pass (v3.1)

- [x] Fix action mapping mismatch between Python trainer and Lua `ACTION_MAPPING`
- [x] Fix binary payload double-header parsing in `frame_capture.py`
- [x] Fix struct format mismatch in `comm_manager.py` (i8 not i16 velocities)
- [x] Disable WebSocket library-level ping/pong (Lua client can't answer them)
- [x] Send integer `action` ID alongside button dict for reliable Lua parsing
- [x] Fix undefined `payload` variable in Lua ping frame handler
- [x] Fix terminal state detection race condition in `RewardCalculator`
- [x] Add `websocket:` section to `training_config.yaml`

---

## Phase 1: Make It Run

- [x] Run end-to-end smoke test: Python + Lua connect, mario_x increases in logs
- [x] Verify save state slot 10 loads World 1-1 correctly
- [x] Audit memory addresses against NESDev wiki SMB RAM map:
  - Score digits: `0x07DD-0x07E2` (6 individual decimal digits)
  - Timer digits: `0x07F8-0x07FA` (3 individual decimal digits)
  - `MARIO_POWER`: `0x0756`, `MARIO_VELOCITY_X`: `0x0057`, `MARIO_VELOCITY_Y`: `0x009F`
  - `ONEUP_FLAG`: `0x075D` (was overlapping END_OF_LEVEL_FLAG)
- [x] Wrap `GPUtil` import in try/except
- [x] Fix `WeightInitializer` `@classmethod` using `cls` instead of `self`

---

## Phase 2: Make It Learn

- [x] Reduce warmup from 1000 to 50 episodes
- [x] Epsilon decay per-episode (0.998) instead of per-step (0.9995)
- [x] Reward clipping to [-1, +1] for Q-value stability
- [x] Frame skipping: act every 4 frames, repeat action on skipped frames
- [x] Soft target updates (Polyak tau=0.005)
- [x] Add TensorBoard logging for loss, reward, Q-values, epsilon curves
- [x] Log action distribution per episode to detect collapsed exploration (histogram + entropy + unique count)
- [x] Implement gradient accumulation for larger effective batch sizes (default 4x = 128 effective batch)

---

## Phase 2.5: Training Log Fixes (from first live run analysis)

- [x] Disable curriculum `epsilon_override` (was locking epsilon at 0.8 for 10k episodes)
- [x] Increase `stuck_timeout` from 600 to 1800 frames (30s -- Mario needs time for pipe jumps)
- [x] Fix frame desync at episode boundaries (Lua resets frame_id to 0, now detected as normal)
- [x] Fix episode triple-counting (trainer, Lua event, and game state handler all created episodes)
- [x] Fix false death on episode start: initial `lives=3` vs SMB `lives=2` (displayed-1) triggered -50 penalty every first frame
- [x] Fix NoisyNet warmup directional lock: force uniform random actions during warmup episodes
- [x] Fix death not ending episodes: Lua now sends terminal game_state frame BEFORE save state reset; Python `_handle_episode_event` marks episode as FAILED/COMPLETED/TERMINATED

---

## Phase 2.75: Frame Capture Overhaul

- [x] **CRITICAL**: Frame capture was never started -- DQN was learning from all-zero 84x84x4 frames
- [x] Add Lua-side screen capture using `gui.gdscreenshot()` (no window visibility needed)
- [x] Send screen data as binary message type `0x02` (1-byte type + 4-byte frame_id + GD data)
- [x] Python GD format decoder: ARGB -> grayscale -> resize 84x84 -> normalize [0,1]
- [x] Lua captures every 4 frames (matches Python frame_skip) to limit bandwidth
- [x] Capture priority: Lua frames > Win32 GDI > zero frames
- [x] State vector: replaced 2 wasted zero slots with `mario_x_vel` and `mario_y_vel`
- [x] Start/stop capture in trainer lifecycle
- [x] WebSocket buffer increased to 512KB for screenshot data

---

## Cleanup / Tech Debt (completed)

- [x] Clean up `requirements.txt` (removed 12 unused packages)
- [x] Remove `json.lua`, `2.0.0` file, old checkpoints
- [x] Rename duplicate `FramePreprocessor` to `GameFramePreprocessor`
- [x] Add `pyproject.toml` (v3.1.0)
- [x] Move 10 test files to `tests/`, 5 scripts to `scripts/`, 6 docs to `docs/`
- [x] Delete stale validation artifacts
- [x] Remove `setup.py`, `setup_minimal.py`, `MANIFEST.in`, egg-info
- [x] Rewrite `README.md` with accurate architecture, action space, parameters
- [x] Clear stale logs/ (15 CSV files), checkpoints/ (3 files), lua/logs/ (2 files) from failed overnight run
- [x] Create `.gitignore` (checkpoints/*.pt, rotation backups, ROMs, save states, __pycache__)
- [x] Add `.gitkeep` to empty `logs/`, `checkpoints/`, `lua/logs/` directories

---

## Phase 2.9: Training Pipeline Bugfixes (2026-03-25)

> **Status:** Complete. Training confirmed working in session `20260325_182117`
> (non-zero loss, Q-values, improving performance).

- [x] Silent training failure in `_end_episode()` -- no try/except around `train_step()` meant training errors were swallowed silently
- [x] `training_state.training_phase` stuck on "warmup" forever -- phase transition logic never fired
- [x] Episode CSV dual-write corruption -- `EpisodeManager` + `CSVLogger` both writing to same CSV file, causing interleaved/corrupt rows
- [x] Missing diagnostic logging in `train_step()` -- no visibility into whether training was actually running
- [x] `EpisodeManager` logger init order bug -- logger used before initialization complete

---

## Phase 2.95: Plateau-Breaking Reward Shaping (2026-03-27)

> **Status:** Deployed. Addresses plateau at ~720 avg distance (pits at x=450 and x=900).
> Session `20260325_182117` reached 3 completions but avg distance stagnated.

- [x] Airborne forward bonus -- +0.05/frame for forward movement while jumping; landing bonus for long high jumps (>20px forward, >10px height)
- [x] Pit clear bonus -- +5.0 one-time reward per known pit crossed (4 pit zones defined in `WORLD_1_1_PITS`)
- [x] Enemy kill bonus -- +0.5 per estimated kill (score delta >= 100 points = 1 kill)
- [x] Replay-on-filter -- filtered episodes now train on existing replay buffer instead of wasting wall-clock time
- [x] PER alpha raised from 0.6 to 0.75 for more aggressive prioritization of high-TD-error transitions (completions, pit crossings)

---

## Phase 3: Make It Reliable

> **Goal:** Training runs for hours without crashes or connection drops.

- [x] **P0** -- Replace hand-rolled Lua WebSocket -- N/A: FCEUX Lua 5.1 has no WS library; hand-rolled is only option. Hardened with error throttling instead.
- [x] **P1** -- Switch to JSON-only protocol (drop binary game state) -- Lua sends `type:"game_state"` JSON, Python handles via `_handle_json_game_state`. Binary fallback kept for backward compat. Screen frames stay binary.
- [x] **P1** -- Protocol version negotiation: Lua checks `init_ack.protocol_version` and warns on mismatch
- [ ] **P2** -- Auto-create save state on first run (skipped for now)
- [x] **P1** -- Rotating log file handler (50 MB x 3 backups = 200 MB cap) in `main.py`
- [x] **P1** -- Error rate-limiting in `trainer.py`: burst of 3 then 1 log per 30s per error category
- [x] **P1** -- Error rate-limiting in `websocket_server.py`: same throttle for message processing errors
- [x] **P2** -- CSV write decimation in `csv_logger.py`: training steps logged every 10th frame
- [x] **P2** -- CSV file size cap (200 MB per file) -- writes silently dropped when exceeded
- [x] **P2** -- Sync quality CSV throttled to every 100 steps instead of every frame
- [x] **P2** -- Tensor shape mismatch errors throttled (were logging every frame)

---

## Phase 4: Make It Fast

> **Goal:** Train 10x faster to iterate on reward design.

- [x] **P1** -- N-step returns (n=3): buffer N transitions, compute discounted return, bootstrap with gamma^n
- [ ] **P1** -- Run multiple FCEUX instances in parallel (not possible, too resource intense.)
- [x] **P2** -- Prioritized experience replay enabled (`prioritized_replay: true`)
- [x] **P2** -- NoisyNet: factorized Gaussian noise in FC layers, replaces epsilon-greedy, self-annealing
- [x] **P2** -- Rainbow DQN: 6/6 components complete (Double + Dueling + PER + N-step + NoisyNet + C51)

---

## Phase 5: Make It Smart

> **Goal:** Complete World 1-1 reliably, then generalize to other levels.

- [ ] Intrinsic motivation (curiosity) for discovering hidden blocks and warp zones
- [D] Learn from human speedrun demonstrations (inverse RL or behavioral cloning) (Deferred)
- [D] Model-based RL (Dreamer/MuZero) for planning ahead (deferred to another project)

### Play Mode with Live Visualization

> CLI: `python python/main.py play --checkpoint checkpoints/models/1-1_master.pth`
>
> The agent plays in real-time at normal NES speed with an OSD overlay
> showing what the neural network is "thinking." The overlay uses FCEUX's
> `gui.text()` / `gui.drawbox()` / `gui.drawline()` which draw on a
> **separate display layer** -- they do NOT modify the NES framebuffer, so
> the model's CNN input (`gui.gdscreenshot()`) is completely unaffected.

#### Core Play Mode
- [ ] New `play` command in `python/main.py` (load checkpoint, set eval mode, epsilon=0)
- [ ] `model.eval()` disables dropout and NoisyNet noise (deterministic policy)
- [ ] Skip all training, replay buffer, reward calculation, CSV logging
- [ ] Lua: `emu.speedmode("normal")` for real-time 60fps playback
- [ ] Reduced frame skip (1-2 instead of 4) for smoother movement

#### Tier 1: Action Q-Value Bar Chart (OSD)
- [ ] Python sends all 12 Q-values alongside action in WebSocket response
- [ ] Lua draws 12 horizontal bars at bottom of screen, one per action
- [ ] Bar width proportional to Q-value (normalized to max)
- [ ] Selected action highlighted green, others gray
- [ ] Action names labeled: NOOP, R, L, J, R+J, L+J, Run, R+R, L+R, R+J+R, L+J+R, Down
- [ ] Confidence display: `max_Q / sum_Q` as percentage

#### Tier 2: State Info Panel (OSD)
- [ ] Top-right corner: episode number, X position, best distance
- [ ] Current Q-max value, chosen action name
- [ ] Model filename, current level (world-level)
- [ ] Epsilon value (should be 0.0 in play mode)

#### Tier 3: Network Activation Visualization (OSD)
- [ ] Simplified layer diagram (not 4.5M connections -- summarized activations)
- [ ] 3 conv layers shown as colored bar strips (avg activation per filter)
- [ ] Fusion layer as a heatmap band
- [ ] Value stream: single bar showing V(s) estimate
- [ ] Advantage stream: 12 bars (same as Q-value chart but separated)
- [ ] Connections between layer strips with opacity proportional to activation magnitude
- [ ] Lit green for positive activations, red for negative (MarI/O style)

#### Tier 4: Saliency Map (OSD, advanced)
- [ ] Gradient-based saliency: which input pixels matter most for chosen action
- [ ] Overlay translucent heatmap on game screen (red = high importance)
- [ ] Shows what the model is "looking at" (enemy ahead, gap below, etc.)
- [ ] Computed in Python, sent as compressed data to Lua for rendering
- [ ] ~20ms overhead per frame -- may need to run at 30fps instead of 60

#### Protocol Extension
- [ ] Action response includes `q_values` array, `confidence`, `selected_action_name`
- [ ] Optional: `layer_activations` array for Tier 3 visualization
- [ ] Optional: `saliency_data` compressed grid for Tier 4

---

## Phase 6: Level Progression System

> **Goal:** Detect new levels automatically, save states per level, master each
> level via completion streaks, and transfer learned skills to the next level.
>
> **Design doc:** [`plans/level-progression-system.md`](plans/level-progression-system.md)

### Save State Bridge
- [ ] Add `save_state` and `load_state` commands to `handle_training_control()` in `lua/mario_ai.lua`
- [ ] Add `send_save_state(slot)` and `send_load_state(slot)` to `python/communication/websocket_server.py`
- [ ] Slot mapping: slot 10 = 1-1 (default), slots 1-9 for discovered levels
- [ ] Protocol: JSON messages + ack responses over existing WebSocket

### Level Detection
- [ ] Dual confirmation: level byte changed AND timer transitions 400 -> 399
- [ ] Lua reads `0x075F` (world), `0x075C` (level), `0x07F8` (timer hundreds)
- [ ] On detection: auto-save state to designated slot, send `level_transition` event to Python
- [ ] Handle edge cases: warp zones, underground sub-areas, game over screen

### Completion Streak Tracker
- [ ] Track consecutive level completions in `LevelManager`
- [ ] Promotion threshold: 20 consecutive completions (configurable)
- [ ] Streak resets on any death
- [ ] Log streak progress to `logs/streak_{session_id}.csv`

### Transfer Learning on Promotion
- [ ] Freeze and save master model as `checkpoints/models/{world}-{level}_master.pth`
- [ ] Clone weights to initialize next level's model (CNN features transfer)
- [ ] Reset epsilon to 0.5 for partial re-exploration
- [ ] Reset NoisyNet sigma parameters to initial values
- [ ] Clear replay buffer (old level data not useful)
- [ ] Optionally reset optimizer state (Adam momentum)

### New Module: `python/training/level_manager.py`
- [ ] `LevelManager` class: state machine for current level, promotion logic
- [ ] `StreakTracker` class: consecutive completion counter
- [ ] Save state slot registry persisted to `checkpoints/level_registry.json`
- [ ] Model checkpoint registry: tracks mastered models per level
- [ ] Configuration from `config/training_config.yaml` under `level_progression:`

### Stretch Goals
- [ ] Per-level reward tuning (underwater levels weight survival higher)
- [ ] Regression detection: fall back to previous level if performance drops
- [ ] Model distillation: merge per-level models into a single universal model

---

## Priority Key

| Tag | Meaning |
|-----|---------|
| **P0** | Blocking -- nothing works right until this is fixed |
| **P1** | High value -- directly improves training quality or reliability |
| **P2** | Nice to have -- improves speed, code quality, or developer experience |
