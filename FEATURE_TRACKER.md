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
- [x] **P2** -- Rainbow DQN: 5/6 components done (Double + Dueling + PER + N-step + NoisyNet). Missing: C51

---

## Phase 5: Make It Smart

> **Goal:** Complete World 1-1 reliably, then generalize to other levels.

- [ ] Intrinsic motivation (curiosity) for discovering hidden blocks and warp zones
- [ ] Learn from human speedrun demonstrations (inverse RL or behavioral cloning)
- [ ] Model-based RL (Dreamer/MuZero) for planning ahead
- [ ] "Play mode" where the trained agent plays in real-time with live visualization

---

## Phase 6: Per-Level Models with Auto-Progression

> **Goal:** Detect new levels automatically, create dedicated save states and models
> for each level, and chain them together so later levels inherit the skills learned
> on earlier ones.

### Level Detection
- [ ] Monitor `WORLD` (0x075F) and `LEVEL` (0x0760) memory addresses every frame
- [ ] Detect level transitions: when (world, level) changes, the agent has reached a new stage
- [ ] Log every level transition with timestamp, episode, and score for analysis

### Per-Level Save States
- [ ] When a new level is detected for the first time, auto-create a save state at the start of that level
- [ ] Assign a dedicated save state slot per level (e.g. slot 10 = 1-1, slot 11 = 1-2, etc.)
- [ ] On episode reset, load the save state for whichever level the agent is currently training on
- [ ] Store save state metadata (world, level, power state, lives) in a JSON manifest

### Per-Level Models
- [ ] Maintain a separate checkpoint directory per level: `checkpoints/world1-1/`, etc.
- [ ] When starting training on a new level, initialize from the best checkpoint of the previous level
- [ ] Track per-level metrics independently: completion rate, best distance, average reward

### Auto-Curriculum Progression
- [ ] Define a completion threshold per level (e.g. 80% completion rate over last 100 episodes)
- [ ] When the threshold is met, automatically advance to the next level
- [ ] If performance drops below a regression threshold (e.g. 40%), fall back for refresher training
- [ ] Support manual override to force training on a specific level

### Level-Specific Reward Tuning
- [ ] Allow per-level reward configuration (e.g. underwater levels weight survival higher)
- [ ] Adjust level length constants per level for accurate progress calculation
- [ ] Add level-specific hazard detection (e.g. lava in castle levels, water in 2-2)

### Model Merging / Ensemble (stretch)
- [ ] Experiment with distilling all per-level models into a single universal model
- [ ] Try an ensemble approach where a meta-controller selects which level-specific model to use

---

## Priority Key

| Tag | Meaning |
|-----|---------|
| **P0** | Blocking -- nothing works right until this is fixed |
| **P1** | High value -- directly improves training quality or reliability |
| **P2** | Nice to have -- improves speed, code quality, or developer experience |
