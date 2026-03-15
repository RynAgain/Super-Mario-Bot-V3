# Feature Tracker -- Super Mario Bot V3

A prioritized list of work items. Items are ranked by impact on getting the bot
to actually learn. Work top-down -- each section builds on the one above it.

---

## Phase 0: Recently Fixed (v3.1)

- [x] Fix action mapping mismatch between Python trainer and Lua `ACTION_MAPPING`
- [x] Fix binary payload double-header parsing in `frame_capture.py`
- [x] Fix struct format mismatch in `comm_manager.py` (i8 not i16 velocities)
- [x] Disable WebSocket library-level ping/pong (Lua client can't answer them)
- [x] Send integer `action` ID alongside button dict for reliable Lua parsing
- [x] Fix undefined `payload` variable in Lua ping frame handler
- [x] Fix terminal state detection race condition in `RewardCalculator`
- [x] Add `websocket:` section to `training_config.yaml`

---

## Phase 1: Make It Run (do these first -- nothing learns until these work)

> **Goal:** Python starts, Lua connects, game state flows, actions execute, episodes reset.

- [ ] **P0** -- Run an end-to-end smoke test: start Python, load Lua in FCEUX, confirm mario_x increases in logs
- [x] **P0** -- Verify save state slot 10 exists and loads World 1-1 correctly (create one manually if missing)
- [x] **P0** -- Audit memory addresses against NESDev wiki SMB RAM map:
  - Fixed: Score digits now at `0x07DD-0x07E2` (6 individual decimal digits)
  - Fixed: Timer digits at `0x07F8-0x07FA` (3 individual decimal digits, not binary word)
  - Fixed: `MARIO_POWER` moved to `0x0756` (PlayerStatus)
  - Fixed: `MARIO_VELOCITY_X` moved to `0x0057`, `MARIO_VELOCITY_Y` to `0x009F`
  - Fixed: `ONEUP_FLAG` moved to `0x075D` (was overlapping END_OF_LEVEL_FLAG)
- [x] **P1** -- Wrap `GPUtil` import in try/except in `training_utils.py` (crashes if not installed)
- [x] **P1** -- Fix `WeightInitializer.initialize_model()` using `cls` instead of `self` in `@classmethod`

---

## Phase 2: Make It Learn (do these once Phase 1 runs clean)

> **Goal:** Mario consistently moves right and improves over 500+ episodes.

- [x] **P0** -- Reduce warmup from 1000 episodes to 50 (was wasting hours doing random actions with zero learning)
- [x] **P0** -- Tune epsilon decay: changed from per-step `0.9995` to per-episode `0.998` (~1500 episodes to reach 0.05)
- [x] **P0** -- Normalize rewards: clip to [-1, +1] range for Q-value stability
- [x] **P1** -- Add frame skipping: act every 4 frames, repeat action on skipped frames (4x speedup)
- [x] **P1** -- Switch to soft target updates (Polyak tau=0.005) -- blends weights every training step
- [ ] **P1** -- Add TensorBoard logging for loss, reward, Q-values, epsilon curves
- [ ] **P2** -- Log action distribution per episode to detect collapsed exploration
- [ ] **P2** -- Implement gradient accumulation for larger effective batch sizes

---

## Phase 3: Make It Reliable (do these once learning is confirmed)

> **Goal:** Training runs for hours without crashes or connection drops.

- [ ] **P0** -- Replace hand-rolled Lua WebSocket with a proper library (e.g. `lua-websockets`)
- [ ] **P1** -- Switch to JSON-only protocol (drop binary) -- saves 100 bytes per frame but eliminates 90% of parsing bugs
- [ ] **P1** -- Add protocol version negotiation so Lua/Python reject incompatible versions immediately
- [ ] **P1** -- Frame capture is never started in training loop -- either start it or send pixels from Lua
- [ ] **P2** -- Deduplicate `FramePreprocessor` (exists in both `preprocessing.py` and `frame_capture.py`)
- [ ] **P2** -- Auto-create World 1-1 save state on first run

---

## Phase 4: Make It Fast (do these once stability is solid)

> **Goal:** Train 10x faster to iterate on reward design.

- [ ] **P1** -- Send screen pixels from Lua via binary payload instead of Win32 GDI window capture
- [ ] **P1** -- Implement n-step returns (n=3) for faster value propagation
- [ ] **P1** -- Run multiple FCEUX instances in parallel for faster experience collection
- [ ] **P2** -- Enable prioritized experience replay (already implemented but disabled)
- [ ] **P2** -- Try Noisy Networks as alternative to epsilon-greedy
- [ ] **P2** -- Experiment with Rainbow DQN (combines 6 improvements in one)

---

## Phase 5: Make It Smart (research / stretch goals)

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
- [ ] Assign a dedicated save state slot per level (e.g. slot 10 = 1-1, slot 11 = 1-2, slot 12 = 1-3, slot 13 = 1-4, slot 14 = 2-1, ...)
- [ ] On episode reset, load the save state for whichever level the agent is currently training on
- [ ] Store save state metadata (world, level, power state, lives) in a JSON manifest

### Per-Level Models
- [ ] Maintain a separate checkpoint directory per level: `checkpoints/world1-1/`, `checkpoints/world1-2/`, etc.
- [ ] When starting training on a new level, initialize the model from the best checkpoint of the previous level (transfer learning)
- [ ] This lets the agent skip re-learning basic movement and jumping on each new level
- [ ] Track per-level metrics independently: completion rate, best distance, average reward

### Auto-Curriculum Progression
- [ ] Define a completion threshold per level (e.g. 80% completion rate over last 100 episodes)
- [ ] When the threshold is met, automatically advance to the next level
- [ ] If performance drops below a regression threshold (e.g. 40%), fall back to the previous level for refresher training
- [ ] Support manual override to force training on a specific level

### Level-Specific Reward Tuning
- [ ] Allow per-level reward configuration (e.g. underwater levels might weight survival higher)
- [ ] Adjust level length constants per level for accurate progress calculation
- [ ] Add level-specific hazard detection (e.g. lava in castle levels, water in 2-2)

### Model Merging / Ensemble (stretch)
- [ ] Experiment with distilling all per-level models into a single universal model
- [ ] Try an ensemble approach where a meta-controller selects which level-specific model to use
- [ ] Evaluate whether a single model trained across all levels outperforms the per-level approach

---

## Cleanup / Tech Debt (do whenever convenient)

- [x] Clean up `requirements.txt` -- removed 12 unused packages (`triton`, `asyncio-mqtt`, `cupy-cuda12x`, `seaborn`, `scipy`, `numba`, `wandb`, `colorama`, etc.)
- [x] Remove `json.lua` (unused -- Lua script has a built-in encoder/decoder, no files reference it)
- [x] Clean up the `2.0.0` file in project root
- [x] Delete incompatible old checkpoints in `checkpoints/` (from Sept 2025, wrong model architecture)
- [x] Rename duplicate `FramePreprocessor` to `GameFramePreprocessor` in `frame_capture.py`
- [x] Add `pyproject.toml` for modern Python packaging (v3.1.0, with optional deps for gpu/win/dev)

---

## Priority Key

| Tag | Meaning |
|-----|---------|
| **P0** | Blocking -- nothing works right until this is fixed |
| **P1** | High value -- directly improves training quality or reliability |
| **P2** | Nice to have -- improves speed, code quality, or developer experience |
