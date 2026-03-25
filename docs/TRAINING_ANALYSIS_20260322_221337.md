# Training Analysis: Session 20260322_221337

**Date**: 2026-03-25
**Session Duration**: ~55 hours (2026-03-22 22:13 to 2026-03-25 11:25)
**Total Episodes**: 24,022
**Level Completions**: 6 (0.025%)

---

## CRITICAL FINDING: The Network Is Not Learning

The single most important finding from this analysis:

```
Loss values: 0 non-zero out of 294,515 total
  *** CRITICAL: ALL LOSS VALUES ARE ZERO - NO LEARNING IS HAPPENING ***

Q-values: 0 non-zero out of 294,515 total
  *** CRITICAL: ALL Q-VALUES ARE ZERO - MODEL IS NOT PRODUCING MEANINGFUL OUTPUTS ***
```

**Every single training step across 294,515 logged rows shows zero loss and zero Q-values.** The neural network weights are never being updated. The entire 55-hour training run produced no actual learning. All observed behavior is the result of epsilon-greedy random exploration, not learned policy.

---

## Performance Regression (Not a Plateau)

The agent's performance **peaked around episodes 6,000-8,000** and has been **declining by ~31%** since:

| Phase | Avg Distance | Avg Reward | Exploration % |
|-------|-------------|------------|---------------|
| Ep 1-2000 | 443.5 | 372.2 | 78.3% |
| Ep 6001-8000 (peak) | 757.3 | 710.3 | 33.9% |
| Ep 22001-24000 (recent) | 485.5 | 445.4 | 6.6% |
| Last 100 episodes | 441.8 | 401.8 | ~6% |

- **Mid-training avg distance**: 690.9
- **Late-training avg distance**: 476.1
- **Delta**: -214.8 (-31.1%)
- **Verdict**: REGRESSION -- performance has declined

This regression pattern is entirely explained by the zero-learning problem. Early high-exploration ratios (78% random in the first bucket, 59% in the first quarter) meant random actions were occasionally producing good trajectories. As epsilon decayed toward 0.06, the agent increasingly relied on its "learned" policy -- which outputs only zeros, causing it to default to a fixed action and perform worse than random.

---

## Action Distribution Collapse

The agent has collapsed to heavily favoring a single action:

| Action | Name | Count | Percentage |
|--------|------|-------|------------|
| 9 | Right+B (Run) | 173,174 | **58.8%** |
| 4 | Right+A+B | 16,677 | 5.7% |
| 7 | Left+A | 14,647 | 5.0% |
| 1 | Right | 12,136 | 4.1% |
| All others | -- | ~68,000 | ~23% combined |

- **Action entropy**: 2.390 / 3.585 (normalized: 0.667)
- A healthy, exploring agent would have entropy closer to 1.0 (uniform distribution)
- This near-59% concentration on action 9 confirms the Q-network is outputting near-identical values for all actions (likely all zeros), with action 9 being selected as the argmax tiebreaker

---

## Death Cause Analysis

| Cause | Count | Percentage |
|-------|-------|------------|
| death | 22,048 | 91.8% |
| stuck_timeout | 1,967 | 8.2% |
| level_complete | 6 | 0.025% |
| timeout | 1 | 0.004% |

- `stuck_timeout` was dominant in the first bucket (1,210/2,000 = 60.5%) when the agent was mostly random and often got stuck going left
- After exploration decayed, the agent moved right consistently (action 9 = Right+B) and hit obstacles/enemies instead, shifting to `death` as the primary cause

---

## Distance Distribution (Multi-Modal)

The distance distribution shows clustering at specific obstacles:

| Range | Count | Pct | Interpretation |
|-------|-------|-----|----------------|
| 300-399 | 9,201 | **38.3%** | Primary death zone -- likely first major obstacle/gap |
| 600-799 | 7,027 | **29.3%** | Second death zone -- another obstacle cluster |
| 1100-1199 | 1,952 | 8.1% | Third barrier |
| 1400-1499 | 1,094 | 4.6% | Fourth barrier |

These clusters suggest the agent hits specific level geometry features (pipes, gaps, enemies) and cannot learn to overcome them because the network is not training.

---

## The 6 Level Completions Were Lucky Random Runs

All 6 completions occurred between episodes 2,151 and 9,155 -- the high-exploration phase:

| Episode | Distance | Reward | Steps |
|---------|----------|--------|-------|
| 2,151 | 3,138 | 3,076.2 | 3,926 |
| 5,145 | 3,138 | 3,086.1 | 2,711 |
| 5,244 | 3,139 | 3,077.3 | 2,457 |
| 6,075 | 3,137 | 3,087.1 | 2,928 |
| 6,890 | 3,137 | 3,051.9 | 3,110 |
| 9,155 | 3,137 | 3,078.6 | 2,329 |

After episode 9,155, the best distance achieved was 3,057 (ep 10,679) and then 3,032 (ep 12,047), both non-completions. No completions in the last 15,000 episodes as exploration fell below ~25%.

---

## Training Pipeline Status

| Metric | Value | Status |
|--------|-------|--------|
| Epsilon | 1.0 -> 0.06 | [OK] Decaying correctly |
| Replay buffer | 0 -> 50,000 (full) | [OK] Filling correctly |
| Learning rate | 0.000250 (constant) | [OK] Static as configured |
| Loss | ALL ZERO | [BROKEN] No gradient updates |
| Q-values | ALL ZERO | [BROKEN] Network outputs are zero |
| Action entropy | 0.667 | [BAD] Policy has collapsed |

---

## Root Cause Investigation Needed

The zero loss / zero Q-value situation points to one of these causes:

1. **Training loop never calls `optimizer.step()`** -- the forward/backward pass or optimizer step may be skipped or guarded by a condition that is never met
2. **Loss computation returns zero** -- the loss function may be receiving identical predicted and target Q-values (both zero), producing zero loss
3. **The model forward pass is broken** -- the network architecture may have a bug causing all outputs to be zero (e.g., bad weight initialization, dead ReLU, wrong input shape)
4. **Replay buffer sampling issue** -- the buffer may not be returning valid training batches, causing the training step to be skipped entirely
5. **Target network issue** -- if the target network and online network both produce zeros, the TD-target would be zero and loss would be zero

The next step should be to inspect the training loop in `python/training/trainer.py` and the agent's `learn()` method in `python/agents/dqn_agent.py` to determine why no gradient updates are occurring despite 294,515 logged training steps.

---

## Fixes Applied (2026-03-25)

### Bug 1: Silent training failures in `_end_episode()` -- `trainer.py:553`

The training loop inside `_end_episode()` had **no try/except** around `agent.train_step()`. Any exception (shape mismatch, CUDA error, C51 projection issue) would propagate up and silently abort the episode end handler without any log output. This made it impossible to diagnose training failures.

**Fix**: Wrapped each `train_step()` call in a try/except that logs the full exception with traceback. Added counters for success/empty/error outcomes and a summary log line per qualifying episode.

### Bug 2: `training_state.training_phase` never updated from "warmup" -- `trainer.py:1183`

`_update_training_phase()` correctly set `self.training_phase` (the local enum) but never synced it to `self.training_state.training_phase` (the JSON-persisted string in `training_state.json`). The state file always showed `"warmup"` even after 24,000 episodes.

**Fix**: Added sync line `self.training_state.training_phase = self.training_phase.value` and phase transition logging.

### Bug 3: Episode CSV dual-write corruption -- `trainer.py:210`

Both `EpisodeManager` and `CSVLogger.log_episode_summary()` wrote to the same `episodes_{session_id}.csv` file with **completely different column schemas**. EpisodeManager wrote ~30 columns (reward components, death causes, etc.) while CSVLogger wrote 21 columns (Q-values, exploration stats, etc.). This produced alternating malformed rows where values from one schema appeared in the wrong columns of the other.

**Fix**: Changed EpisodeManager to write to `episode_detail_{session_id}.csv` instead. CSVLogger retains `episodes_{session_id}.csv` as the primary analysis-friendly format.

### Bug 4: No diagnostic logging in `train_step()` -- `dqn_agent.py:336`

`train_step()` returned `{}` silently when the buffer wasn't ready, and returned metrics without any logging. There was no way to tell from logs whether training was happening, what loss values were, or if the C51 distributional path was producing valid outputs.

**Fix**: Added:
- Buffer-not-ready warning (periodic)
- Loss/TD-error diagnostic logging for first 5 steps and every 500th step
- Q-value sample logging for first 5 steps
- Zero-loss warning when `loss=0.0` (periodic)

### Next Steps

The diagnostic logging will now reveal on the next training run whether `train_step()` is actually being called and what values it produces. If training still shows zero loss, the next investigation targets are:
1. C51 distributional loss in `_compute_distributional_loss()` -- shape mismatches or numerical issues
2. Mixed precision (autocast) interaction with C51 log-softmax
3. Replay buffer sampling returning all-zero tensors

---

## Summary

| Metric | Value |
|--------|-------|
| Episodes trained | 24,022 |
| Hours spent | 55 |
| Actual learning | **NONE** |
| Performance trend | **Declining (-31%)** |
| Root cause | Zero loss / zero Q-values -- training loop failures were silently swallowed |
| Priority | **P0 -- Must fix before any further training** |
| Status | **4 bugs fixed, diagnostic logging added -- ready for verification run** |
