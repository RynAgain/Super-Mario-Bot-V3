# Training Log Analysis: Session 20260317_225242

**Analyzed**: 2026-03-19  
**Training Period**: 2026-03-17 22:52 --> 2026-03-19 11:34 (~36.7 hours)  
**Episodes Completed**: 8,594  
**Training Step Rows**: 180,948  

---

## Executive Summary

The network **is learning** -- loss and Q-values are non-zero and reasonable after the 50-episode warmup phase. However, performance has **plateaued around episode 4000** with no meaningful improvement through episode 8594. The agent consistently reaches x=600-1400 but cannot reliably advance further. Several structural issues are throttling learning efficiency.

---

## 1. Loss & Q-Value Trends (The Network IS Training)

The earlier observation of loss=0.0 / q_value_mean=0.0 was limited to the **warmup phase** (episodes 1-50). During warmup, `trainer.py:765` skips `agent.train_step()` entirely, so zeros are logged.

| Training Phase | Episodes | Loss | Q-Value Mean | Epsilon |
|---|---|---|---|---|
| Warmup | 1-50 | 0.0 | 0.0 | 1.0 -> 0.905 |
| Early training | 50-2500 | 0.05-0.8 | growing to ~8.0 | 0.905 -> 0.01 |
| Mid training (~ep 3564) | 2500-5000 | 0.05-0.55 | 7.5-8.5 | 0.01 (floor) |
| Late training (~ep 7990) | 5000-8594 | 0.05-0.96 | 7.0-8.1 | 0.01 (floor) |

**Insight**: Q-values stabilized around 7.5-8.0 and show no upward trend from episode 3000 onward. The network learned a value function but is not improving it. Loss variance is increasing slightly in late training (occasional spikes to 0.8-0.96), suggesting the replay buffer may contain contradictory experiences.

---

## 2. Episode Performance Analysis

### 2.1 Distance Progression (mario_x_max)

Performance across the full run, sampled from episodes log:

| Episode Range | Typical mario_x | Best mario_x | Common Death Position |
|---|---|---|---|
| 1-1000 | 300-900 | 3267 (100%!) | x=272-600 |
| 1000-4000 | 400-1400 | 1958 (62%) | x=300-700 |
| 4000-6000 | 300-1500 | 2023 (64%) | x=272-900 |
| 6000-8000 | 300-1500 | 2473 (78%) | x=272-900 |
| 8000-8594 | 300-1500 | 2027 (64%) | x=272-900 |

**Insight**: There is no clear improvement trend after episode ~2000. The agent learned early behaviors during epsilon decay (episodes 50-2500) but has stagnated since epsilon hit the floor. Rare outlier episodes reach x=2000+ but these are not becoming more frequent.

### 2.2 Death Cause Distribution

Two dominant failure modes:

1. **`death` (enemy/pit)** -- ~80% of episodes. Mario dies at recurring positions suggesting specific obstacles it cannot navigate.
2. **`stuck_timeout`** -- ~15% of episodes. Mario reaches a position (commonly x=722-723 or x=898) and loops indefinitely until the 30-second timeout.

### 2.3 The Stuck Timeout Problem (Critical)

Episodes ending in `stuck_timeout` are severely damaging training:

- Each stuck episode lasts **36-65 seconds** vs 3-15s for normal episodes
- Mario oscillates at x=722 (likely a pipe) or x=898 (likely a gap)
- During oscillation, **hundreds of zero-reward transitions flood the replay buffer**
- These episodes generate 2000-3900 steps of useless data each
- Example: episodes 8008, 8016, 8017, 8043, 8051, 8074, 8075, 8087, 8089 -- ALL stuck at x=722-723

**Impact calculation**: ~15% of 8594 episodes = ~1289 stuck episodes x ~45s average = **~16 hours of wasted training** (44% of total runtime). These episodes also corrupt the replay buffer with thousands of zero-reward "go nowhere" experiences.

### 2.4 Short Death Episodes

Many episodes die almost immediately at x=270-316 (1-3 actions after frame skip):

- These episodes last 2.5-3.5 seconds
- They represent Mario dying at the very start of 1-1
- At epsilon=0.01, this means the **learned policy itself** sometimes walks Mario into the first Goomba

---

## 3. Exploration Analysis

### 3.1 Epsilon Decay Was Too Fast

With `epsilon_decay=0.998` per episode:
- Episode 50 (warmup end): epsilon = 0.905
- Episode 500: epsilon = 0.405  
- Episode 1000: epsilon = 0.135
- Episode 2300: epsilon = 0.01 (floor reached)

This means **6,294 out of 8,594 episodes (73%)** ran at minimum exploration (epsilon=0.01). Combined with the NoisyNet 5% floor, total random action rate is ~5-6%.

### 3.2 NoisyNet Sigma Collapse Risk

With `noisy_networks: true`, the NoisyNet layers provide state-dependent exploration. However, the code includes a `noisy_epsilon_floor = 0.05` in `dqn_agent.py:239` as insurance against sigma collapse. This is good, but if NoisyNet sigmas have collapsed, the agent is effectively running at 5% random exploration for 6000+ episodes -- insufficient for discovering new strategies.

---

## 4. Architecture & Hyperparameter Issues

### 4.1 Replay Buffer Too Small

`replay_buffer_size: 20000` is critically undersized for Rainbow DQN:

- At frame_skip=4, each episode generates ~25-300 meaningful transitions (after n-step)
- 20,000 / 150 (avg transitions/episode) = buffer holds only ~133 episodes
- With episodes lasting 3-65 seconds, the buffer turns over every **~30 minutes**
- Good experiences from rare long runs are quickly overwritten by short death episodes
- Standard Rainbow papers use 1M buffer; even 100K would be a 5x improvement

### 4.2 C51 Support Range May Be Wrong

```yaml
v_min: -10.0
v_max: 10.0
```

With `reward_clip=10.0` and `n_step=3` with `gamma=0.99`:
- Maximum n-step return: 10 + 0.99*10 + 0.99^2*10 = 29.7
- Maximum discounted episode return: could reach 100+ for long episodes

The support range [-10, 10] may be compressing all returns above 10 into the last atom, losing discriminative power for good episodes. The v_max should at minimum be **30** to cover n-step returns, ideally higher.

### 4.3 Reward Clip vs Episode Rewards Mismatch

Per-step `reward_clip=10.0` but episode total rewards range from 30 to 2600+. The C51 support is on per-step Q-values (discounted sums), not per-step rewards, so the support needs to cover the discounted return -- not just one step's reward.

### 4.4 Soft Target Updates

`tau: 0.005` means continuous soft updates. This is fine but combined with the small replay buffer, the target network tracks the Q-network too closely, potentially creating instability.

---

## 5. Frame Desync Errors

From the debug_events log (8,575 entries), **every single event** is:
```
Lua error [FRAME_DESYNC]: Severe frame ID mismatch detected
```

These desyncs occur throughout the entire training run. While the sync_quality log shows the system recovers (sync delay stays at 3-40ms), the persistent desync errors indicate the Lua and Python frame counters are fundamentally misaligned. This may cause:
- Actions being applied to wrong frames
- Reward attribution errors
- Occasional corrupted state transitions in the replay buffer

---

## 6. System Performance

- **FPS**: Stable at 60.0 (emulator-locked)
- **Processing time**: 28-45ms per decision step
- **GPU memory**: ~5.8GB (stable)
- **GPU utilization**: 15-20% (underutilized -- batch_size=32 is small)
- **RAM**: ~680MB (stable)

---

## 7. Recommended Fixes (Priority Order)

### P0: Fix Stuck Timeout (largest impact)

Add a **stuck detection penalty** that triggers much faster than the current 30-second timeout. If mario_x hasn't increased by >10 pixels in the last 50 steps, apply a negative reward and end the episode.

```python
# In episode_manager or reward_calculator
if steps_without_progress > 50:
    end_episode(reason="stuck_timeout", penalty=-5.0)
```

Reduce the existing stuck timeout from 30s to 10s at most.

### P1: Increase Replay Buffer Size

Change from 20,000 to at least 100,000 (ideally 200,000):

```yaml
replay_buffer_size: 200000
```

This preserves good experiences longer and provides more diverse training data. Memory cost is manageable (~2-3GB additional).

### P2: Fix C51 Support Range

Expand the distributional support to cover actual return ranges:

```yaml
v_min: -30.0
v_max: 50.0
```

This lets the network distinguish between mediocre episodes (return ~5) and great episodes (return ~30+).

### P3: Slow Down Epsilon Decay

Change to reach floor around episode 8000-10000 instead of 2300:

```yaml
epsilon_decay: 0.9995  # reaches 0.01 at ~9200 episodes
```

Or use linear decay over a defined range:
```yaml
epsilon_decay_type: "linear"
epsilon_decay_episodes: 10000
```

### P4: Increase Batch Size

GPU is only at 15-20% utilization. Increase batch size to use more GPU:

```yaml
batch_size: 128  # or even 256
```

This provides more stable gradients and better utilizes the hardware.

### P5: Address Frame Desync

Investigate the persistent FRAME_DESYNC errors. The Lua/Python frame counters need synchronization. Consider:
- Resetting frame counters at episode boundaries
- Using sequence numbers instead of absolute frame IDs
- Adding a frame ID acknowledgment protocol

---

## 8. Summary Statistics

| Metric | Value |
|---|---|
| Total runtime | ~36.7 hours |
| Total episodes | 8,594 |
| Best distance ever | x=3267 (100% - episode 196) |
| Best distance (late training) | x=2473 (78% - episode 8057) |
| Median distance (late training) | ~600-700 |
| Stuck timeout episodes | ~1,289 (~15%) |
| Training time wasted on stuck | ~16 hours (44%) |
| Final epsilon | 0.01 (reached at episode ~2300) |
| Final Q-value mean | 7.0-8.1 |
| Final loss range | 0.05-0.96 |
| Replay buffer utilization | 20,000/20,000 (full since early training) |
| Debug desync errors | 8,575 total |
