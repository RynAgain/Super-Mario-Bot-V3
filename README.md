# Super Mario Bot V3

An AI training system that teaches a neural network to play Super Mario Bros on NES using the FCEUX emulator. Uses a full Rainbow DQN (Dueling + Double + C51 Distributional + N-step + NoisyNet + Prioritized Replay) with 4-frame stacking, WebSocket communication between Lua and Python, and a distance-based reward system with stuck detection.

## System Overview

This project is a 2-part AI system:

1. **Lua Script (FCEUX side)** -- Controls frame progression, reads NES memory, executes controller inputs
2. **Python Trainer (GPU side)** -- Rainbow DQN with experience replay, reward calculation, and episode management

### Architecture

```
FCEUX (NES Emulator)                    Python Trainer
+------------------+                    +------------------+
| mario_ai.lua     |  WebSocket (8765)  | trainer.py       |
|  - Memory read   | <===============> |  - DQN Agent     |
|  - Input execute |  JSON game state   |  - Reward calc   |
|  - Frame sync    |  + JSON control    |  - Frame capture |
+------------------+                    +------------------+
```

### Key Features

- **Rainbow DQN** -- all 6 components of the Rainbow architecture:
  - Dueling DQN (separate value/advantage streams)
  - Double DQN (decoupled action selection and evaluation)
  - C51 Distributional (51-atom return distribution, support [-30, 50])
  - N-step returns (3-step bootstrapping)
  - NoisyNet (state-dependent exploration with epsilon floor)
  - Prioritized Experience Replay
- **4-frame stacking** for temporal context (84x84 grayscale)
- **12-action space** matching standard NES controller combinations
- **WebSocket communication** -- JSON payloads for game state and control
- **Frame skipping** -- acts every 4 frames for 4x training speedup
- **Soft target updates** (Polyak averaging, tau=0.005)
- **Reward clipping** to [-10, +10] per step for Q-value stability
- **Per-episode epsilon decay** (0.9995) for extended exploration
- **Aggressive stuck detection** -- 5-second timeout with escalating penalty
- **CPU-based replay buffer** -- avoids GPU OOM by storing experiences in RAM
- **TensorBoard logging** for loss, Q-values, rewards, and action distributions

## Requirements

### Software
- **FCEUX 2.6.4+** -- NES emulator with Lua scripting (included in `fceux-2.6.6-win64/`)
- **Python 3.10+** with PyTorch 2.0+ and CUDA support
- **Super Mario Bros (World).nes** ROM file (user must provide legally)

### Hardware
- **GPU**: NVIDIA with 6GB+ VRAM (recommended)
- **RAM**: 16GB+ system memory (replay buffer uses ~10.5GB on CPU)
- **Storage**: 2GB+ free space (logs + checkpoints)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Or with optional dependencies:
pip install -e ".[gpu,win]"
```

### 2. Create Save State

1. Open FCEUX and load the Super Mario Bros ROM
2. Start the game and get to World 1-1 gameplay (past title screen)
3. Save state to **slot 10**: `Shift+F10`

### 3. Start Training

**Terminal 1** -- Start the Python trainer:
```bash
python python/main.py train
```

**Terminal 2 (FCEUX)** -- Load the Lua script:
1. Open FCEUX with the ROM loaded
2. `File > Lua > New Lua Script Window`
3. Browse to `lua/mario_ai.lua` and click Run
4. The script will auto-connect to the Python trainer

Or use the batch file:
```bash
run_training.bat
```

### 4. Monitor Progress

Watch the Python terminal for log output showing:
- Episode number and total reward
- Mario's X position (should increase over time)
- Epsilon value (exploration rate, decreasing over episodes)
- Loss and Q-value means (should be non-zero after warmup)

Optional: Monitor with TensorBoard:
```bash
tensorboard --logdir runs/
```

## Project Structure

```
Super-Mario-Bot-V3/
|-- config/                  # YAML configuration files
|   |-- training_config.yaml # Hyperparameters, WebSocket, training schedule
|   |-- network_config.yaml  # Neural network architecture
|   |-- game_config.yaml     # Action space, memory addresses
|   +-- logging_config.yaml  # Log levels and formats
|-- lua/
|   |-- mario_ai.lua         # FCEUX Lua script (main)
|   +-- mario_ai_fallback.lua
|-- python/
|   |-- main.py              # CLI entry point
|   |-- agents/
|   |   +-- dqn_agent.py     # Rainbow DQN agent
|   |-- capture/
|   |   +-- frame_capture.py # Screen capture and game state parsing
|   |-- communication/
|   |   |-- websocket_server.py  # WebSocket server
|   |   +-- comm_manager.py      # Message routing
|   |-- environment/
|   |   |-- reward_calculator.py # Reward system with stuck detection
|   |   +-- episode_manager.py   # Episode lifecycle
|   |-- models/
|   |   +-- dueling_dqn.py      # Dueling DQN with C51 + NoisyNet
|   |-- training/
|   |   |-- trainer.py          # Main training loop
|   |   +-- training_utils.py   # State management, health monitoring
|   +-- utils/
|       |-- preprocessing.py    # Frame stacking, state normalization
|       |-- replay_buffer.py    # Prioritized experience replay (CPU-based)
|       |-- config_loader.py    # YAML config loading
|       +-- model_utils.py      # Device management, checkpointing
|-- logs/                    # CSV training logs (per-session)
|-- checkpoints/             # Model checkpoints
|-- docs/                    # Detailed documentation
|-- plans/                   # Improvement plans and analysis
|-- FEATURE_TRACKER.md       # Prioritized roadmap and progress
|-- pyproject.toml           # Python packaging
+-- requirements.txt         # Python dependencies
```

## Neural Network

### Rainbow DQN (Dueling + C51 Distributional + NoisyNet)

```
Input: 4x84x84 grayscale frames + 12-dim state vector
  |
  v
Conv2d(4, 32, 8x8, stride=4)  -->  ReLU
Conv2d(32, 64, 4x4, stride=2) -->  ReLU
Conv2d(64, 64, 3x3, stride=1) -->  ReLU
  |
  v  (flatten + concatenate with state vector)
  |
NoisyLinear(7756, 512)  -->  ReLU  -->  Dropout(0.3)
  |                                    |
  v                                    v
Value Stream                   Advantage Stream
NoisyLinear(512, 256) -> ReLU  NoisyLinear(512, 256) -> ReLU
NoisyLinear(256, 51)           NoisyLinear(256, 12*51)
  |                                    |
  +--------> Q = V + (A - mean(A)) <---+
  |          (per-atom combination)
  v
12 Q-value distributions (51 atoms each, support [-30, 50])
```

### Action Space

| ID | Action | Buttons |
|----|--------|---------|
| 0 | No action | -- |
| 1 | Right | Right |
| 2 | Left | Left |
| 3 | Jump | A |
| 4 | Right + Jump | Right + A |
| 5 | Left + Jump | Left + A |
| 6 | Run/Fire | B |
| 7 | Right + Run | Right + B |
| 8 | Left + Run | Left + B |
| 9 | Right + Jump + Run | Right + A + B |
| 10 | Left + Jump + Run | Left + A + B |
| 11 | Crouch | Down |

## Reward System

Per-step rewards are clipped to [-10, +10]. The C51 distributional network models the full return distribution over support [-30, 50].

| Component | Value | Description |
|-----------|-------|-------------|
| New max distance | +1.0/pixel | First time reaching a new X position |
| Rightward movement | +0.1/pixel | Any rightward movement (even revisiting) |
| Backward movement | -0.1/pixel | Moving left (net zero for oscillation) |
| Death penalty | -(5 + 0.01 * max_x) | Scales with progress to always outweigh gains |
| Stuck penalty | -0.1/frame | Escalating after 1s grace period, up to -24 |
| Stuck termination | 5 seconds | Episode ends after 300 frames without progress |
| Level complete | +1000 | Reaching the flagpole |

## Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 0.00025 | Adam optimizer |
| Batch size | 128 | Transferred from CPU replay to GPU per step |
| Replay buffer | 50,000 | CPU-based (10.5GB RAM), 2.5x original |
| Gamma (discount) | 0.99 | Future reward weight |
| N-step returns | 3 | Multi-step bootstrapping |
| Epsilon start | 1.0 | Full exploration |
| Epsilon end | 0.01 | Minimal exploration |
| Epsilon decay | 0.9995/episode | Reaches floor at ~9,200 episodes |
| NoisyNet floor | 5% | Minimum random action rate |
| Frame skip | 4 | Act every 4 frames |
| Target update | Polyak tau=0.005 | Soft update every training step |
| Warmup | 50 episodes | Random actions before learning |
| Reward clip | [-10, +10] | Per-step clipping |
| C51 atoms | 51 | Return distribution resolution |
| C51 support | [-30, 50] | Min/max return range |
| Stuck timeout | 300 frames (5s) | Episode termination |
| Stuck grace | 60 frames (1s) | Before penalty starts |
| Gradient clip | 10.0 | Max gradient norm |

## Configuration

Edit [`config/training_config.yaml`](config/training_config.yaml) for:
- Learning rates, batch sizes, exploration parameters
- WebSocket host/port settings
- Frame skip and target update settings
- Stuck detection thresholds
- C51 distributional parameters (v_min, v_max, num_atoms)

### Memory Tuning

The replay buffer pre-allocates all storage at initialization. Each entry uses ~220KB (two 4x84x84 float32 frame stacks). The buffer is stored on **CPU RAM** to avoid GPU OOM.

| Buffer Size | RAM Usage | Holds ~N Episodes |
|-------------|-----------|-------------------|
| 20,000 | 4.2 GB | ~133 episodes |
| 30,000 | 6.3 GB | ~200 episodes |
| **50,000** | **10.5 GB** | **~333 episodes** |
| 100,000 | 21.0 GB | ~667 episodes |

Reduce `replay_buffer_size` in `training_config.yaml` if your system has limited RAM.

## Troubleshooting

### WebSocket Connection Failed
- Make sure Python trainer is running FIRST, then load the Lua script
- Check that port 8765 is not in use: `netstat -an | findstr 8765`
- Verify firewall allows localhost connections

### No Mario Movement
- Verify save state slot 10 contains World 1-1 gameplay (not title screen)
- Check that `mario_x` values are increasing in Python logs
- Enable debug logging: `python python/main.py train --log-level DEBUG`

### GPU Out of Memory
- The replay buffer is stored on CPU by default -- GPU OOM is unlikely unless running other programs
- If GPU OOM during training, reduce `batch_size` (try 64 or 32)
- Check GPU memory with `nvidia-smi`

### High RAM Usage
- The replay buffer uses ~10.5GB RAM at the default 50K capacity
- Reduce `replay_buffer_size` to 30000 (6.3GB) or 20000 (4.2GB)

### Training Plateaus
- Check the training logs for stuck_timeout episodes consuming too much time
- Review `docs/TRAINING_ANALYSIS_20260317_225242.md` for analysis methodology
- Ensure epsilon hasn't reached the floor too early (check epsilon in logs)

## Documentation

See [`FEATURE_TRACKER.md`](FEATURE_TRACKER.md) for the full development roadmap including:
- Phase 0-2: Bug fixes and training improvements (complete)
- Phase 3: Reliability improvements
- Phase 4: Performance optimization
- Phase 5: Advanced RL techniques
- Phase 6: Per-level models with auto-progression

Detailed docs available in [`docs/`](docs/):
- [Architecture](docs/architecture.md)
- [Communication Protocol](docs/communication-protocol.md)
- [Memory Addresses](docs/memory-addresses.md)
- [Reward System](docs/reward-system.md)
- [Neural Network](docs/neural-network-architecture.md)
- [Training Analysis](docs/TRAINING_ANALYSIS_20260317_225242.md)

Improvement plans in [`plans/`](plans/):
- [Training Improvements v4](plans/training-improvements-v4.md)

## License

MIT License. Users must provide their own legally obtained ROM files.
