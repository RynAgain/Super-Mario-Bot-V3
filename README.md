# Super Mario Bot V3

An AI training system that teaches a neural network to play Super Mario Bros on NES using the FCEUX emulator. Uses a Dueling DQN with 4-frame stacking, WebSocket communication between Lua and Python, and a distance-based reward system focused on level progression.

## System Overview

This project is a 2-part AI system:

1. **Lua Script (FCEUX side)** -- Controls frame progression, reads NES memory, executes controller inputs
2. **Python Trainer (GPU side)** -- Dueling DQN with experience replay, reward calculation, and episode management

### Architecture

```
FCEUX (NES Emulator)                    Python Trainer
+------------------+                    +------------------+
| mario_ai.lua     |  WebSocket (8765)  | trainer.py       |
|  - Memory read   | <===============> |  - DQN Agent     |
|  - Input execute |  128-byte binary   |  - Reward calc   |
|  - Frame sync    |  + JSON control    |  - Frame capture |
+------------------+                    +------------------+
```

### Key Features

- **Dueling DQN** with separate value and advantage streams
- **4-frame stacking** for temporal context (84x84 grayscale)
- **12-action space** matching standard NES controller combinations
- **WebSocket communication** -- binary payloads for game state, JSON for control
- **Frame skipping** -- acts every 4 frames for 4x training speedup
- **Soft target updates** (Polyak averaging, tau=0.005)
- **Reward clipping** to [-1, +1] for Q-value stability
- **Per-episode epsilon decay** (0.998) for balanced exploration

## Requirements

### Software
- **FCEUX 2.6.4+** -- NES emulator with Lua scripting ([download](http://fceux.com))
- **Python 3.10+** with PyTorch 2.0+ and CUDA support
- **Super Mario Bros (World).nes** ROM file (user must provide legally)

### Hardware
- **GPU**: NVIDIA with 4GB+ VRAM (recommended)
- **RAM**: 8GB+ system memory
- **Storage**: 1GB+ free space

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

### 4. Monitor Progress

Watch the Python terminal for log output showing:
- Episode number and total reward
- Mario's X position (should increase over time)
- Epsilon value (exploration rate, decreasing over episodes)

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
|   |   +-- dqn_agent.py     # DQN agent with replay buffer
|   |-- capture/
|   |   +-- frame_capture.py # Window capture and game state parsing
|   |-- communication/
|   |   |-- websocket_server.py  # WebSocket server
|   |   +-- comm_manager.py      # Message routing
|   |-- environment/
|   |   |-- reward_calculator.py # Reward system
|   |   +-- episode_manager.py   # Episode lifecycle
|   |-- models/
|   |   +-- dueling_dqn.py      # Dueling DQN network
|   |-- training/
|   |   |-- trainer.py          # Main training loop
|   |   +-- training_utils.py   # State management, health monitoring
|   +-- utils/
|       |-- preprocessing.py    # Frame stacking, state normalization
|       |-- replay_buffer.py    # Experience replay
|       |-- config_loader.py    # YAML config loading
|       +-- model_utils.py      # Device management, checkpointing
|-- FEATURE_TRACKER.md      # Prioritized roadmap and progress
|-- pyproject.toml           # Python packaging
+-- requirements.txt         # Python dependencies
```

## Neural Network

### Dueling DQN (4-frame stack + 12-feature state vector)

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
Linear(7756, 512)  -->  ReLU  -->  Dropout(0.3)
  |                                    |
  v                                    v
Value Stream                   Advantage Stream
Linear(512, 256) -> ReLU       Linear(512, 256) -> ReLU
Linear(256, 1)                 Linear(256, 12)
  |                                    |
  +----------> Q = V + (A - mean(A)) <-+
  |
  v
12 Q-values (one per action)
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

Rewards are clipped to [-1, +1] for stable Q-learning.

| Component | Raw Value | Description |
|-----------|-----------|-------------|
| New max distance | +1.0/pixel | First time reaching a new X position |
| Rightward movement | +0.1/pixel | Any rightward movement (even revisiting) |
| Backward movement | -0.05/pixel | Moving left |
| Death | -50.0 | Losing a life |
| Level complete | +1000.0 | Reaching the flagpole |

## Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 0.00025 | Adam optimizer |
| Batch size | 32 | From replay buffer |
| Replay buffer | 20,000 | Circular buffer |
| Gamma (discount) | 0.99 | Future reward weight |
| Epsilon start | 1.0 | Full exploration |
| Epsilon end | 0.01 | Minimal exploration |
| Epsilon decay | 0.998/episode | ~1500 episodes to reach 0.05 |
| Frame skip | 4 | Act every 4 frames |
| Target update | Polyak tau=0.005 | Soft update every training step |
| Warmup | 50 episodes | Random actions before learning |
| Reward clip | [-1, +1] | Q-value stability |

## Configuration

Edit [`config/training_config.yaml`](config/training_config.yaml) for:
- Learning rates, batch sizes, exploration parameters
- WebSocket host/port settings
- Frame skip and target update settings
- Curriculum learning phases

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
- Reduce `replay_buffer_size` in `training_config.yaml` (try 10000)
- Reduce `batch_size` (try 16)

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

## License

MIT License. Users must provide their own legally obtained ROM files.
