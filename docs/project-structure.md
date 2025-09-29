# Super Mario Bros AI Training System - Project Structure

## Directory Layout

```
Super-Mario-Bot-V3/
├── README.md                          # Main project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore patterns
├── .gitattributes                    # Git attributes
│
├── docs/                             # Documentation
│   ├── architecture.md               # System architecture overview
│   ├── project-structure.md          # This file
│   ├── setup-guide.md               # Installation and setup instructions
│   ├── memory-addresses.md          # NES memory mapping reference
│   └── training-guide.md            # Training procedures and tips
│
├── config/                           # Configuration files
│   ├── training_config.yaml         # Training hyperparameters
│   ├── network_config.yaml          # Neural network architecture
│   ├── game_config.yaml             # Game-specific settings
│   └── logging_config.yaml          # Logging configuration
│
├── lua/                              # FCEUX Lua scripts
│   ├── mario_bot.lua                 # Main Lua controller script
│   ├── memory_reader.lua             # Memory address reading utilities
│   ├── websocket_client.lua          # WebSocket communication
│   └── utils/                        # Lua utility functions
│       ├── binary_protocol.lua       # Binary message encoding/decoding
│       └── game_state.lua           # Game state extraction helpers
│
├── python/                           # Python training system
│   ├── main.py                       # Main training entry point
│   ├── requirements.txt              # Python-specific dependencies
│   │
│   ├── core/                         # Core training components
│   │   ├── __init__.py
│   │   ├── trainer.py                # Main training loop
│   │   ├── dqn_agent.py             # Dueling DQN implementation
│   │   ├── replay_buffer.py         # Experience replay buffer
│   │   └── frame_processor.py       # Frame preprocessing and stacking
│   │
│   ├── communication/                # WebSocket and protocol handling
│   │   ├── __init__.py
│   │   ├── websocket_server.py      # WebSocket server implementation
│   │   ├── message_protocol.py      # Message encoding/decoding
│   │   └── frame_synchronizer.py    # Frame sync coordination
│   │
│   ├── game/                         # Game-specific logic
│   │   ├── __init__.py
│   │   ├── mario_environment.py     # Mario game environment wrapper
│   │   ├── reward_calculator.py     # Reward function implementation
│   │   ├── action_space.py          # NES controller action definitions
│   │   └── memory_parser.py         # Parse memory data from Lua
│   │
│   ├── neural/                       # Neural network components
│   │   ├── __init__.py
│   │   ├── dueling_dqn.py           # Dueling DQN architecture
│   │   ├── network_utils.py         # Network utility functions
│   │   └── model_checkpoints.py     # Model saving/loading
│   │
│   ├── utils/                        # Utility modules
│   │   ├── __init__.py
│   │   ├── logger.py                # CSV and console logging
│   │   ├── config_loader.py         # Configuration file handling
│   │   ├── frame_capture.py         # cv2 screen capture utilities
│   │   └── performance_monitor.py   # Performance tracking
│   │
│   └── tests/                        # Unit tests
│       ├── __init__.py
│       ├── test_dqn_agent.py
│       ├── test_communication.py
│       ├── test_reward_calculator.py
│       └── test_frame_processor.py
│
├── data/                             # Data storage
│   ├── models/                       # Trained model checkpoints
│   │   ├── checkpoints/              # Training checkpoints
│   │   └── best_models/              # Best performing models
│   │
│   ├── logs/                         # Training logs
│   │   ├── training_logs/            # CSV training performance logs
│   │   ├── tensorboard/              # TensorBoard logs (optional)
│   │   └── debug_logs/               # Debug and error logs
│   │
│   ├── replays/                      # Experience replay data
│   │   └── replay_buffers/           # Serialized replay buffer saves
│   │
│   └── screenshots/                  # Debug screenshots and recordings
│       ├── training_progress/        # Progress visualization images
│       └── debug_frames/             # Debug frame captures
│
├── scripts/                          # Utility scripts
│   ├── setup_environment.py         # Environment setup automation
│   ├── test_connection.py           # Test Lua-Python communication
│   ├── visualize_training.py       # Training progress visualization
│   └── benchmark_performance.py    # Performance benchmarking
│
└── roms/                            # Game ROM files (user-provided)
    └── Super Mario Bros (World).nes  # Super Mario Bros ROM
```

## Component Organization Principles

### 1. Separation by Technology
- **`lua/`**: All FCEUX emulator-side code
- **`python/`**: All AI training and neural network code
- **`config/`**: Centralized configuration management

### 2. Modular Architecture
- Each directory contains focused, single-responsibility modules
- Clear interfaces between components
- Easy to test and maintain independently

### 3. Data Management
- **`data/`**: All persistent data (models, logs, replays)
- Organized by data type and purpose
- Easy backup and version control exclusion

### 4. Documentation First
- **`docs/`**: Comprehensive documentation
- Architecture, setup, and usage guides
- Technical specifications and references

## Key File Purposes

### Lua Components
- **`mario_bot.lua`**: Main script loaded by FCEUX, coordinates all operations
- **`memory_reader.lua`**: Extracts comprehensive game state from NES memory
- **`websocket_client.lua`**: Handles communication with Python training system

### Python Components
- **`main.py`**: Entry point, orchestrates training process
- **`dqn_agent.py`**: Implements Dueling DQN with 8-frame stacking
- **`websocket_server.py`**: Manages communication with Lua script
- **`mario_environment.py`**: Wraps game interaction in standard RL interface

### Configuration
- **`training_config.yaml`**: Learning rates, batch sizes, exploration parameters
- **`network_config.yaml`**: Network architecture, layer sizes, activation functions
- **`game_config.yaml`**: Memory addresses, action mappings, reward weights

This structure supports:
- **Scalability**: Easy to add new levels, algorithms, or features
- **Maintainability**: Clear separation of concerns
- **Testability**: Isolated components with clear interfaces
- **Collaboration**: Multiple developers can work on different components
- **Deployment**: Clear separation between development and runtime files