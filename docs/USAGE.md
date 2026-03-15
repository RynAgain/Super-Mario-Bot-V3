# Usage Guide - Super Mario Bros AI Training System

This guide provides comprehensive usage instructions, examples, and best practices for the Super Mario Bros AI Training System.

## 🚀 Getting Started

### Quick Start Training

The easiest way to start training is using the Windows launcher:

```batch
# Start the training launcher
run_training.bat

# Select option 1: Quick Start Training
# Follow the prompts to begin training
```

### Command Line Training

For direct command-line usage:

```bash
# Basic training with default configuration
python python/main.py train

# Training with custom configuration
python python/main.py train --config examples/basic_training.yaml

# Resume training from checkpoint
python python/main.py train --resume checkpoints/mario_ai_episode_1000.pth

# Training with specific session ID
python python/main.py train --session my_training_session
```

## 📋 Command Reference

### Main Commands

```bash
# Training commands
python python/main.py train                    # Start training
python python/main.py train --help            # Show training options
python python/main.py evaluate               # Evaluate trained model
python python/main.py analyze                # Analyze training results

# System commands
python validate_system.py                    # Validate system setup
python test_complete_system_integration.py   # Run comprehensive tests
python python/logging/plotter.py            # Generate performance plots
```

### Training Command Options

```bash
python python/main.py train [OPTIONS]

Options:
  --config PATH          Configuration file path (default: config/training_config.yaml)
  --resume PATH          Resume from checkpoint file
  --session TEXT         Custom session ID for logging
  --episodes INTEGER     Maximum episodes to train (overrides config)
  --device TEXT          Device to use (cpu/cuda/auto)
  --log-level TEXT       Logging level (DEBUG/INFO/WARNING/ERROR)
  --no-save             Disable checkpoint saving
  --no-plots            Disable plot generation
  --help                Show help message
```

## 🎮 Training Workflow

### Step 1: Prepare FCEUX

1. **Launch FCEUX emulator**
2. **Load Super Mario Bros ROM**: `File > Open ROM`
3. **Load Lua script**: `File > Lua > New Lua Script Window`
4. **Browse to script**: Select `lua/mario_ai.lua`
5. **Run script**: Click "Run" button
6. **Verify connection**: Script should show "Waiting for connection..."

### Step 2: Start Python Training

```bash
# Option 1: Use training launcher (Windows)
run_training.bat

# Option 2: Direct command
python python/main.py train

# Option 3: Custom configuration
python python/main.py train --config examples/advanced_training.yaml
```

### Step 3: Monitor Training

Training progress can be monitored through:

- **Console output**: Real-time episode progress
- **CSV logs**: Detailed metrics in `logs/` directory
- **Checkpoints**: Model saves in `checkpoints/` directory
- **Performance plots**: Generated automatically or on-demand

### Step 4: Analyze Results

```bash
# Generate performance plots
python python/logging/plotter.py --session your_session_id

# Analyze training data
python examples/analyze_results.py --session your_session_id

# View training summary
python python/main.py analyze --session your_session_id
```

## ⚙️ Configuration Management

### Configuration Files

The system uses YAML configuration files for different aspects:

- **`config/training_config.yaml`**: Training parameters and hyperparameters
- **`config/network_config.yaml`**: Neural network architecture settings
- **`config/game_config.yaml`**: Game-specific settings and memory addresses
- **`config/logging_config.yaml`**: Logging and monitoring configuration

### Custom Configuration Example

```yaml
# examples/my_training_config.yaml
training:
  learning_rate: 0.0005
  batch_size: 64
  max_episodes: 10000
  max_steps_per_episode: 1000
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.9995
  
  curriculum:
    enabled: true
    phases:
      - name: "exploration"
        episodes: 2000
        epsilon_override: 0.8
      - name: "optimization"
        episodes: 8000
        epsilon_override: null

performance:
  device: "auto"  # auto-detect GPU/CPU
  mixed_precision: true
  compile_model: true

rewards:
  distance_reward_scale: 2.0
  completion_reward: 2000.0
  death_penalty: -200.0
```

### Using Custom Configuration

```bash
# Train with custom configuration
python python/main.py train --config examples/my_training_config.yaml

# Validate configuration before training
python python/utils/config_loader.py --validate examples/my_training_config.yaml
```

## 📊 Monitoring and Logging

### Real-time Monitoring

During training, you'll see console output like:

```
Episode 1250 | Steps: 847 | Reward: 1247.5 | Epsilon: 0.234 | Loss: 0.0823
Mario Position: x=2847 (max: 2847) | Lives: 2 | Score: 15400
Level Progress: 89.6% | Completion: False | Death: timeout
Processing: 16.2ms/frame | Sync Quality: 98.7% | Buffer: 15234/50000
```

### CSV Logging

The system generates detailed CSV logs in the `logs/` directory:

```
logs/
├── session_20231201_143022/
│   ├── training_steps.csv      # Step-by-step training data
│   ├── episode_summaries.csv   # Episode completion summaries
│   ├── performance_metrics.csv # System performance data
│   ├── sync_quality.csv       # Frame synchronization data
│   └── debug_events.csv       # Debug and error events
```

### Log Analysis

```bash
# View recent training progress
tail -f logs/latest_session/training_steps.csv

# Analyze episode summaries
python -c "
import pandas as pd
df = pd.read_csv('logs/your_session/episode_summaries.csv')
print(f'Average reward: {df.total_reward.mean():.2f}')
print(f'Completion rate: {df.level_completed.mean()*100:.1f}%')
"

# Generate performance plots
python python/logging/plotter.py --session your_session_id --output plots/
```

## 🎯 Training Strategies

### Beginner Strategy: Basic Training

```yaml
# examples/basic_training.yaml
training:
  learning_rate: 0.001
  batch_size: 32
  max_episodes: 5000
  epsilon_decay: 0.995
  curriculum:
    enabled: false
```

```bash
python python/main.py train --config examples/basic_training.yaml
```

### Intermediate Strategy: Curriculum Learning

```yaml
# examples/intermediate_training.yaml
training:
  learning_rate: 0.0005
  batch_size: 64
  max_episodes: 15000
  curriculum:
    enabled: true
    phases:
      - name: "exploration"
        episodes: 3000
        epsilon_override: 0.9
      - name: "learning"
        episodes: 7000
        epsilon_override: 0.5
      - name: "optimization"
        episodes: 5000
        epsilon_override: null
```

### Advanced Strategy: Fine-tuning

```yaml
# examples/advanced_training.yaml
training:
  learning_rate: 0.0001
  batch_size: 128
  max_episodes: 50000
  target_update_frequency: 2000
  replay_buffer_size: 200000
  
performance:
  mixed_precision: true
  compile_model: true
  
rewards:
  distance_reward_scale: 1.5
  completion_reward: 3000.0
  milestone_bonuses:
    25_percent: 200
    50_percent: 500
    75_percent: 800
    90_percent: 1200
```

## 🔄 Checkpoint Management

### Automatic Checkpoints

The system automatically saves checkpoints based on configuration:

```yaml
training:
  save_frequency: 100  # Save every 100 episodes
  keep_best_n: 5      # Keep 5 best checkpoints
  checkpoint_metrics: ["total_reward", "completion_rate"]
```

### Manual Checkpoint Operations

```bash
# List available checkpoints
ls -la checkpoints/

# Resume from specific checkpoint
python python/main.py train --resume checkpoints/mario_ai_episode_2500.pth

# Evaluate checkpoint performance
python python/main.py evaluate --checkpoint checkpoints/mario_ai_best.pth

# Convert checkpoint to deployment format
python python/utils/model_utils.py --export checkpoints/mario_ai_best.pth --format onnx
```

### Checkpoint Analysis

```bash
# Analyze checkpoint performance
python examples/analyze_results.py --checkpoint checkpoints/mario_ai_episode_1000.pth

# Compare multiple checkpoints
python examples/analyze_results.py --compare checkpoints/mario_ai_episode_*.pth
```

## 🧪 Testing and Validation

### System Validation

```bash
# Quick system check
python validate_system.py

# Comprehensive integration tests
python test_complete_system_integration.py

# Component-specific tests
python test_neural_network_components.py
python test_communication_system.py
```

### Performance Testing

```bash
# Test training performance
python examples/quick_test.py --episodes 10 --benchmark

# Test frame processing speed
python python/capture/frame_capture.py --benchmark

# Test neural network inference speed
python python/models/dueling_dqn.py --benchmark
```

## 📈 Performance Optimization

### GPU Optimization

```yaml
# config/performance_config.yaml
performance:
  device: "cuda"
  mixed_precision: true
  compile_model: true
  pin_memory: true
  num_workers: 4
  
training:
  batch_size: 128  # Larger batch for GPU
  accumulation_steps: 1
```

### CPU Optimization

```yaml
# config/cpu_config.yaml
performance:
  device: "cpu"
  mixed_precision: false
  compile_model: false
  num_workers: 2
  
training:
  batch_size: 32  # Smaller batch for CPU
  accumulation_steps: 4
```

### Memory Optimization

```yaml
training:
  replay_buffer_size: 50000  # Reduce for less memory
  batch_size: 16            # Smaller batches
  gradient_checkpointing: true
  
capture:
  frame_stack_size: 4       # Reduce frame history
```

## 🎮 Advanced Usage

### Multi-Level Training

```yaml
# Train on multiple levels
game:
  levels:
    - "1-1"
    - "1-2" 
    - "2-1"
  level_progression:
    enabled: true
    mastery_threshold: 0.8  # 80% completion rate
    episodes_per_level: 1000
```

### Custom Reward Functions

```python
# examples/custom_rewards.py
from python.environment.reward_calculator import RewardCalculator

class CustomRewardCalculator(RewardCalculator):
    def calculate_reward(self, game_state, prev_state, action):
        reward = super().calculate_reward(game_state, prev_state, action)
        
        # Add custom reward logic
        if game_state.get('powerup_collected'):
            reward += 500  # Bonus for power-ups
            
        if game_state.get('secret_area_found'):
            reward += 1000  # Bonus for secrets
            
        return reward
```

### Distributed Training

```bash
# Multi-GPU training (experimental)
python python/main.py train --config examples/distributed_config.yaml --gpus 0,1,2,3

# Multi-process training
python python/main.py train --workers 4 --config examples/multiprocess_config.yaml
```

## 🔧 Troubleshooting Usage Issues

### Common Training Issues

#### Training Not Starting
```bash
# Check system status
python validate_system.py

# Verify FCEUX connection
netstat -an | grep 8765

# Check configuration
python python/utils/config_loader.py --validate config/training_config.yaml
```

#### Poor Performance
```bash
# Check GPU utilization
nvidia-smi

# Monitor system resources
python python/training/training_utils.py --monitor

# Analyze reward distribution
python examples/analyze_results.py --session your_session --rewards
```

#### Memory Issues
```bash
# Reduce batch size
python python/main.py train --config examples/low_memory_config.yaml

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"

# Monitor memory usage
python python/training/training_utils.py --memory-profile
```

### Configuration Issues

#### Invalid Configuration
```bash
# Validate configuration file
python python/utils/config_loader.py --validate your_config.yaml

# Show configuration schema
python python/utils/config_loader.py --schema

# Generate default configuration
python python/utils/config_loader.py --generate-default > my_config.yaml
```

## 📚 Best Practices

### Training Best Practices

1. **Start Simple**: Begin with basic configuration and gradually increase complexity
2. **Monitor Progress**: Regularly check logs and plots for training progress
3. **Save Frequently**: Use appropriate checkpoint frequency for your training duration
4. **Validate System**: Run system validation before long training sessions
5. **Use Curriculum Learning**: Enable curriculum learning for better convergence

### Configuration Best Practices

1. **Version Control**: Keep configuration files in version control
2. **Document Changes**: Comment configuration changes and their purpose
3. **Test Configurations**: Validate configurations before training
4. **Backup Configs**: Keep backups of working configurations
5. **Environment-Specific**: Use different configs for different environments

### Monitoring Best Practices

1. **Regular Checkpoints**: Save checkpoints at regular intervals
2. **Performance Tracking**: Monitor system performance metrics
3. **Log Analysis**: Regularly analyze training logs for issues
4. **Plot Generation**: Generate plots to visualize training progress
5. **Resource Monitoring**: Keep track of system resource usage

## 📞 Getting Help

For usage questions and support:

1. **Check Documentation**: Review this guide and other documentation files
2. **Run Validation**: Use `python validate_system.py` to identify issues
3. **Check Logs**: Review log files for error messages and warnings
4. **Test System**: Run integration tests to verify system functionality
5. **Community Support**: Create issues on GitHub for community help

See also:
- [INSTALLATION.md](INSTALLATION.md) - Installation and setup guide
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- [README.md](README.md) - Project overview and quick start
- [docs/](docs/) - Detailed technical documentation

---

**Happy Training!** 🎮🚀