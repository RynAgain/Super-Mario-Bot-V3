"""
Integration Test for Super Mario Bros AI Training System

Tests the complete training system integration including:
- Main trainer initialization
- CSV logging system
- Performance plotting
- Training utilities
- Configuration loading
- Component orchestration
"""

import asyncio
import logging
import os
import sys
import tempfile
import time
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import all components to test
from python.training.trainer import MarioTrainer, TrainingPhase
from python.logging.csv_logger import CSVLogger
from python.logging.plotter import PerformancePlotter
from python.training.training_utils import TrainingStateManager, SystemHealthMonitor
from python.utils.config_loader import ConfigLoader


class MockGameState:
    """Mock game state for testing."""
    
    def __init__(self):
        self.mario_x = 32
        self.mario_y = 208
        self.score = 0
        self.coins = 0
        self.lives = 3
        self.time_remaining = 400
        self.level_complete = False
        self.death = False
    
    def to_dict(self):
        return {
            'mario_x': self.mario_x,
            'mario_y': self.mario_y,
            'score': self.score,
            'coins': self.coins,
            'lives': self.lives,
            'time_remaining': self.time_remaining,
            'level_complete': self.level_complete,
            'death': self.death
        }
    
    def to_binary(self):
        """Convert to mock binary data."""
        # Simple binary representation for testing
        data = np.array([
            self.mario_x, self.mario_y, self.score, self.coins,
            self.lives, self.time_remaining, 
            int(self.level_complete), int(self.death)
        ], dtype=np.int32)
        return data.tobytes()


def setup_test_environment():
    """Setup test environment with temporary directories."""
    test_dir = Path(tempfile.mkdtemp(prefix="mario_ai_test_"))
    
    # Create test directories
    (test_dir / "logs").mkdir()
    (test_dir / "checkpoints").mkdir()
    (test_dir / "config").mkdir()
    
    # Create test configuration
    test_config = {
        'training': {
            'learning_rate': 0.001,
            'batch_size': 16,
            'max_episodes': 10,
            'max_steps_per_episode': 100,
            'warmup_episodes': 2,
            'save_frequency': 5,
            'evaluation_frequency': 5,
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.99,
            'replay_buffer_size': 1000,
            'target_update_frequency': 100,
            'curriculum': {
                'enabled': True,
                'phases': [
                    {'name': 'exploration', 'episodes': 5, 'epsilon_override': 0.8},
                    {'name': 'optimization', 'episodes': 5, 'epsilon_override': None}
                ]
            }
        },
        'performance': {
            'device': 'cpu',
            'mixed_precision': False,
            'compile_model': False
        },
        'network': {
            'host': 'localhost',
            'port': 8765
        },
        'rewards': {
            'distance_reward_scale': 1.0,
            'completion_reward': 1000.0,
            'death_penalty': -100.0
        },
        'capture': {
            'frame_width': 84,
            'frame_height': 84,
            'frame_stack_size': 8
        }
    }
    
    # Save test configuration
    import yaml
    config_path = test_dir / "config" / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    return test_dir, config_path


def test_csv_logger():
    """Test CSV logger functionality."""
    print("Testing CSV Logger...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = CSVLogger(log_directory=temp_dir, session_id="test_session")
        
        # Test training step logging
        logger.log_training_step(
            episode=1, step=1, reward=1.0, total_reward=1.0, epsilon=1.0,
            loss=0.5, q_values={'mean': 0.1, 'std': 0.05},
            mario_state={'x': 32, 'y': 208, 'x_max': 32},
            action_taken=1, processing_time_ms=15.2,
            learning_rate=0.001, replay_buffer_size=100
        )
        
        # Test episode summary logging
        logger.log_episode_summary(
            episode=1, duration_seconds=30.5, total_steps=100, total_reward=150.0,
            mario_final_state={'x': 500, 'x_max': 500},
            level_completed=False, death_cause="timeout",
            game_stats={'lives': 2, 'score': 1200, 'coins': 5, 'enemies_killed': 3, 'powerups': 1, 'time_remaining': 350},
            q_value_stats={'max': 2.5, 'min': -1.2},
            action_stats={'exploration': 80, 'exploitation': 20}
        )
        
        # Test performance metrics logging
        logger.log_performance_metrics(episode=1, step=50)
        
        # Test sync quality logging
        logger.log_sync_quality(
            episode=1, step=25, frame_id=1000, sync_delay_ms=16.7,
            desync_detected=False, recovery_time_ms=0.0, frame_drops=0,
            buffer_size=1, lua_timestamp=int(time.time() * 1000),
            python_timestamp=int(time.time() * 1000)
        )
        
        # Test debug event logging
        logger.log_debug_event(
            episode=1, step=75, event_type="warning", severity="medium",
            component="test", message="Test warning message",
            mario_state={'x': 250, 'y': 208}, action_taken=2
        )
        
        # Verify log files were created
        log_files = logger.get_log_files()
        for log_type, log_path in log_files.items():
            assert log_path.exists(), f"Log file not created: {log_type}"
            assert log_path.stat().st_size > 0, f"Log file is empty: {log_type}"
        
        logger.close()
    
    print("✓ CSV Logger test passed")


def test_training_state_manager():
    """Test training state manager functionality."""
    print("Testing Training State Manager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        state_manager = TrainingStateManager(
            checkpoint_dir=temp_dir,
            auto_save_interval=10
        )
        
        # Test state initialization
        config = {'epsilon_start': 1.0, 'learning_rate': 0.001}
        state = state_manager.initialize_training_state("test_session", config)
        
        assert state.session_id == "test_session"
        assert state.epsilon == 1.0
        assert state.learning_rate == 0.001
        
        # Test state updates
        state_manager.update_episode_start(1)
        state_manager.update_step(1, 1.0, 100, 0.99, 0.001, 50)
        state_manager.update_episode_end(150.0, 500, 30.5, False)
        
        # Test state persistence
        state_manager.save_training_state()
        
        # Test state loading
        loaded_state = state_manager.load_training_state("test_session")
        assert loaded_state is not None
        assert loaded_state.session_id == "test_session"
        assert loaded_state.total_episodes_completed == 1
        
        # Test checkpoint creation (mock model and optimizer)
        mock_model_state = {'layer1.weight': torch.randn(10, 5)}
        mock_optimizer_state = {'state': {}, 'param_groups': []}
        
        checkpoint_path = state_manager.create_checkpoint(
            mock_model_state, mock_optimizer_state, {'test_data': 'test_value'}
        )
        
        assert Path(checkpoint_path).exists()
        
        # Test checkpoint loading
        model_state, optimizer_state, metadata = state_manager.load_checkpoint(checkpoint_path)
        assert 'layer1.weight' in model_state
        assert metadata['additional_data']['test_data'] == 'test_value'
        
        # Test training summary
        summary = state_manager.get_training_summary()
        assert 'session_info' in summary
        assert 'progress' in summary
        assert 'performance' in summary
    
    print("✓ Training State Manager test passed")


def test_system_health_monitor():
    """Test system health monitor functionality."""
    print("Testing System Health Monitor...")
    
    health_monitor = SystemHealthMonitor()
    
    # Test health check
    health_data = health_monitor.check_system_health()
    
    assert 'timestamp' in health_data
    assert 'cpu_percent' in health_data
    assert 'memory_usage_mb' in health_data
    assert 'status' in health_data
    assert 'warnings' in health_data
    
    # Test health summary
    # Add some mock data to history
    for _ in range(5):
        health_monitor.check_system_health()
        time.sleep(0.1)
    
    summary = health_monitor.get_health_summary()
    assert 'current_status' in summary
    assert 'avg_cpu_percent' in summary
    
    print("✓ System Health Monitor test passed")


def test_performance_plotter():
    """Test performance plotter functionality."""
    print("Testing Performance Plotter...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some mock CSV data
        csv_logger = CSVLogger(log_directory=temp_dir, session_id="test_session")
        
        # Generate mock training data
        for episode in range(1, 6):
            for step in range(1, 21):
                csv_logger.log_training_step(
                    episode=episode, step=step, 
                    reward=np.random.normal(1.0, 0.5),
                    total_reward=step * np.random.normal(1.0, 0.5),
                    epsilon=max(0.1, 1.0 - episode * 0.1),
                    loss=np.random.exponential(0.5),
                    q_values={'mean': np.random.normal(0.0, 1.0), 'std': np.random.exponential(0.5)},
                    mario_state={'x': 32 + step * 10, 'y': 208, 'x_max': 32 + step * 10},
                    action_taken=np.random.randint(0, 12),
                    processing_time_ms=np.random.normal(16.7, 2.0),
                    learning_rate=0.001, replay_buffer_size=step * 10
                )
            
            # Log episode summary
            csv_logger.log_episode_summary(
                episode=episode, duration_seconds=np.random.normal(60.0, 10.0),
                total_steps=20, total_reward=np.random.normal(100.0, 50.0),
                mario_final_state={'x': 32 + 20 * 10, 'x_max': 32 + 20 * 10},
                level_completed=np.random.random() > 0.7,
                death_cause="timeout" if np.random.random() > 0.5 else "enemy_contact",
                game_stats={'lives': 3, 'score': np.random.randint(500, 2000), 'coins': np.random.randint(0, 10), 
                           'enemies_killed': np.random.randint(0, 5), 'powerups': np.random.randint(0, 3), 'time_remaining': 300},
                q_value_stats={'max': np.random.normal(2.0, 1.0), 'min': np.random.normal(-2.0, 1.0)},
                action_stats={'exploration': np.random.randint(10, 15), 'exploitation': np.random.randint(5, 10)}
            )
        
        csv_logger.close()
        
        # Test plotter
        plotter = PerformancePlotter(
            log_directory=temp_dir,
            session_id="test_session"
        )
        
        # Test static analysis creation
        analysis_path = plotter.create_static_analysis()
        if analysis_path:  # Only check if matplotlib is available
            assert Path(analysis_path).exists()
            print(f"  Analysis plot created: {analysis_path}")
        
        # Test summary statistics export
        stats = plotter.export_summary_stats()
        assert isinstance(stats, dict)
        
        if 'episodes' in stats:
            assert 'total_episodes' in stats['episodes']
            assert stats['episodes']['total_episodes'] == 5
        
        print("  Summary stats exported successfully")
    
    print("✓ Performance Plotter test passed")


async def test_trainer_initialization():
    """Test trainer initialization without full training loop."""
    print("Testing Trainer Initialization...")
    
    test_dir, config_path = setup_test_environment()
    
    try:
        # Change to test directory
        original_cwd = os.getcwd()
        os.chdir(test_dir)
        
        # Mock external dependencies
        with patch('python.training.trainer.WebSocketServer') as mock_ws, \
             patch('python.training.trainer.FrameCapture') as mock_fc, \
             patch('python.training.trainer.DQNAgent') as mock_agent:
            
            # Setup mocks
            mock_ws_instance = Mock()
            mock_ws_instance.is_client_connected.return_value = False
            mock_ws_instance.start_server = AsyncMock()
            mock_ws_instance.stop_server = AsyncMock()
            mock_ws.return_value = mock_ws_instance
            
            mock_fc_instance = Mock()
            mock_fc.return_value = mock_fc_instance
            
            mock_agent_instance = Mock()
            mock_agent_instance.epsilon = 1.0
            mock_agent_instance.learning_rate = 0.001
            mock_agent_instance.replay_buffer = []
            mock_agent_instance.get_stats.return_value = {'episode_reward_mean': 0.0}
            mock_agent.return_value = mock_agent_instance
            
            # Test trainer initialization
            trainer = MarioTrainer(str(config_path))
            
            # Verify trainer state
            assert trainer.session_id is not None
            assert trainer.training_phase == TrainingPhase.WARMUP
            assert trainer.is_running == False
            assert trainer.should_stop == False
            assert trainer.current_episode == 0
            assert trainer.current_step == 0
            
            # Test trainer status
            status = trainer.get_training_status()
            assert 'session_id' in status
            assert 'is_running' in status
            assert 'training_phase' in status
            assert 'current_episode' in status
            assert 'current_step' in status
            
            # Test subsystem initialization
            assert trainer.csv_logger is not None
            assert trainer.state_manager is not None
            assert trainer.health_monitor is not None
            assert trainer.agent is not None
            assert trainer.reward_calculator is not None
            assert trainer.episode_manager is not None
            assert trainer.frame_capture is not None
            assert trainer.websocket_server is not None
            
            print(f"  Trainer initialized with session: {trainer.session_id}")
            print(f"  Training config: {trainer.training_config.max_episodes} episodes")
            print(f"  CSV logger session: {trainer.csv_logger.session_id}")
            print(f"  State manager initialized: {trainer.state_manager.training_state is not None}")
    
    finally:
        os.chdir(original_cwd)
        # Cleanup test directory
        import shutil
        shutil.rmtree(test_dir)
    
    print("✓ Trainer Initialization test passed")


def test_config_loader():
    """Test configuration loader functionality."""
    print("Testing Configuration Loader...")
    
    test_dir, config_path = setup_test_environment()
    
    try:
        config_loader = ConfigLoader()
        
        # Test config loading
        config = config_loader.load_config(str(config_path))
        
        assert 'training' in config
        assert 'performance' in config
        assert 'network' in config
        assert 'rewards' in config
        
        # Test specific config values
        training_config = config['training']
        assert training_config['learning_rate'] == 0.001
        assert training_config['batch_size'] == 16
        assert training_config['max_episodes'] == 10
        
        performance_config = config['performance']
        assert performance_config['device'] == 'cpu'
        assert performance_config['mixed_precision'] == False
        
        print("  Configuration loaded successfully")
        print(f"  Training episodes: {training_config['max_episodes']}")
        print(f"  Learning rate: {training_config['learning_rate']}")
        print(f"  Device: {performance_config['device']}")
    
    finally:
        # Cleanup test directory
        import shutil
        shutil.rmtree(test_dir)
    
    print("✓ Configuration Loader test passed")


async def run_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("SUPER MARIO BROS AI TRAINING SYSTEM - INTEGRATION TESTS")
    print("=" * 60)
    print()
    
    # Setup logging for tests
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    try:
        # Test individual components
        test_csv_logger()
        print()
        
        test_training_state_manager()
        print()
        
        test_system_health_monitor()
        print()
        
        test_performance_plotter()
        print()
        
        test_config_loader()
        print()
        
        # Test trainer initialization
        await test_trainer_initialization()
        print()
        
        print("=" * 60)
        print("ALL INTEGRATION TESTS PASSED! ✓")
        print("=" * 60)
        print()
        print("The Super Mario Bros AI Training System is ready for use!")
        print()
        print("To start training:")
        print("  python python/main.py train")
        print()
        print("To see all available commands:")
        print("  python python/main.py --help")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run integration tests
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)