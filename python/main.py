"""
Main Entry Point for Super Mario Bros AI Training System

Provides command-line interface for starting training, managing sessions,
and monitoring training progress.
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional
import signal

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from python.training.trainer import MarioTrainer
from python.mario_logging.plotter import PerformancePlotter
from python.training.training_utils import TrainingStateManager
from python.utils.config_loader import ConfigLoader


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        # Default log file
        handlers.append(logging.FileHandler(log_dir / "mario_ai_training.log"))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Set specific logger levels
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


async def start_training(args):
    """
    Start training session.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Super Mario Bros AI Training System")
    
    try:
        # Initialize trainer
        config_path = args.config or "config/training_config.yaml"
        trainer = MarioTrainer(config_path)
        
        # Setup signal handler for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            trainer.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start training
        resume_checkpoint = args.resume if hasattr(args, 'resume') else None
        await trainer.start_training(resume_from_checkpoint=resume_checkpoint)
        
        logger.info("Training completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def resume_training(args):
    """
    Resume training from checkpoint.
    
    Args:
        args: Command line arguments
    """
    if not args.checkpoint:
        print("Error: Checkpoint path is required for resume command")
        sys.exit(1)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    # Start training with resume
    args.resume = args.checkpoint
    asyncio.run(start_training(args))


def list_sessions(args):
    """
    List available training sessions.
    
    Args:
        args: Command line arguments
    """
    # Look for checkpoint files
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("No checkpoint directory found")
        return
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
    
    if not checkpoint_files:
        print("No training sessions found")
        return
    
    print("Available training sessions:")
    print("-" * 60)
    
    for checkpoint in sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            import torch
            checkpoint_data = torch.load(checkpoint, map_location='cpu')
            
            training_state = checkpoint_data.get('training_state', {})
            timestamp = checkpoint_data.get('timestamp', 'Unknown')
            
            session_id = training_state.get('session_id', 'Unknown')
            episode = training_state.get('episode', 0)
            total_reward = training_state.get('best_episode_reward', 0)
            completion_rate = training_state.get('completion_rate', 0) * 100
            
            print(f"Session: {session_id}")
            print(f"  File: {checkpoint.name}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Episode: {episode}")
            print(f"  Best Reward: {total_reward:.1f}")
            print(f"  Completion Rate: {completion_rate:.1f}%")
            print()
            
        except Exception as e:
            print(f"Error reading {checkpoint.name}: {e}")


def show_status(args):
    """
    Show training status and statistics.
    
    Args:
        args: Command line arguments
    """
    # Load training state
    state_manager = TrainingStateManager()
    training_state = state_manager.load_training_state()
    
    if not training_state:
        print("No active training session found")
        return
    
    # Get training summary
    summary = state_manager.get_training_summary()
    
    print("Training Status")
    print("=" * 50)
    
    # Session info
    session_info = summary.get('session_info', {})
    print(f"Session ID: {session_info.get('session_id', 'Unknown')}")
    print(f"Start Time: {session_info.get('start_time', 'Unknown')}")
    print(f"Duration: {session_info.get('duration_hours', 0):.1f} hours")
    print(f"Current Episode: {session_info.get('current_episode', 0)}")
    print(f"Total Steps: {session_info.get('total_steps', 0)}")
    print()
    
    # Progress info
    progress = summary.get('progress', {})
    print("Progress")
    print("-" * 20)
    print(f"Episodes Completed: {progress.get('episodes_completed', 0)}")
    print(f"Successful Episodes: {progress.get('successful_episodes', 0)}")
    print(f"Overall Completion Rate: {progress.get('overall_completion_rate', 0) * 100:.1f}%")
    print(f"Recent Completion Rate: {progress.get('recent_completion_rate', 0) * 100:.1f}%")
    print(f"Curriculum Phase: {progress.get('curriculum_phase', 'Unknown')}")
    print(f"Training Phase: {progress.get('training_phase', 'Unknown')}")
    print()
    
    # Performance info
    performance = summary.get('performance', {})
    print("Performance")
    print("-" * 20)
    print(f"Best Episode Reward: {performance.get('best_episode_reward', 0):.1f}")
    print(f"Best Episode Distance: {performance.get('best_episode_distance', 0)}")
    print(f"Average Episode Reward: {performance.get('avg_episode_reward', 0):.1f}")
    print(f"Average Episode Distance: {performance.get('avg_episode_distance', 0):.1f}")
    print(f"Recent Average Reward: {performance.get('recent_avg_reward', 0):.1f}")
    print(f"Recent Max Distance: {performance.get('recent_max_distance', 0)}")
    print(f"Improvement Trend: {performance.get('improvement_trend', 0):.3f}")
    print()
    
    # Current state
    current_state = summary.get('current_state', {})
    print("Current State")
    print("-" * 20)
    print(f"Epsilon: {current_state.get('epsilon', 0):.3f}")
    print(f"Learning Rate: {current_state.get('learning_rate', 0):.6f}")
    print(f"Replay Buffer Size: {current_state.get('replay_buffer_size', 0)}")
    print(f"Current Episode Reward: {current_state.get('current_episode_reward', 0):.1f}")
    print(f"Current Episode Steps: {current_state.get('current_episode_steps', 0)}")
    print(f"Current Episode Max X: {current_state.get('current_episode_max_x', 0)}")


def create_analysis(args):
    """
    Create training analysis plots.
    
    Args:
        args: Command line arguments
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize plotter
        session_id = args.session if hasattr(args, 'session') else None
        plotter = PerformancePlotter(
            log_directory="logs",
            session_id=session_id
        )
        
        # Create static analysis
        output_path = args.output if hasattr(args, 'output') else None
        analysis_path = plotter.create_static_analysis(output_path)
        
        if analysis_path:
            print(f"Training analysis saved to: {analysis_path}")
        else:
            print("Failed to create training analysis")
        
        # Export summary statistics
        stats = plotter.export_summary_stats()
        if stats:
            print("\nSummary Statistics:")
            print("-" * 30)
            
            if 'episodes' in stats:
                ep_stats = stats['episodes']
                print(f"Total Episodes: {ep_stats.get('total_episodes', 0)}")
                print(f"Average Reward: {ep_stats.get('avg_reward', 0):.1f}")
                print(f"Max Reward: {ep_stats.get('max_reward', 0):.1f}")
                print(f"Average Distance: {ep_stats.get('avg_distance', 0):.1f}")
                print(f"Max Distance: {ep_stats.get('max_distance', 0)}")
                print(f"Completion Rate: {ep_stats.get('completion_rate', 0):.1f}%")
                print(f"Recent Completion Rate: {ep_stats.get('recent_completion_rate', 0):.1f}%")
            
            if 'training' in stats:
                train_stats = stats['training']
                print(f"Total Steps: {train_stats.get('total_steps', 0)}")
                print(f"Average Loss: {train_stats.get('avg_loss', 0):.4f}")
                print(f"Final Loss: {train_stats.get('final_loss', 0):.4f}")
                print(f"Average Q-Value: {train_stats.get('avg_q_value', 0):.3f}")
                print(f"Final Epsilon: {train_stats.get('final_epsilon', 0):.3f}")
            
            if 'performance' in stats:
                perf_stats = stats['performance']
                print(f"Average CPU Usage: {perf_stats.get('avg_cpu_usage', 0):.1f}%")
                print(f"Average GPU Usage: {perf_stats.get('avg_gpu_usage', 0):.1f}%")
                print(f"Average Memory Usage: {perf_stats.get('avg_memory_usage_gb', 0):.1f} GB")
                print(f"Max Memory Usage: {perf_stats.get('max_memory_usage_gb', 0):.1f} GB")
        
    except Exception as e:
        logger.error(f"Failed to create analysis: {e}")
        print(f"Error creating analysis: {e}")


def start_monitor(args):
    """
    Start real-time training monitor.
    
    Args:
        args: Command line arguments
    """
    try:
        session_id = args.session if hasattr(args, 'session') else None
        plotter = PerformancePlotter(
            log_directory="logs",
            session_id=session_id
        )
        
        print("Starting real-time training monitor...")
        print("Close the plot window to exit.")
        
        plotter.start_realtime_monitoring()
        
    except Exception as e:
        print(f"Error starting monitor: {e}")


def validate_config(args):
    """
    Validate training configuration.
    
    Args:
        args: Command line arguments
    """
    try:
        config_path = args.config or "config/training_config.yaml"
        config_loader = ConfigLoader()
        # Extract just the filename if a full path is provided
        if "/" in config_path or "\\" in config_path:
            config_filename = config_path.split("/")[-1].split("\\")[-1]
        else:
            config_filename = config_path
        config = config_loader.load_config(config_filename)
        
        print(f"Configuration file: {config_path}")
        print("Configuration validation: PASSED")
        print()
        
        # Display key configuration values
        training_config = config.get('training', {})
        print("Training Configuration:")
        print(f"  Max Episodes: {training_config.get('max_episodes', 'Not set')}")
        print(f"  Learning Rate: {training_config.get('learning_rate', 'Not set')}")
        print(f"  Batch Size: {training_config.get('batch_size', 'Not set')}")
        print(f"  Epsilon Start: {training_config.get('epsilon_start', 'Not set')}")
        print(f"  Epsilon End: {training_config.get('epsilon_end', 'Not set')}")
        print(f"  Replay Buffer Size: {training_config.get('replay_buffer_size', 'Not set')}")
        
        curriculum = training_config.get('curriculum', {})
        if curriculum.get('enabled'):
            print(f"  Curriculum Learning: Enabled")
            phases = curriculum.get('phases', [])
            for i, phase in enumerate(phases):
                print(f"    Phase {i+1}: {phase.get('name', 'Unknown')} ({phase.get('episodes', 0)} episodes)")
        else:
            print(f"  Curriculum Learning: Disabled")
        
        performance_config = config.get('performance', {})
        print("\nPerformance Configuration:")
        print(f"  Device: {performance_config.get('device', 'Not set')}")
        print(f"  Mixed Precision: {performance_config.get('mixed_precision', 'Not set')}")
        print(f"  Compile Model: {performance_config.get('compile_model', 'Not set')}")
        
        # Check both network and websocket config sections
        network_config = config.get('network', {})
        websocket_config = config.get('websocket', {})
        print("\nNetwork Configuration:")
        host = websocket_config.get('host') or network_config.get('host', 'Not set')
        port = websocket_config.get('port') or network_config.get('port', 'Not set')
        print(f"  Host: {host}")
        print(f"  Port: {port}")
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Super Mario Bros AI Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                          # Start new training session
  python main.py train --config custom.yaml    # Use custom configuration
  python main.py resume --checkpoint path.pt   # Resume from checkpoint
  python main.py status                         # Show training status
  python main.py sessions                       # List training sessions
  python main.py analyze                        # Create analysis plots
  python main.py monitor                        # Start real-time monitor
  python main.py validate --config config.yaml # Validate configuration
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path (default: logs/mario_ai_training.log)"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Start training session")
    train_parser.add_argument(
        "--config",
        help="Configuration file path (default: config/training_config.yaml)"
    )
    
    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume training from checkpoint")
    resume_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint file path"
    )
    resume_parser.add_argument(
        "--config",
        help="Configuration file path (default: config/training_config.yaml)"
    )
    
    # Status command
    subparsers.add_parser("status", help="Show training status")
    
    # Sessions command
    subparsers.add_parser("sessions", help="List training sessions")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Create training analysis")
    analyze_parser.add_argument(
        "--session",
        help="Session ID to analyze (default: most recent)"
    )
    analyze_parser.add_argument(
        "--output",
        help="Output file path for analysis plots"
    )
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Start real-time monitor")
    monitor_parser.add_argument(
        "--session",
        help="Session ID to monitor (default: most recent)"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument(
        "--config",
        help="Configuration file path (default: config/training_config.yaml)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Execute command
    if args.command == "train":
        asyncio.run(start_training(args))
    elif args.command == "resume":
        resume_training(args)
    elif args.command == "status":
        show_status(args)
    elif args.command == "sessions":
        list_sessions(args)
    elif args.command == "analyze":
        create_analysis(args)
    elif args.command == "monitor":
        start_monitor(args)
    elif args.command == "validate":
        validate_config(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()