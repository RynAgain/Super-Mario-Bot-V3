"""
Performance Plotter for Super Mario Bros AI Training System

Provides real-time visualization of training progress, performance metrics,
and system monitoring data from CSV logs.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime, timedelta
import seaborn as sns
from collections import deque
import threading
import time


class PerformancePlotter:
    """
    Real-time performance plotter for training visualization.
    
    Creates live updating plots showing:
    - Training progress (rewards, loss, Q-values)
    - Episode statistics (completion rate, distance)
    - System performance (FPS, memory, GPU usage)
    - Synchronization quality
    """
    
    def __init__(self, 
                 log_directory: str = "logs",
                 session_id: Optional[str] = None,
                 update_interval: int = 5000,
                 max_points: int = 1000):
        """
        Initialize performance plotter.
        
        Args:
            log_directory: Directory containing CSV log files
            session_id: Session ID for log files
            update_interval: Plot update interval in milliseconds
            max_points: Maximum data points to display
        """
        self.log_directory = Path(log_directory)
        self.session_id = session_id
        self.update_interval = update_interval
        self.max_points = max_points
        
        # Data buffers for real-time plotting
        self.training_data = deque(maxlen=max_points)
        self.episode_data = deque(maxlen=max_points)
        self.performance_data = deque(maxlen=max_points)
        
        # Plot configuration
        self.setup_plot_style()
        
        # Animation objects
        self.fig = None
        self.axes = None
        self.animation = None
        
        # Data loading
        self.last_training_row = 0
        self.last_episode_row = 0
        self.last_performance_row = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Find log files
        self._find_log_files()
    
    def setup_plot_style(self):
        """Setup matplotlib and seaborn styling."""
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
        # Configure matplotlib
        plt.rcParams.update({
            'figure.facecolor': '#1e1e1e',
            'axes.facecolor': '#2d2d2d',
            'axes.edgecolor': '#ffffff',
            'axes.labelcolor': '#ffffff',
            'text.color': '#ffffff',
            'xtick.color': '#ffffff',
            'ytick.color': '#ffffff',
            'grid.color': '#404040',
            'grid.alpha': 0.3
        })
    
    def _find_log_files(self):
        """Find CSV log files in the directory."""
        if self.session_id:
            # Use specific session files
            self.training_log_path = self.log_directory / f"training_{self.session_id}.csv"
            self.episode_log_path = self.log_directory / f"episodes_{self.session_id}.csv"
            self.performance_log_path = self.log_directory / f"performance_{self.session_id}.csv"
        else:
            # Find most recent log files
            training_files = list(self.log_directory.glob("training_*.csv"))
            episode_files = list(self.log_directory.glob("episodes_*.csv"))
            performance_files = list(self.log_directory.glob("performance_*.csv"))
            
            if training_files:
                self.training_log_path = max(training_files, key=lambda x: x.stat().st_mtime)
            if episode_files:
                self.episode_log_path = max(episode_files, key=lambda x: x.stat().st_mtime)
            if performance_files:
                self.performance_log_path = max(performance_files, key=lambda x: x.stat().st_mtime)
        
        self.logger.info(f"Using log files:")
        self.logger.info(f"  Training: {getattr(self, 'training_log_path', 'Not found')}")
        self.logger.info(f"  Episodes: {getattr(self, 'episode_log_path', 'Not found')}")
        self.logger.info(f"  Performance: {getattr(self, 'performance_log_path', 'Not found')}")
    
    def load_new_data(self):
        """Load new data from CSV files."""
        try:
            # Load training data
            if hasattr(self, 'training_log_path') and self.training_log_path.exists():
                df = pd.read_csv(self.training_log_path)
                if len(df) > self.last_training_row:
                    new_data = df.iloc[self.last_training_row:].to_dict('records')
                    self.training_data.extend(new_data)
                    self.last_training_row = len(df)
            
            # Load episode data
            if hasattr(self, 'episode_log_path') and self.episode_log_path.exists():
                df = pd.read_csv(self.episode_log_path)
                if len(df) > self.last_episode_row:
                    new_data = df.iloc[self.last_episode_row:].to_dict('records')
                    self.episode_data.extend(new_data)
                    self.last_episode_row = len(df)
            
            # Load performance data
            if hasattr(self, 'performance_log_path') and self.performance_log_path.exists():
                df = pd.read_csv(self.performance_log_path)
                if len(df) > self.last_performance_row:
                    new_data = df.iloc[self.last_performance_row:].to_dict('records')
                    self.performance_data.extend(new_data)
                    self.last_performance_row = len(df)
                    
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
    
    def create_training_dashboard(self) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create comprehensive training dashboard.
        
        Returns:
            Tuple of (figure, axes_array)
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle('Super Mario Bros AI Training Dashboard', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Configure each subplot
        subplot_configs = [
            {'title': 'Episode Rewards', 'ylabel': 'Total Reward'},
            {'title': 'Training Loss', 'ylabel': 'Loss'},
            {'title': 'Q-Values', 'ylabel': 'Q-Value'},
            {'title': 'Mario Progress', 'ylabel': 'X Position'},
            {'title': 'Completion Rate', 'ylabel': 'Success Rate'},
            {'title': 'System Performance', 'ylabel': 'Usage %'},
            {'title': 'Memory Usage', 'ylabel': 'Memory (MB)'},
            {'title': 'Frame Rate', 'ylabel': 'FPS'},
            {'title': 'Exploration Rate', 'ylabel': 'Epsilon'}
        ]
        
        for i, (ax, config) in enumerate(zip(axes, subplot_configs)):
            ax.set_title(config['title'], fontweight='bold')
            ax.set_ylabel(config['ylabel'])
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#2d2d2d')
        
        plt.tight_layout()
        return fig, axes
    
    def update_plots(self, frame):
        """
        Update all plots with new data.
        
        Args:
            frame: Animation frame number
        """
        try:
            # Load new data
            self.load_new_data()
            
            if not self.training_data and not self.episode_data:
                return
            
            # Clear all axes
            for ax in self.axes:
                ax.clear()
            
            # Plot 1: Episode Rewards
            if self.episode_data:
                episodes = [d['episode'] for d in self.episode_data]
                rewards = [d['total_reward'] for d in self.episode_data]
                
                self.axes[0].plot(episodes, rewards, 'cyan', linewidth=2, alpha=0.8)
                if len(rewards) > 10:
                    # Add moving average
                    window = min(20, len(rewards) // 4)
                    moving_avg = pd.Series(rewards).rolling(window=window).mean()
                    self.axes[0].plot(episodes, moving_avg, 'yellow', linewidth=3, label=f'{window}-episode avg')
                    self.axes[0].legend()
                
                self.axes[0].set_title('Episode Rewards', fontweight='bold')
                self.axes[0].set_ylabel('Total Reward')
                self.axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Training Loss
            if self.training_data:
                loss_data = [d for d in self.training_data if d.get('loss', 0) > 0]
                if loss_data:
                    steps = [d['step'] for d in loss_data[-200:]]  # Last 200 points
                    losses = [d['loss'] for d in loss_data[-200:]]
                    
                    self.axes[1].plot(steps, losses, 'red', linewidth=1, alpha=0.7)
                    if len(losses) > 10:
                        # Smooth the loss curve
                        smooth_losses = pd.Series(losses).rolling(window=10).mean()
                        self.axes[1].plot(steps, smooth_losses, 'orange', linewidth=2)
                    
                    self.axes[1].set_title('Training Loss', fontweight='bold')
                    self.axes[1].set_ylabel('Loss')
                    self.axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Q-Values
            if self.training_data:
                recent_data = list(self.training_data)[-200:]  # Last 200 points
                steps = [d['step'] for d in recent_data]
                q_means = [d.get('q_value_mean', 0) for d in recent_data]
                q_stds = [d.get('q_value_std', 0) for d in recent_data]
                
                self.axes[2].plot(steps, q_means, 'green', linewidth=2, label='Mean Q-Value')
                self.axes[2].fill_between(steps, 
                                        [m - s for m, s in zip(q_means, q_stds)],
                                        [m + s for m, s in zip(q_means, q_stds)],
                                        alpha=0.3, color='green')
                self.axes[2].set_title('Q-Values', fontweight='bold')
                self.axes[2].set_ylabel('Q-Value')
                self.axes[2].legend()
                self.axes[2].grid(True, alpha=0.3)
            
            # Plot 4: Mario Progress
            if self.episode_data:
                episodes = [d['episode'] for d in self.episode_data]
                max_distances = [d['mario_x_max'] for d in self.episode_data]
                
                self.axes[3].plot(episodes, max_distances, 'lime', linewidth=2)
                self.axes[3].axhline(y=3168, color='red', linestyle='--', alpha=0.7, label='Level End')
                self.axes[3].set_title('Mario Progress', fontweight='bold')
                self.axes[3].set_ylabel('X Position')
                self.axes[3].legend()
                self.axes[3].grid(True, alpha=0.3)
            
            # Plot 5: Completion Rate
            if self.episode_data and len(self.episode_data) > 10:
                episodes = [d['episode'] for d in self.episode_data]
                completions = [1 if d['level_completed'] else 0 for d in self.episode_data]
                
                # Calculate rolling completion rate
                window = min(50, len(completions) // 4)
                completion_rate = pd.Series(completions).rolling(window=window).mean() * 100
                
                self.axes[4].plot(episodes, completion_rate, 'magenta', linewidth=2)
                self.axes[4].set_title('Completion Rate', fontweight='bold')
                self.axes[4].set_ylabel('Success Rate (%)')
                self.axes[4].set_ylim(0, 100)
                self.axes[4].grid(True, alpha=0.3)
            
            # Plot 6: System Performance
            if self.performance_data:
                recent_perf = list(self.performance_data)[-100:]  # Last 100 points
                steps = [d['step'] for d in recent_perf]
                cpu_usage = [d.get('cpu_percent', 0) for d in recent_perf]
                gpu_usage = [d.get('gpu_utilization_percent', 0) for d in recent_perf]
                
                self.axes[5].plot(steps, cpu_usage, 'cyan', linewidth=2, label='CPU %')
                self.axes[5].plot(steps, gpu_usage, 'yellow', linewidth=2, label='GPU %')
                self.axes[5].set_title('System Performance', fontweight='bold')
                self.axes[5].set_ylabel('Usage %')
                self.axes[5].set_ylim(0, 100)
                self.axes[5].legend()
                self.axes[5].grid(True, alpha=0.3)
            
            # Plot 7: Memory Usage
            if self.performance_data:
                recent_perf = list(self.performance_data)[-100:]
                steps = [d['step'] for d in recent_perf]
                ram_usage = [d.get('memory_usage_mb', 0) for d in recent_perf]
                gpu_memory = [d.get('gpu_memory_mb', 0) for d in recent_perf]
                
                self.axes[6].plot(steps, ram_usage, 'orange', linewidth=2, label='RAM (MB)')
                self.axes[6].plot(steps, gpu_memory, 'red', linewidth=2, label='GPU (MB)')
                self.axes[6].set_title('Memory Usage', fontweight='bold')
                self.axes[6].set_ylabel('Memory (MB)')
                self.axes[6].legend()
                self.axes[6].grid(True, alpha=0.3)
            
            # Plot 8: Frame Rate
            if self.performance_data:
                recent_perf = list(self.performance_data)[-100:]
                steps = [d['step'] for d in recent_perf]
                fps = [d.get('fps', 0) for d in recent_perf]
                
                self.axes[7].plot(steps, fps, 'white', linewidth=2)
                self.axes[7].axhline(y=60, color='green', linestyle='--', alpha=0.7, label='Target FPS')
                self.axes[7].set_title('Frame Rate', fontweight='bold')
                self.axes[7].set_ylabel('FPS')
                self.axes[7].legend()
                self.axes[7].grid(True, alpha=0.3)
            
            # Plot 9: Exploration Rate
            if self.training_data:
                recent_data = list(self.training_data)[-200:]
                steps = [d['step'] for d in recent_data]
                epsilon = [d.get('epsilon', 0) for d in recent_data]
                
                self.axes[8].plot(steps, epsilon, 'purple', linewidth=2)
                self.axes[8].set_title('Exploration Rate', fontweight='bold')
                self.axes[8].set_ylabel('Epsilon')
                self.axes[8].set_ylim(0, 1)
                self.axes[8].grid(True, alpha=0.3)
            
            # Update layout
            plt.tight_layout()
            
        except Exception as e:
            self.logger.error(f"Error updating plots: {e}")
    
    def start_realtime_monitoring(self):
        """Start real-time monitoring dashboard."""
        self.logger.info("Starting real-time training monitor...")
        
        # Create dashboard
        self.fig, self.axes = self.create_training_dashboard()
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, 
            self.update_plots,
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False
        )
        
        # Show plot
        plt.show()
    
    def create_static_analysis(self, output_path: Optional[str] = None) -> str:
        """
        Create static analysis plots and save to file.
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to saved analysis file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"training_analysis_{timestamp}.png"
        
        # Load all data
        self.load_new_data()
        
        # Create analysis figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Analysis Report', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        try:
            # Analysis 1: Reward progression
            if self.episode_data:
                episodes = [d['episode'] for d in self.episode_data]
                rewards = [d['total_reward'] for d in self.episode_data]
                
                axes[0].plot(episodes, rewards, 'cyan', alpha=0.6, linewidth=1)
                if len(rewards) > 20:
                    moving_avg = pd.Series(rewards).rolling(window=20).mean()
                    axes[0].plot(episodes, moving_avg, 'yellow', linewidth=3)
                
                axes[0].set_title('Episode Reward Progression')
                axes[0].set_xlabel('Episode')
                axes[0].set_ylabel('Total Reward')
                axes[0].grid(True, alpha=0.3)
            
            # Analysis 2: Distance progression
            if self.episode_data:
                episodes = [d['episode'] for d in self.episode_data]
                distances = [d['mario_x_max'] for d in self.episode_data]
                
                axes[1].plot(episodes, distances, 'lime', alpha=0.6, linewidth=1)
                if len(distances) > 20:
                    moving_avg = pd.Series(distances).rolling(window=20).mean()
                    axes[1].plot(episodes, moving_avg, 'green', linewidth=3)
                
                axes[1].axhline(y=3168, color='red', linestyle='--', alpha=0.7)
                axes[1].set_title('Distance Progression')
                axes[1].set_xlabel('Episode')
                axes[1].set_ylabel('Max X Position')
                axes[1].grid(True, alpha=0.3)
            
            # Analysis 3: Success rate over time
            if self.episode_data and len(self.episode_data) > 50:
                episodes = [d['episode'] for d in self.episode_data]
                completions = [1 if d['level_completed'] else 0 for d in self.episode_data]
                
                success_rate = pd.Series(completions).rolling(window=50).mean() * 100
                axes[2].plot(episodes, success_rate, 'magenta', linewidth=2)
                axes[2].set_title('Success Rate (50-episode window)')
                axes[2].set_xlabel('Episode')
                axes[2].set_ylabel('Success Rate (%)')
                axes[2].set_ylim(0, 100)
                axes[2].grid(True, alpha=0.3)
            
            # Analysis 4: Loss curve
            if self.training_data:
                loss_data = [d for d in self.training_data if d.get('loss', 0) > 0]
                if loss_data:
                    episodes = [d['episode'] for d in loss_data]
                    losses = [d['loss'] for d in loss_data]
                    
                    # Group by episode and take mean
                    loss_df = pd.DataFrame({'episode': episodes, 'loss': losses})
                    episode_loss = loss_df.groupby('episode')['loss'].mean()
                    
                    axes[3].plot(episode_loss.index, episode_loss.values, 'red', alpha=0.6)
                    if len(episode_loss) > 10:
                        smooth_loss = episode_loss.rolling(window=10).mean()
                        axes[3].plot(smooth_loss.index, smooth_loss.values, 'orange', linewidth=2)
                    
                    axes[3].set_title('Training Loss by Episode')
                    axes[3].set_xlabel('Episode')
                    axes[3].set_ylabel('Loss')
                    axes[3].grid(True, alpha=0.3)
            
            # Analysis 5: Performance distribution
            if self.episode_data:
                rewards = [d['total_reward'] for d in self.episode_data]
                axes[4].hist(rewards, bins=30, alpha=0.7, color='cyan', edgecolor='white')
                axes[4].axvline(np.mean(rewards), color='yellow', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.1f}')
                axes[4].set_title('Reward Distribution')
                axes[4].set_xlabel('Total Reward')
                axes[4].set_ylabel('Frequency')
                axes[4].legend()
                axes[4].grid(True, alpha=0.3)
            
            # Analysis 6: System performance summary
            if self.performance_data:
                cpu_usage = [d.get('cpu_percent', 0) for d in self.performance_data]
                gpu_usage = [d.get('gpu_utilization_percent', 0) for d in self.performance_data]
                memory_usage = [d.get('memory_usage_mb', 0) for d in self.performance_data]
                
                metrics = ['CPU %', 'GPU %', 'Memory (GB)']
                values = [np.mean(cpu_usage), np.mean(gpu_usage), np.mean(memory_usage) / 1024]
                
                bars = axes[5].bar(metrics, values, color=['cyan', 'yellow', 'orange'])
                axes[5].set_title('Average System Usage')
                axes[5].set_ylabel('Usage')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[5].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value:.1f}', ha='center', va='bottom')
                
                axes[5].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
            plt.close()
            
            self.logger.info(f"Static analysis saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating static analysis: {e}")
            plt.close()
            return ""
    
    def export_summary_stats(self) -> Dict[str, Any]:
        """
        Export summary statistics from training data.
        
        Returns:
            Dictionary of summary statistics
        """
        self.load_new_data()
        
        stats = {}
        
        try:
            # Episode statistics
            if self.episode_data:
                rewards = [d['total_reward'] for d in self.episode_data]
                distances = [d['mario_x_max'] for d in self.episode_data]
                completions = [1 if d['level_completed'] else 0 for d in self.episode_data]
                
                stats['episodes'] = {
                    'total_episodes': len(self.episode_data),
                    'avg_reward': np.mean(rewards),
                    'max_reward': np.max(rewards),
                    'avg_distance': np.mean(distances),
                    'max_distance': np.max(distances),
                    'completion_rate': np.mean(completions) * 100,
                    'recent_completion_rate': np.mean(completions[-50:]) * 100 if len(completions) >= 50 else 0
                }
            
            # Training statistics
            if self.training_data:
                loss_data = [d['loss'] for d in self.training_data if d.get('loss', 0) > 0]
                q_values = [d.get('q_value_mean', 0) for d in self.training_data]
                
                stats['training'] = {
                    'total_steps': len(self.training_data),
                    'avg_loss': np.mean(loss_data) if loss_data else 0,
                    'final_loss': loss_data[-1] if loss_data else 0,
                    'avg_q_value': np.mean(q_values),
                    'final_epsilon': self.training_data[-1].get('epsilon', 0) if self.training_data else 0
                }
            
            # Performance statistics
            if self.performance_data:
                cpu_usage = [d.get('cpu_percent', 0) for d in self.performance_data]
                gpu_usage = [d.get('gpu_utilization_percent', 0) for d in self.performance_data]
                memory_usage = [d.get('memory_usage_mb', 0) for d in self.performance_data]
                
                stats['performance'] = {
                    'avg_cpu_usage': np.mean(cpu_usage),
                    'avg_gpu_usage': np.mean(gpu_usage),
                    'avg_memory_usage_gb': np.mean(memory_usage) / 1024,
                    'max_memory_usage_gb': np.max(memory_usage) / 1024
                }
            
        except Exception as e:
            self.logger.error(f"Error calculating summary stats: {e}")
        
        return stats


if __name__ == "__main__":
    # Test performance plotter
    plotter = PerformancePlotter("test_logs")
    
    # Create static analysis
    analysis_path = plotter.create_static_analysis()
    print(f"Analysis saved to: {analysis_path}")
    
    # Export summary stats
    stats = plotter.export_summary_stats()
    print(f"Summary stats: {stats}")
    
    # Start real-time monitoring (commented out for testing)
    # plotter.start_realtime_monitoring()