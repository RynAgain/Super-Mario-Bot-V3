"""
Training Results Analysis Script for Super Mario Bros AI Training System

This script provides comprehensive analysis of training results, including
performance metrics, learning curves, action distributions, and detailed
statistics from CSV logs.

Usage:
    python examples/analyze_results.py [OPTIONS]

Examples:
    python examples/analyze_results.py --session session_20231201_143022
    python examples/analyze_results.py --session latest --plot
    python examples/analyze_results.py --compare session1 session2 session3
    python examples/analyze_results.py --checkpoint checkpoints/mario_ai_best.pth
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import system components
from python.logging.csv_logger import CSVLogger
from python.logging.plotter import PerformancePlotter


class TrainingAnalyzer:
    """Comprehensive training results analyzer."""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = Path(log_directory)
        self.session_data = {}
        self.analysis_results = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def find_sessions(self) -> List[str]:
        """Find all available training sessions."""
        if not self.log_directory.exists():
            return []
        
        sessions = []
        for session_dir in self.log_directory.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('session_'):
                sessions.append(session_dir.name)
        
        return sorted(sessions)
    
    def get_latest_session(self) -> Optional[str]:
        """Get the most recent training session."""
        sessions = self.find_sessions()
        return sessions[-1] if sessions else None
    
    def load_session_data(self, session_id: str) -> Dict[str, pd.DataFrame]:
        """Load all CSV data for a training session."""
        session_path = self.log_directory / session_id
        
        if not session_path.exists():
            raise FileNotFoundError(f"Session directory not found: {session_path}")
        
        data = {}
        csv_files = {
            'training_steps': 'training_steps.csv',
            'episode_summaries': 'episode_summaries.csv',
            'performance_metrics': 'performance_metrics.csv',
            'sync_quality': 'sync_quality.csv',
            'debug_events': 'debug_events.csv'
        }
        
        for data_type, filename in csv_files.items():
            file_path = session_path / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    data[data_type] = df
                    print(f"✓ Loaded {data_type}: {len(df)} records")
                except Exception as e:
                    print(f"⚠️  Failed to load {data_type}: {e}")
            else:
                print(f"⚠️  File not found: {filename}")
        
        self.session_data[session_id] = data
        return data
    
    def analyze_training_progress(self, session_id: str) -> Dict[str, Any]:
        """Analyze training progress and learning curves."""
        print(f"\n📈 Analyzing training progress for {session_id}...")
        
        data = self.session_data.get(session_id)
        if not data:
            data = self.load_session_data(session_id)
        
        analysis = {}
        
        # Episode summaries analysis
        if 'episode_summaries' in data:
            episodes_df = data['episode_summaries']
            
            # Basic statistics
            analysis['total_episodes'] = len(episodes_df)
            analysis['total_training_time'] = episodes_df['duration_seconds'].sum() / 3600  # hours
            analysis['avg_episode_duration'] = episodes_df['duration_seconds'].mean()
            
            # Reward analysis
            analysis['avg_reward'] = episodes_df['total_reward'].mean()
            analysis['max_reward'] = episodes_df['total_reward'].max()
            analysis['reward_std'] = episodes_df['total_reward'].std()
            
            # Progress analysis
            analysis['completion_rate'] = episodes_df['level_completed'].mean() * 100
            analysis['avg_distance'] = episodes_df['mario_final_x'].mean() if 'mario_final_x' in episodes_df.columns else 0
            analysis['max_distance'] = episodes_df['mario_final_x_max'].mean() if 'mario_final_x_max' in episodes_df.columns else 0
            
            # Learning curve (moving averages)
            window_size = min(100, len(episodes_df) // 10)
            if window_size > 0:
                analysis['reward_trend'] = episodes_df['total_reward'].rolling(window=window_size).mean().iloc[-1]
                analysis['completion_trend'] = episodes_df['level_completed'].rolling(window=window_size).mean().iloc[-1] * 100
            
            # Death cause analysis
            if 'death_cause' in episodes_df.columns:
                death_causes = episodes_df['death_cause'].value_counts()
                analysis['death_causes'] = death_causes.to_dict()
        
        # Training steps analysis
        if 'training_steps' in data:
            steps_df = data['training_steps']
            
            # Learning metrics
            analysis['total_steps'] = len(steps_df)
            analysis['avg_loss'] = steps_df['loss'].mean() if 'loss' in steps_df.columns else 0
            analysis['final_epsilon'] = steps_df['epsilon'].iloc[-1] if 'epsilon' in steps_df.columns else 0
            
            # Q-value analysis
            if 'q_value_mean' in steps_df.columns:
                analysis['avg_q_value'] = steps_df['q_value_mean'].mean()
                analysis['q_value_stability'] = steps_df['q_value_mean'].std()
        
        # Performance metrics
        if 'performance_metrics' in data:
            perf_df = data['performance_metrics']
            analysis['avg_processing_time'] = perf_df['processing_time_ms'].mean() if 'processing_time_ms' in perf_df.columns else 0
        
        # Sync quality
        if 'sync_quality' in data:
            sync_df = data['sync_quality']
            analysis['avg_sync_quality'] = sync_df['sync_quality_percent'].mean() if 'sync_quality_percent' in sync_df.columns else 0
            analysis['frame_drops'] = sync_df['frame_drops'].sum() if 'frame_drops' in sync_df.columns else 0
        
        self.analysis_results[session_id] = analysis
        return analysis
    
    def analyze_action_distribution(self, session_id: str) -> Dict[str, Any]:
        """Analyze action selection patterns."""
        print(f"\n🎮 Analyzing action distribution for {session_id}...")
        
        data = self.session_data.get(session_id)
        if not data or 'training_steps' not in data:
            return {}
        
        steps_df = data['training_steps']
        
        if 'action_taken' not in steps_df.columns:
            return {}
        
        # Action distribution
        action_counts = steps_df['action_taken'].value_counts().sort_index()
        action_percentages = (action_counts / len(steps_df) * 100).round(2)
        
        # Action names mapping
        action_names = {
            0: "No Action", 1: "Right", 2: "Left", 3: "Jump",
            4: "Right+Jump", 5: "Left+Jump", 6: "Run", 7: "Right+Run",
            8: "Left+Run", 9: "Right+Jump+Run", 10: "Left+Jump+Run", 11: "Crouch"
        }
        
        action_analysis = {}
        for action_id, count in action_counts.items():
            action_name = action_names.get(action_id, f"Action_{action_id}")
            action_analysis[action_name] = {
                'count': count,
                'percentage': action_percentages[action_id]
            }
        
        # Exploration vs exploitation analysis
        if 'epsilon' in steps_df.columns:
            # Approximate exploration actions (when epsilon was high)
            high_epsilon_threshold = 0.5
            exploration_steps = steps_df[steps_df['epsilon'] > high_epsilon_threshold]
            exploitation_steps = steps_df[steps_df['epsilon'] <= high_epsilon_threshold]
            
            action_analysis['exploration_ratio'] = len(exploration_steps) / len(steps_df) * 100
            action_analysis['exploitation_ratio'] = len(exploitation_steps) / len(steps_df) * 100
        
        return action_analysis
    
    def analyze_learning_stability(self, session_id: str) -> Dict[str, Any]:
        """Analyze learning stability and convergence."""
        print(f"\n📊 Analyzing learning stability for {session_id}...")
        
        data = self.session_data.get(session_id)
        if not data:
            return {}
        
        stability_analysis = {}
        
        # Reward stability
        if 'episode_summaries' in data:
            episodes_df = data['episode_summaries']
            rewards = episodes_df['total_reward'].values
            
            # Calculate reward variance over time
            window_size = min(50, len(rewards) // 5)
            if window_size > 0:
                rolling_std = pd.Series(rewards).rolling(window=window_size).std()
                stability_analysis['reward_stability'] = {
                    'early_std': rolling_std.iloc[window_size:window_size*2].mean(),
                    'late_std': rolling_std.iloc[-window_size:].mean(),
                    'overall_std': rolling_std.mean()
                }
        
        # Loss stability
        if 'training_steps' in data:
            steps_df = data['training_steps']
            if 'loss' in steps_df.columns:
                losses = steps_df['loss'].values
                
                # Remove outliers for stability analysis
                q75, q25 = np.percentile(losses, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                filtered_losses = losses[(losses >= lower_bound) & (losses <= upper_bound)]
                
                stability_analysis['loss_stability'] = {
                    'mean_loss': np.mean(filtered_losses),
                    'loss_std': np.std(filtered_losses),
                    'loss_trend': 'decreasing' if np.corrcoef(range(len(losses)), losses)[0,1] < -0.1 else 'stable'
                }
        
        # Q-value stability
        if 'training_steps' in data and 'q_value_mean' in data['training_steps'].columns:
            q_values = data['training_steps']['q_value_mean'].values
            stability_analysis['q_value_stability'] = {
                'mean_q_value': np.mean(q_values),
                'q_value_std': np.std(q_values),
                'q_value_range': np.max(q_values) - np.min(q_values)
            }
        
        return stability_analysis
    
    def generate_plots(self, session_id: str, output_dir: str = "plots") -> List[str]:
        """Generate comprehensive analysis plots."""
        print(f"\n📊 Generating plots for {session_id}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        data = self.session_data.get(session_id)
        if not data:
            return []
        
        plot_files = []
        
        # 1. Training Progress Plot
        if 'episode_summaries' in data:
            episodes_df = data['episode_summaries']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - {session_id}', fontsize=16)
            
            # Reward over time
            axes[0,0].plot(episodes_df.index, episodes_df['total_reward'])
            axes[0,0].set_title('Episode Rewards')
            axes[0,0].set_xlabel('Episode')
            axes[0,0].set_ylabel('Total Reward')
            axes[0,0].grid(True, alpha=0.3)
            
            # Completion rate (moving average)
            window = min(50, len(episodes_df) // 10)
            if window > 0:
                completion_ma = episodes_df['level_completed'].rolling(window=window).mean() * 100
                axes[0,1].plot(episodes_df.index, completion_ma)
            axes[0,1].set_title(f'Level Completion Rate (MA-{window})')
            axes[0,1].set_xlabel('Episode')
            axes[0,1].set_ylabel('Completion Rate (%)')
            axes[0,1].grid(True, alpha=0.3)
            
            # Distance progress
            if 'mario_final_x_max' in episodes_df.columns:
                axes[1,0].plot(episodes_df.index, episodes_df['mario_final_x_max'])
                axes[1,0].set_title('Maximum Distance Reached')
                axes[1,0].set_xlabel('Episode')
                axes[1,0].set_ylabel('Distance (pixels)')
                axes[1,0].grid(True, alpha=0.3)
            
            # Episode duration
            axes[1,1].plot(episodes_df.index, episodes_df['duration_seconds'])
            axes[1,1].set_title('Episode Duration')
            axes[1,1].set_xlabel('Episode')
            axes[1,1].set_ylabel('Duration (seconds)')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_path / f'{session_id}_training_progress.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))
        
        # 2. Learning Curves Plot
        if 'training_steps' in data:
            steps_df = data['training_steps']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Learning Curves - {session_id}', fontsize=16)
            
            # Loss over time
            if 'loss' in steps_df.columns:
                # Smooth loss curve
                window = max(100, len(steps_df) // 100)
                loss_smooth = steps_df['loss'].rolling(window=window).mean()
                axes[0,0].plot(steps_df.index, loss_smooth)
                axes[0,0].set_title('Training Loss (Smoothed)')
                axes[0,0].set_xlabel('Training Step')
                axes[0,0].set_ylabel('Loss')
                axes[0,0].grid(True, alpha=0.3)
            
            # Epsilon decay
            if 'epsilon' in steps_df.columns:
                axes[0,1].plot(steps_df.index, steps_df['epsilon'])
                axes[0,1].set_title('Epsilon Decay')
                axes[0,1].set_xlabel('Training Step')
                axes[0,1].set_ylabel('Epsilon')
                axes[0,1].grid(True, alpha=0.3)
            
            # Q-values
            if 'q_value_mean' in steps_df.columns:
                window = max(100, len(steps_df) // 100)
                q_smooth = steps_df['q_value_mean'].rolling(window=window).mean()
                axes[1,0].plot(steps_df.index, q_smooth)
                axes[1,0].set_title('Q-Values (Smoothed)')
                axes[1,0].set_xlabel('Training Step')
                axes[1,0].set_ylabel('Mean Q-Value')
                axes[1,0].grid(True, alpha=0.3)
            
            # Reward per step
            if 'reward' in steps_df.columns:
                window = max(100, len(steps_df) // 100)
                reward_smooth = steps_df['reward'].rolling(window=window).mean()
                axes[1,1].plot(steps_df.index, reward_smooth)
                axes[1,1].set_title('Step Rewards (Smoothed)')
                axes[1,1].set_xlabel('Training Step')
                axes[1,1].set_ylabel('Reward')
                axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_path / f'{session_id}_learning_curves.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))
        
        # 3. Action Distribution Plot
        action_analysis = self.analyze_action_distribution(session_id)
        if action_analysis:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Action Analysis - {session_id}', fontsize=16)
            
            # Action distribution pie chart
            actions = [k for k in action_analysis.keys() if isinstance(action_analysis[k], dict)]
            percentages = [action_analysis[k]['percentage'] for k in actions]
            
            ax1.pie(percentages, labels=actions, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Action Distribution')
            
            # Exploration vs exploitation
            if 'exploration_ratio' in action_analysis:
                ratios = [action_analysis['exploration_ratio'], action_analysis['exploitation_ratio']]
                labels = ['Exploration', 'Exploitation']
                ax2.pie(ratios, labels=labels, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Exploration vs Exploitation')
            
            plt.tight_layout()
            plot_file = output_path / f'{session_id}_action_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))
        
        # 4. Performance Metrics Plot
        if 'performance_metrics' in data or 'sync_quality' in data:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(f'Performance Metrics - {session_id}', fontsize=16)
            
            # Processing time
            if 'performance_metrics' in data and 'processing_time_ms' in data['performance_metrics'].columns:
                perf_df = data['performance_metrics']
                axes[0].plot(perf_df.index, perf_df['processing_time_ms'])
                axes[0].set_title('Processing Time per Frame')
                axes[0].set_xlabel('Sample')
                axes[0].set_ylabel('Time (ms)')
                axes[0].axhline(y=16.67, color='r', linestyle='--', label='60 FPS target')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Sync quality
            if 'sync_quality' in data and 'sync_quality_percent' in data['sync_quality'].columns:
                sync_df = data['sync_quality']
                axes[1].plot(sync_df.index, sync_df['sync_quality_percent'])
                axes[1].set_title('Frame Synchronization Quality')
                axes[1].set_xlabel('Sample')
                axes[1].set_ylabel('Sync Quality (%)')
                axes[1].axhline(y=95, color='r', linestyle='--', label='Target threshold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_path / f'{session_id}_performance.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))
        
        return plot_files
    
    def compare_sessions(self, session_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple training sessions."""
        print(f"\n🔄 Comparing {len(session_ids)} training sessions...")
        
        comparison = {
            'sessions': session_ids,
            'metrics': {}
        }
        
        # Load data for all sessions
        for session_id in session_ids:
            if session_id not in self.session_data:
                self.load_session_data(session_id)
            if session_id not in self.analysis_results:
                self.analyze_training_progress(session_id)
        
        # Compare key metrics
        metrics_to_compare = [
            'total_episodes', 'avg_reward', 'max_reward', 'completion_rate',
            'avg_distance', 'max_distance', 'total_training_time'
        ]
        
        for metric in metrics_to_compare:
            comparison['metrics'][metric] = {}
            for session_id in session_ids:
                analysis = self.analysis_results.get(session_id, {})
                comparison['metrics'][metric][session_id] = analysis.get(metric, 0)
        
        return comparison
    
    def generate_comparison_plot(self, session_ids: List[str], output_dir: str = "plots") -> str:
        """Generate comparison plot for multiple sessions."""
        comparison = self.compare_sessions(session_ids)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Sessions Comparison', fontsize=16)
        
        # Metrics to plot
        plot_metrics = [
            ('avg_reward', 'Average Reward'),
            ('completion_rate', 'Completion Rate (%)'),
            ('max_distance', 'Maximum Distance'),
            ('total_training_time', 'Training Time (hours)')
        ]
        
        for i, (metric, title) in enumerate(plot_metrics):
            ax = axes[i//2, i%2]
            
            values = [comparison['metrics'][metric].get(session, 0) for session in session_ids]
            bars = ax.bar(range(len(session_ids)), values)
            
            ax.set_title(title)
            ax.set_xlabel('Session')
            ax.set_ylabel(title.split('(')[0].strip())
            ax.set_xticks(range(len(session_ids)))
            ax.set_xticklabels([s.replace('session_', '') for s in session_ids], rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = output_path / 'sessions_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def print_analysis_report(self, session_id: str):
        """Print comprehensive analysis report."""
        analysis = self.analysis_results.get(session_id)
        if not analysis:
            analysis = self.analyze_training_progress(session_id)
        
        print(f"\n" + "="*60)
        print(f"TRAINING ANALYSIS REPORT - {session_id}")
        print("="*60)
        
        # Training Overview
        print(f"\n📊 Training Overview:")
        print(f"  Total Episodes: {analysis.get('total_episodes', 0):,}")
        print(f"  Total Steps: {analysis.get('total_steps', 0):,}")
        print(f"  Training Time: {analysis.get('total_training_time', 0):.2f} hours")
        print(f"  Avg Episode Duration: {analysis.get('avg_episode_duration', 0):.1f} seconds")
        
        # Performance Metrics
        print(f"\n🎯 Performance Metrics:")
        print(f"  Average Reward: {analysis.get('avg_reward', 0):.2f}")
        print(f"  Maximum Reward: {analysis.get('max_reward', 0):.2f}")
        print(f"  Reward Std Dev: {analysis.get('reward_std', 0):.2f}")
        print(f"  Level Completion Rate: {analysis.get('completion_rate', 0):.1f}%")
        print(f"  Average Distance: {analysis.get('avg_distance', 0):.0f} pixels")
        print(f"  Maximum Distance: {analysis.get('max_distance', 0):.0f} pixels")
        
        # Learning Metrics
        print(f"\n🧠 Learning Metrics:")
        print(f"  Average Loss: {analysis.get('avg_loss', 0):.4f}")
        print(f"  Final Epsilon: {analysis.get('final_epsilon', 0):.4f}")
        print(f"  Average Q-Value: {analysis.get('avg_q_value', 0):.4f}")
        print(f"  Q-Value Stability: {analysis.get('q_value_stability', 0):.4f}")
        
        # System Performance
        print(f"\n⚡ System Performance:")
        print(f"  Avg Processing Time: {analysis.get('avg_processing_time', 0):.2f} ms/frame")
        print(f"  Avg Sync Quality: {analysis.get('avg_sync_quality', 0):.1f}%")
        print(f"  Total Frame Drops: {analysis.get('frame_drops', 0):,}")
        
        # Death Causes
        if 'death_causes' in analysis:
            print(f"\n💀 Death Causes:")
            for cause, count in analysis['death_causes'].items():
                percentage = (count / analysis.get('total_episodes', 1)) * 100
                print(f"  {cause}: {count} ({percentage:.1f}%)")
        
        # Trends
        print(f"\n📈 Recent Trends:")
        if 'reward_trend' in analysis:
            print(f"  Reward Trend: {analysis['reward_trend']:.2f}")
        if 'completion_trend' in analysis:
            print(f"  Completion Trend: {analysis['completion_trend']:.1f}%")
        
        print("="*60)


def main():
    """Main function for training analysis."""
    parser = argparse.ArgumentParser(description="Analyze Super Mario Bros AI Training Results")
    parser.add_argument("--session", type=str, help="Session ID to analyze (use 'latest' for most recent)")
    parser.add_argument("--compare", nargs="+", help="Compare multiple sessions")
    parser.add_argument("--checkpoint", type=str, help="Analyze specific checkpoint")
    parser.add_argument("--plot", action="store_true", help="Generate analysis plots")
    parser.add_argument("--output", type=str, default="plots", help="Output directory for plots")
    parser.add_argument("--log-dir", type=str, default="logs", help="Logs directory")
    parser.add_argument("--list", action="store_true", help="List available sessions")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TrainingAnalyzer(log_directory=args.log_dir)
    
    print("🎮 Super Mario Bros AI Training Results Analyzer")
    print("="*60)
    
    # List available sessions
    if args.list:
        sessions = analyzer.find_sessions()
        if sessions:
            print(f"\nAvailable training sessions ({len(sessions)}):")
            for session in sessions:
                print(f"  - {session}")
        else:
            print("\nNo training sessions found.")
        return
    
    # Compare multiple sessions
    if args.compare:
        if len(args.compare) < 2:
            print("❌ Need at least 2 sessions to compare")
            return
        
        try:
            comparison = analyzer.compare_sessions(args.compare)
            
            print(f"\n📊 Session Comparison Results:")
            for metric, values in comparison['metrics'].items():
                print(f"\n{metric.replace('_', ' ').title()}:")
                for session, value in values.items():
                    print(f"  {session}: {value:.2f}")
            
            if args.plot:
                plot_file = analyzer.generate_comparison_plot(args.compare, args.output)
                print(f"\n📊 Comparison plot saved: {plot_file}")
                
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
        return
    
    # Analyze single session
    session_id = args.session
    if session_id == "latest":
        session_id = analyzer.get_latest_session()
        if not session_id:
            print("❌ No training sessions found")
            return
        print(f"Using latest session: {session_id}")
    elif not session_id:
        # If no session specified, use latest
        session_id = analyzer.get_latest_session()
        if not session_id:
            print("❌ No training sessions found. Specify --session or run training first.")
            return
        print(f"No session specified, using latest: {session_id}")
    
    try:
        # Load and analyze session data
        analyzer.load_session_data(session_id)
        analyzer.analyze_training_progress(session_id)
        analyzer.analyze_action_distribution(session_id)
        analyzer.analyze_learning_stability(session_id)
        
        # Print analysis report
        analyzer.print_analysis_report(session_id)
        
        # Generate plots if requested
        if args.plot:
            plot_files = analyzer.generate_plots(session_id, args.output)
            if plot_files:
                print(f"\n📊 Analysis plots generated:")
                for plot_file in plot_files:
                    print(f"  - {plot_file}")
            else:
                print("\n⚠️  No plots generated (insufficient data)")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return


if __name__ == "__main__":
    main()