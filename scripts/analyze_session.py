"""Analyze the latest training session and produce a report with visualizations."""
import csv
import sys
import os
from collections import defaultdict
from pathlib import Path

# Use Agg backend for non-interactive rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SESSION = "20260328_203158"
LOG_DIR = Path("logs")
OUTPUT_DIR = Path("docs/analysis_charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_episodes(filepath):
    """Parse alternating-line episode CSV format."""
    episodes = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 1  # skip header
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        parts = line.split(',')
        
        if parts[0].startswith('2026-'):
            try:
                ep = {
                    'timestamp': parts[0],
                    'episode': int(parts[1]),
                    'duration': float(parts[2]),
                    'steps': int(parts[3]),
                    'reward': float(parts[4]),
                    'mario_x': int(parts[5]),
                    'mario_x_max': int(parts[6]),
                    'completed': parts[7] == 'True',
                    'death_cause': parts[8],
                    'completion_pct': float(parts[15]),
                    'avg_reward_per_step': float(parts[16]),
                    'max_q': float(parts[17]),
                    'min_q': float(parts[18]),
                    'exploration': int(parts[19]),
                    'exploitation': int(parts[20]),
                }
                episodes.append(ep)
            except (ValueError, IndexError):
                pass
        i += 1
    
    return episodes

def parse_training(filepath):
    """Parse training step CSV."""
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    'episode': int(row['episode']),
                    'step': int(row['step']),
                    'reward': float(row['reward']),
                    'total_reward': float(row['total_reward']),
                    'epsilon': float(row['epsilon']),
                    'loss': float(row['loss']),
                    'q_mean': float(row['q_value_mean']),
                    'q_std': float(row['q_value_std']),
                    'mario_x': int(row['mario_x']),
                    'mario_y': int(row['mario_y']),
                    'action': int(row['action_taken']),
                    'lr': float(row['learning_rate']),
                    'buffer_size': int(row['replay_buffer_size']),
                })
            except (ValueError, KeyError):
                pass
    return rows

def moving_average(data, window=200):
    """Compute moving average."""
    if len(data) < window:
        return data
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result

def plot_rewards(episodes):
    """Plot reward over episodes with moving average."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ep_nums = [e['episode'] for e in episodes]
    rewards = [e['reward'] for e in episodes]
    ma = moving_average(rewards, 500)
    
    ax.scatter(ep_nums, rewards, alpha=0.03, s=1, color='#4488cc', label='Per-episode')
    ax.plot(ep_nums, ma, color='#cc4444', linewidth=2, label='Moving avg (500)')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Episode Reward Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'reward_over_time.png', dpi=150)
    plt.close(fig)
    print(f"  [+] Saved reward_over_time.png")

def plot_distance(episodes):
    """Plot max distance over episodes."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ep_nums = [e['episode'] for e in episodes]
    distances = [e['mario_x_max'] for e in episodes]
    ma = moving_average(distances, 500)
    
    ax.scatter(ep_nums, distances, alpha=0.03, s=1, color='#44aa44', label='Per-episode')
    ax.plot(ep_nums, ma, color='#cc4444', linewidth=2, label='Moving avg (500)')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Max X Position', fontsize=12)
    ax.set_title('Max Distance Reached Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'distance_over_time.png', dpi=150)
    plt.close(fig)
    print(f"  [+] Saved distance_over_time.png")

def plot_steps(episodes):
    """Plot episode length over training."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ep_nums = [e['episode'] for e in episodes]
    steps = [e['steps'] for e in episodes]
    ma = moving_average(steps, 500)
    
    ax.scatter(ep_nums, steps, alpha=0.03, s=1, color='#aa8844', label='Per-episode')
    ax.plot(ep_nums, ma, color='#cc4444', linewidth=2, label='Moving avg (500)')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Steps', fontsize=12)
    ax.set_title('Episode Length Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'steps_over_time.png', dpi=150)
    plt.close(fig)
    print(f"  [+] Saved steps_over_time.png")

# Readable labels and colors for death types
DEATH_LABELS = {
    'death_fall': 'Fell in Pit',
    'death_enemy': 'Enemy Contact',
    'death_timeout': 'Time Expired',
    'death_unknown': 'Unknown Death',
    'death': 'Death (unclassified)',
    'timeout': 'Timeout (no life loss)',
    'stuck_timeout': 'Stuck Too Long',
    'level_complete': 'Level Complete',
    '': 'No Cause Logged',
}

DEATH_COLORS = {
    'death_fall': '#e74c3c',       # red
    'death_enemy': '#e67e22',      # orange
    'death_timeout': '#f39c12',    # amber
    'death_unknown': '#95a5a6',    # grey
    'death': '#c0392b',            # dark red (legacy)
    'timeout': '#f1c40f',          # yellow
    'stuck_timeout': '#9b59b6',    # purple
    'level_complete': '#2ecc71',   # green
    '': '#bdc3c7',                 # light grey
}

def plot_death_causes(episodes):
    """Plot death cause distribution as pie chart with granular types."""
    dc = defaultdict(int)
    for e in episodes:
        dc[e['death_cause']] += 1
    
    # Sort by frequency
    paired = sorted(dc.items(), key=lambda x: -x[1])
    labels = [DEATH_LABELS.get(k, k) for k, _ in paired]
    sizes = [v for _, v in paired]
    colors = [DEATH_COLORS.get(k, '#bdc3c7') for k, _ in paired]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90,
                                       textprops={'fontsize': 10})
    ax.set_title('Death Cause Distribution', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'death_causes.png', dpi=150)
    plt.close(fig)
    print(f"  [+] Saved death_causes.png")

def plot_death_positions(episodes):
    """Plot death X-position histogram colored by death type.
    
    Shows WHERE on the level Mario dies most, with stacked bars
    colored by cause (pit/enemy/stuck/timeout). This reveals
    obstacle-specific bottlenecks the DQN needs to overcome.
    """
    # Collect (x_position, death_type) for all death episodes
    death_data = []
    for e in episodes:
        cause = e['death_cause']
        if cause.startswith('death') or cause in ('stuck_timeout', 'timeout'):
            death_data.append((e['mario_x_max'], cause))
    
    if not death_data:
        print("  [!] No death data for death_positions chart")
        return
    
    # Define X bins (every 100 pixels across the level)
    bin_edges = list(range(0, 3300, 100))
    
    # Group deaths by cause
    cause_order = ['death_fall', 'death_enemy', 'death_timeout', 'death_unknown',
                   'death', 'stuck_timeout', 'timeout']
    cause_bins = {}
    for cause in cause_order:
        positions = [x for x, c in death_data if c == cause]
        if positions:
            counts, _ = np.histogram(positions, bins=bin_edges)
            cause_bins[cause] = counts
    
    if not cause_bins:
        print("  [!] No binnable death data for death_positions chart")
        return
    
    fig, ax = plt.subplots(figsize=(16, 6))
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
    width = 80
    
    # Stacked bar chart
    bottom = np.zeros(len(bin_centers))
    for cause in cause_order:
        if cause in cause_bins:
            label = DEATH_LABELS.get(cause, cause)
            color = DEATH_COLORS.get(cause, '#bdc3c7')
            ax.bar(bin_centers, cause_bins[cause], width=width, bottom=bottom,
                   label=label, color=color, alpha=0.85)
            bottom += cause_bins[cause]
    
    # Mark known pit locations (only 3 real pits in 1-1; x~500 is pipes+goombas)
    PITS = [(847, 933), (1456, 1582), (2476, 2548)]
    for pit_start, pit_end in PITS:
        ax.axvspan(pit_start, pit_end, alpha=0.1, color='red', zorder=0)
        ax.text((pit_start + pit_end) / 2, ax.get_ylim()[1] * 0.95, 'PIT',
                ha='center', va='top', fontsize=7, color='red', alpha=0.6)
    
    ax.set_xlabel('Level X Position', fontsize=12)
    ax.set_ylabel('Deaths', fontsize=12)
    ax.set_title('Death Heatmap by Level Position (colored by cause)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 3200)
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'death_positions.png', dpi=150)
    plt.close(fig)
    print(f"  [+] Saved death_positions.png")

def plot_exploration(episodes):
    """Plot exploration ratio over training."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ep_nums = []
    ratios = []
    for e in episodes:
        total = e['exploration'] + e['exploitation']
        if total > 0:
            ep_nums.append(e['episode'])
            ratios.append(e['exploration'] / total)
    
    if ratios:
        ma = moving_average(ratios, 500)
        ax.plot(ep_nums, ma, color='#8844cc', linewidth=2)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Exploration Ratio', fontsize=12)
        ax.set_title('Exploration Rate Over Training (epsilon-greedy)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'exploration_rate.png', dpi=150)
    plt.close(fig)
    print(f"  [+] Saved exploration_rate.png")

def plot_distance_histogram(episodes):
    """Histogram of max distances reached."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    distances = [e['mario_x_max'] for e in episodes]
    
    # Full histogram
    axes[0].hist(distances, bins=100, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Max X Position', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distance Distribution (All Episodes)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Last 2000 episodes
    recent = episodes[-2000:]
    recent_dist = [e['mario_x_max'] for e in recent]
    axes[1].hist(recent_dist, bins=60, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Max X Position', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Distance Distribution (Last 2000 Eps)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'distance_histogram.png', dpi=150)
    plt.close(fig)
    print(f"  [+] Saved distance_histogram.png")

def plot_bucketed_summary(episodes, bucket_size=1000):
    """Bar chart of bucketed performance metrics."""
    buckets = []
    for start in range(0, len(episodes), bucket_size):
        batch = episodes[start:start+bucket_size]
        if not batch:
            break
        buckets.append({
            'label': f"{batch[0]['episode']//1000}k",
            'avg_reward': sum(e['reward'] for e in batch) / len(batch),
            'avg_dist': sum(e['mario_x_max'] for e in batch) / len(batch),
            'max_dist': max(e['mario_x_max'] for e in batch),
            'avg_steps': sum(e['steps'] for e in batch) / len(batch),
        })
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    labels = [b['label'] for b in buckets]
    x = range(len(buckets))
    
    # Avg reward
    axes[0, 0].bar(x, [b['avg_reward'] for b in buckets], color='#3498db', alpha=0.8)
    axes[0, 0].set_title('Avg Reward per Bucket', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=45, fontsize=7)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Avg distance
    axes[0, 1].bar(x, [b['avg_dist'] for b in buckets], color='#2ecc71', alpha=0.8)
    axes[0, 1].set_title('Avg Distance per Bucket', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=45, fontsize=7)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Max distance
    axes[1, 0].bar(x, [b['max_dist'] for b in buckets], color='#e74c3c', alpha=0.8)
    axes[1, 0].set_title('Max Distance per Bucket', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=45, fontsize=7)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Avg steps
    axes[1, 1].bar(x, [b['avg_steps'] for b in buckets], color='#f39c12', alpha=0.8)
    axes[1, 1].set_title('Avg Episode Length per Bucket', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=45, fontsize=7)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Training Progress by 1000-Episode Buckets (Session {SESSION})', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'bucketed_summary.png', dpi=150)
    plt.close(fig)
    print(f"  [+] Saved bucketed_summary.png")

def plot_action_distribution(training):
    """Plot action distribution."""
    actions = defaultdict(int)
    for t in training:
        actions[t['action']] += 1
    
    ACTION_NAMES = {
        0: 'NOOP', 1: 'Right', 2: 'Right+A', 3: 'Right+B',
        4: 'Right+A+B', 5: 'A (Jump)', 6: 'Left', 7: 'Left+A',
        8: 'Left+B', 9: 'Right+B (Run)', 10: 'B (Run)', 11: 'Down'
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_actions = sorted(actions.items())
    labels = [ACTION_NAMES.get(a, f'Action {a}') for a, _ in sorted_actions]
    counts = [c for _, c in sorted_actions]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    bars = ax.bar(range(len(labels)), counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Action Distribution Across Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'action_distribution.png', dpi=150)
    plt.close(fig)
    print(f"  [+] Saved action_distribution.png")

def plot_loss_and_qvalues(training):
    """Plot loss and Q-values over training."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    episodes = [t['episode'] for t in training]
    losses = [t['loss'] for t in training]
    q_means = [t['q_mean'] for t in training]
    epsilons = [t['epsilon'] for t in training]
    
    # Loss
    axes[0].plot(range(len(losses)), losses, color='#e74c3c', alpha=0.5, linewidth=0.5)
    if any(l > 0 for l in losses):
        nonzero_idx = [i for i, l in enumerate(losses) if l > 0]
        nonzero_loss = [losses[i] for i in nonzero_idx]
        ma = moving_average(nonzero_loss, 100)
        axes[0].plot(nonzero_idx, ma, color='#2c3e50', linewidth=2, label='MA(100)')
        axes[0].legend()
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Q-values and epsilon
    ax2 = axes[1]
    ax2.plot(range(len(q_means)), q_means, color='#3498db', alpha=0.5, linewidth=0.5, label='Q-mean')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(len(epsilons)), epsilons, color='#e67e22', linewidth=1, label='Epsilon', alpha=0.7)
    ax2_twin.set_ylabel('Epsilon', fontsize=11, color='#e67e22')
    ax2.set_xlabel('Training Step (logged)', fontsize=11)
    ax2.set_ylabel('Q-value Mean', fontsize=11, color='#3498db')
    ax2.set_title('Q-Values and Epsilon', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'loss_and_qvalues.png', dpi=150)
    plt.close(fig)
    print(f"  [+] Saved loss_and_qvalues.png")

def plot_combined_dashboard(episodes):
    """Create a single dashboard image with key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    ep_nums = [e['episode'] for e in episodes]
    rewards = [e['reward'] for e in episodes]
    distances = [e['mario_x_max'] for e in episodes]
    steps_list = [e['steps'] for e in episodes]
    
    # 1. Reward MA
    ma_r = moving_average(rewards, 500)
    axes[0, 0].plot(ep_nums, ma_r, color='#3498db', linewidth=2)
    axes[0, 0].set_title('Avg Reward (MA 500)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distance MA
    ma_d = moving_average(distances, 500)
    axes[0, 1].plot(ep_nums, ma_d, color='#2ecc71', linewidth=2)
    axes[0, 1].set_title('Avg Distance (MA 500)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Steps MA
    ma_s = moving_average(steps_list, 500)
    axes[0, 2].plot(ep_nums, ma_s, color='#f39c12', linewidth=2)
    axes[0, 2].set_title('Avg Steps (MA 500)', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Distance histogram (recent)
    recent = episodes[-2000:]
    axes[1, 0].hist([e['mario_x_max'] for e in recent], bins=50, color='#e74c3c', alpha=0.7)
    axes[1, 0].set_title('Distance Dist. (Last 2000)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Death causes
    dc = defaultdict(int)
    for e in episodes:
        dc[e['death_cause']] += 1
    sorted_dc = sorted(dc.items(), key=lambda x: -x[1])[:6]
    axes[1, 1].barh([d[0] for d in sorted_dc], [d[1] for d in sorted_dc], color='#9b59b6', alpha=0.8)
    axes[1, 1].set_title('Death Causes', fontweight='bold')
    axes[1, 1].invert_yaxis()
    
    # 6. Key stats text box
    axes[1, 2].axis('off')
    stats_text = (
        f"Session: {SESSION}\n"
        f"Total Episodes: {len(episodes):,}\n"
        f"Best Distance: {max(distances):,}\n"
        f"Best Reward: {max(rewards):,.1f}\n"
        f"Avg Reward (all): {sum(rewards)/len(rewards):.1f}\n"
        f"Avg Distance (all): {sum(distances)/len(distances):.1f}\n"
        f"Avg Reward (last 100): {sum(e['reward'] for e in episodes[-100:])/100:.1f}\n"
        f"Avg Distance (last 100): {sum(e['mario_x_max'] for e in episodes[-100:])/100:.1f}\n"
        f"Completions: {sum(1 for e in episodes if e['completed'])}\n"
    )
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                     fontsize=13, verticalalignment='center', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    axes[1, 2].set_title('Key Statistics', fontweight='bold')
    
    fig.suptitle(f'Mario AI Training Dashboard - Session {SESSION}', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'training_dashboard.png', dpi=150)
    plt.close(fig)
    print(f"  [+] Saved training_dashboard.png")

def plot_zone_survival(episodes, window=500):
    """
    Multi-line chart showing rolling survival rate past major death zones.
    
    Each line tracks what percentage of episodes in a rolling window made it
    past a specific X threshold. Shows how the agent improves at navigating
    each obstacle over training time.
    """
    # Define death zones with human-readable names
    # World 1-1 has 3 pits: x~847, x~1456, x~2476
    # x~450-600 is tall pipes + goombas (NOT a pit)
    DEATH_ZONES = [
        (200, 'Goombas / Early Pipes (x>200)'),
        (500, 'Tall Pipes + Goombas (x>500)'),
        (850, 'Past Pit 1 (x>850)'),
        (1100, 'Mid-level Stretch (x>1100)'),
        (1580, 'Past Pit 2 (x>1580)'),
        (2000, 'Deep Run (x>2000)'),
        (2550, 'Past Pit 3 (x>2550)'),
        (3100, 'Level Complete (x>3100)'),
    ]
    
    # Color palette for the lines
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#c0392b']
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ep_nums = [e['episode'] for e in episodes]
    distances = [e['mario_x_max'] for e in episodes]
    
    for i, (threshold, label) in enumerate(DEATH_ZONES):
        # Calculate rolling survival rate past this threshold
        survival = []
        for j in range(len(episodes)):
            start = max(0, j - window + 1)
            batch = distances[start:j + 1]
            if batch:
                survived = sum(1 for d in batch if d >= threshold)
                survival.append(survived / len(batch) * 100)
            else:
                survival.append(0)
        
        # Only plot if there's meaningful data (at least some episodes past threshold)
        max_survival = max(survival) if survival else 0
        if max_survival > 0.5:  # At least 0.5% survival rate at some point
            color = colors[i % len(colors)]
            ax.plot(ep_nums, survival, color=color, linewidth=2, label=label, alpha=0.85)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(f'Survival Rate (%, MA {window})', fontsize=12)
    ax.set_title(f'Zone Survival Rates Over Training (Session {SESSION})', fontsize=14, fontweight='bold')
    ax.set_ylim(-2, 105)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal reference lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axhline(y=25, color='gray', linestyle=':', alpha=0.2, linewidth=0.8)
    ax.axhline(y=75, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'zone_survival.png', dpi=150)
    plt.close(fig)
    print(f"  [+] Saved zone_survival.png")

def analyze():
    episodes = parse_episodes(LOG_DIR / f"episodes_{SESSION}.csv")
    training = parse_training(LOG_DIR / f"training_{SESSION}.csv")
    
    if not episodes:
        print("ERROR: No episodes parsed!")
        return
    
    print(f"=" * 80)
    print(f"TRAINING SESSION ANALYSIS: {SESSION}")
    print(f"=" * 80)
    print(f"Total episodes parsed: {len(episodes)}")
    print(f"Episode range: {episodes[0]['episode']} - {episodes[-1]['episode']}")
    print(f"Time span: {episodes[0]['timestamp']} to {episodes[-1]['timestamp']}")
    print()
    
    # Overall stats
    rewards = [e['reward'] for e in episodes]
    distances = [e['mario_x_max'] for e in episodes]
    steps_list = [e['steps'] for e in episodes]
    completed = [e for e in episodes if e['completed']]
    
    print(f"--- OVERALL STATISTICS ---")
    print(f"Level completions: {len(completed)} / {len(episodes)} ({100*len(completed)/len(episodes):.4f}%)")
    print(f"Reward  -> min: {min(rewards):.1f}, max: {max(rewards):.1f}, mean: {sum(rewards)/len(rewards):.1f}, median: {sorted(rewards)[len(rewards)//2]:.1f}")
    print(f"Distance-> min: {min(distances)}, max: {max(distances)}, mean: {sum(distances)/len(distances):.1f}")
    print(f"Steps   -> min: {min(steps_list)}, max: {max(steps_list)}, mean: {sum(steps_list)/len(steps_list):.1f}")
    print()
    
    # Death cause analysis
    death_causes = defaultdict(int)
    for e in episodes:
        death_causes[e['death_cause']] += 1
    print(f"--- DEATH CAUSE BREAKDOWN ---")
    for cause, count in sorted(death_causes.items(), key=lambda x: -x[1]):
        print(f"  {cause:20s}: {count:>6} ({100*count/len(episodes):.1f}%)")
    print()
    
    # Epsilon progression
    explorations = [e['exploration'] for e in episodes if e['exploration'] + e['exploitation'] > 0]
    exploitations = [e['exploitation'] for e in episodes if e['exploration'] + e['exploitation'] > 0]
    if explorations:
        total_explore = sum(explorations)
        total_exploit = sum(exploitations)
        print(f"--- EXPLORATION vs EXPLOITATION ---")
        print(f"  Total exploration actions: {total_explore:,}")
        print(f"  Total exploitation actions: {total_exploit:,}")
        print(f"  Exploration ratio: {total_explore / (total_explore + total_exploit):.4f}")
    print()
    
    # Bucketed analysis
    print(f"--- PERFORMANCE BY PHASE (2000-episode buckets) ---")
    print(f"{'Ep Range':>16s} | {'Avg Reward':>10s} | {'Avg Dist':>8s} | {'Max Dist':>8s} | {'Avg Steps':>9s} | {'Done':>5s} | {'Explore%':>8s} | {'Top Death':>20s}")
    print("-" * 110)
    
    bucket_size = 2000
    for start in range(0, len(episodes), bucket_size):
        batch = episodes[start:start+bucket_size]
        if not batch:
            break
        ep_start = batch[0]['episode']
        ep_end = batch[-1]['episode']
        avg_r = sum(e['reward'] for e in batch) / len(batch)
        avg_d = sum(e['mario_x_max'] for e in batch) / len(batch)
        max_d = max(e['mario_x_max'] for e in batch)
        avg_s = sum(e['steps'] for e in batch) / len(batch)
        completions = sum(1 for e in batch if e['completed'])
        
        total_ex = sum(e['exploration'] for e in batch)
        total_xt = sum(e['exploitation'] for e in batch)
        expl_ratio = total_ex / max(1, total_ex + total_xt) * 100
        
        dc = defaultdict(int)
        for e in batch:
            dc[e['death_cause']] += 1
        top_death = max(dc.items(), key=lambda x: x[1])
        
        print(f"{ep_start:>7}-{ep_end:>7} | {avg_r:>10.1f} | {avg_d:>8.1f} | {max_d:>8} | {avg_s:>9.1f} | {completions:>5} | {expl_ratio:>7.1f}% | {top_death[0]:>15s}({top_death[1]})")
    print()
    
    # Q-value and loss analysis
    if training:
        print(f"--- TRAINING METRICS (from step logs) ---")
        print(f"Total training rows: {len(training):,}")
        
        losses = [t['loss'] for t in training]
        nonzero_losses = [l for l in losses if l > 0]
        q_means = [t['q_mean'] for t in training]
        nonzero_q = [q for q in q_means if q != 0]
        
        print(f"Loss values: {len(nonzero_losses):,} non-zero out of {len(losses):,} total")
        if nonzero_losses:
            print(f"  Loss -> min: {min(nonzero_losses):.6f}, max: {max(nonzero_losses):.6f}, mean: {sum(nonzero_losses)/len(nonzero_losses):.6f}")
        else:
            print(f"  *** CRITICAL: ALL LOSS VALUES ARE ZERO - NO LEARNING IS HAPPENING ***")
        
        print(f"Q-values: {len(nonzero_q):,} non-zero out of {len(q_means):,} total")
        if nonzero_q:
            print(f"  Q-val -> min: {min(nonzero_q):.4f}, max: {max(nonzero_q):.4f}, mean: {sum(nonzero_q)/len(nonzero_q):.4f}")
        else:
            print(f"  *** CRITICAL: ALL Q-VALUES ARE ZERO - MODEL IS NOT PRODUCING MEANINGFUL OUTPUTS ***")
        
        epsilons = [t['epsilon'] for t in training]
        print(f"Epsilon: start={epsilons[0]:.4f}, end={epsilons[-1]:.4f}")
        
        buffer_sizes = [t['buffer_size'] for t in training]
        print(f"Buffer: start={buffer_sizes[0]:,}, end={buffer_sizes[-1]:,}, max={max(buffer_sizes):,}")
        
        actions = defaultdict(int)
        for t in training:
            actions[t['action']] += 1
        print(f"\n--- ACTION DISTRIBUTION ---")
        for act, count in sorted(actions.items()):
            pct = 100 * count / len(training)
            print(f"  Action {act:>2}: {count:>8,} ({pct:.1f}%)")
    print()
    
    # Best episodes
    print(f"--- TOP 10 EPISODES BY DISTANCE ---")
    sorted_by_dist = sorted(episodes, key=lambda e: e['mario_x_max'], reverse=True)[:10]
    for e in sorted_by_dist:
        print(f"  Ep {e['episode']:>6}: dist={e['mario_x_max']:>5}, reward={e['reward']:>8.1f}, steps={e['steps']:>5}, completed={e['completed']}, death={e['death_cause']}")
    print()
    
    print(f"--- TOP 10 EPISODES BY REWARD ---")
    sorted_by_reward = sorted(episodes, key=lambda e: e['reward'], reverse=True)[:10]
    for e in sorted_by_reward:
        print(f"  Ep {e['episode']:>6}: reward={e['reward']:>8.1f}, dist={e['mario_x_max']:>5}, steps={e['steps']:>5}, completed={e['completed']}")
    print()

    # Recent 100 episodes
    recent = episodes[-100:]
    print(f"--- LAST 100 EPISODES ---")
    avg_r = sum(e['reward'] for e in recent) / len(recent)
    avg_d = sum(e['mario_x_max'] for e in recent) / len(recent)
    max_d = max(e['mario_x_max'] for e in recent)
    avg_s = sum(e['steps'] for e in recent) / len(recent)
    print(f"  Avg reward:   {avg_r:.1f}")
    print(f"  Avg distance: {avg_d:.1f}")
    print(f"  Max distance: {max_d}")
    print(f"  Avg steps:    {avg_s:.1f}")
    dc = defaultdict(int)
    for e in recent:
        dc[e['death_cause']] += 1
    print(f"  Deaths: {dict(dc)}")
    
    # Plateau detection
    if len(episodes) >= 6000:
        mid = episodes[len(episodes)//2 - 500:len(episodes)//2 + 500]
        late = episodes[-1000:]
        mid_avg = sum(e['mario_x_max'] for e in mid) / len(mid)
        late_avg = sum(e['mario_x_max'] for e in late) / len(late)
        print(f"\n--- PLATEAU DETECTION ---")
        print(f"  Mid-training avg distance: {mid_avg:.1f}")
        print(f"  Late-training avg distance: {late_avg:.1f}")
        delta = late_avg - mid_avg
        print(f"  Delta: {delta:.1f} ({delta / max(1, mid_avg) * 100:.1f}%)")
        if abs(delta) < mid_avg * 0.05:
            print(f"  >>> PLATEAU DETECTED - performance has stagnated <<<")
        elif delta > 0:
            print(f"  >>> IMPROVEMENT - performance is still growing <<<")
        else:
            print(f"  >>> REGRESSION - performance has declined <<<")

    # Generate charts
    print(f"\n{'=' * 80}")
    print(f"GENERATING VISUALIZATIONS -> {OUTPUT_DIR}/")
    print(f"{'=' * 80}")
    
    plot_rewards(episodes)
    plot_distance(episodes)
    plot_steps(episodes)
    plot_death_causes(episodes)
    plot_death_positions(episodes)
    plot_exploration(episodes)
    plot_distance_histogram(episodes)
    plot_bucketed_summary(episodes)
    plot_combined_dashboard(episodes)
    plot_zone_survival(episodes)
    
    if training:
        plot_action_distribution(training)
        plot_loss_and_qvalues(training)
    
    print(f"\nAll charts saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    analyze()
