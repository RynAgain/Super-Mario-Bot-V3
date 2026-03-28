"""Text-only analysis of the latest training session (no chart generation)."""
import csv
import sys
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

SESSION = "20260327_002726"
LOG_DIR = Path("logs")


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
        result.append(sum(data[start:i + 1]) / (i - start + 1))
    return result


def percentile(data, pct):
    """Simple percentile calculation."""
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def analyze():
    episodes = parse_episodes(LOG_DIR / f"episodes_{SESSION}.csv")
    training = parse_training(LOG_DIR / f"training_{SESSION}.csv")

    if not episodes:
        print("ERROR: No episodes parsed!")
        return

    print(f"=" * 100)
    print(f"TRAINING SESSION ANALYSIS (TEXT-ONLY): {SESSION}")
    print(f"=" * 100)
    print(f"Total episodes parsed: {len(episodes)}")
    print(f"Episode range: {episodes[0]['episode']} - {episodes[-1]['episode']}")
    print(f"Time span: {episodes[0]['timestamp']} to {episodes[-1]['timestamp']}")
    print()

    # Overall stats
    rewards = [e['reward'] for e in episodes]
    distances = [e['mario_x_max'] for e in episodes]
    steps_list = [e['steps'] for e in episodes]
    completed = [e for e in episodes if e['completed']]
    durations = [e['duration'] for e in episodes]

    print(f"--- OVERALL STATISTICS ---")
    print(f"Level completions: {len(completed)} / {len(episodes)} ({100 * len(completed) / len(episodes):.4f}%)")
    print(f"Total training time: {sum(durations) / 3600:.2f} hours ({sum(durations):.0f} seconds)")
    print(f"Avg episode duration: {sum(durations) / len(durations):.2f}s")
    print(f"Reward  -> min: {min(rewards):.1f}, max: {max(rewards):.1f}, mean: {sum(rewards) / len(rewards):.1f}, "
          f"median: {sorted(rewards)[len(rewards) // 2]:.1f}, p25: {percentile(rewards, 25):.1f}, p75: {percentile(rewards, 75):.1f}")
    print(f"Distance-> min: {min(distances)}, max: {max(distances)}, mean: {sum(distances) / len(distances):.1f}, "
          f"median: {sorted(distances)[len(distances) // 2]}, p25: {percentile(distances, 25)}, p75: {percentile(distances, 75)}")
    print(f"Steps   -> min: {min(steps_list)}, max: {max(steps_list)}, mean: {sum(steps_list) / len(steps_list):.1f}, "
          f"median: {sorted(steps_list)[len(steps_list) // 2]}")
    print()

    # Death cause analysis
    death_causes = defaultdict(int)
    for e in episodes:
        death_causes[e['death_cause']] += 1
    print(f"--- DEATH CAUSE BREAKDOWN ---")
    for cause, count in sorted(death_causes.items(), key=lambda x: -x[1]):
        print(f"  {cause:20s}: {count:>6} ({100 * count / len(episodes):.1f}%)")
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
        # Early vs late exploration
        early_eps = [e for e in episodes[:len(episodes) // 4] if e['exploration'] + e['exploitation'] > 0]
        late_eps = [e for e in episodes[-len(episodes) // 4:] if e['exploration'] + e['exploitation'] > 0]
        if early_eps and late_eps:
            early_ratio = sum(e['exploration'] for e in early_eps) / max(1, sum(e['exploration'] + e['exploitation'] for e in early_eps))
            late_ratio = sum(e['exploration'] for e in late_eps) / max(1, sum(e['exploration'] + e['exploitation'] for e in late_eps))
            print(f"  Early exploration ratio (first 25%): {early_ratio:.4f}")
            print(f"  Late exploration ratio (last 25%):   {late_ratio:.4f}")
    print()

    # Bucketed analysis
    print(f"--- PERFORMANCE BY PHASE (2000-episode buckets) ---")
    print(f"{'Ep Range':>16s} | {'Avg Reward':>10s} | {'Avg Dist':>8s} | {'Max Dist':>8s} | {'Avg Steps':>9s} | {'Done':>5s} | {'Explore%':>8s} | {'Top Death':>20s}")
    print("-" * 110)

    bucket_size = 2000
    bucket_data = []
    for start in range(0, len(episodes), bucket_size):
        batch = episodes[start:start + bucket_size]
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

        bucket_data.append({
            'ep_start': ep_start, 'ep_end': ep_end,
            'avg_r': avg_r, 'avg_d': avg_d, 'max_d': max_d,
            'avg_s': avg_s, 'completions': completions,
            'expl_ratio': expl_ratio, 'top_death': top_death,
        })

        print(f"{ep_start:>7}-{ep_end:>7} | {avg_r:>10.1f} | {avg_d:>8.1f} | {max_d:>8} | {avg_s:>9.1f} | {completions:>5} | {expl_ratio:>7.1f}% | {top_death[0]:>15s}({top_death[1]})")
    print()

    # Trend analysis across buckets
    if len(bucket_data) >= 3:
        print(f"--- BUCKET TREND ANALYSIS ---")
        avg_rewards_per_bucket = [b['avg_r'] for b in bucket_data]
        avg_dists_per_bucket = [b['avg_d'] for b in bucket_data]
        max_dists_per_bucket = [b['max_d'] for b in bucket_data]

        first_third = bucket_data[:len(bucket_data) // 3]
        last_third = bucket_data[-len(bucket_data) // 3:]

        avg_r_early = sum(b['avg_r'] for b in first_third) / len(first_third)
        avg_r_late = sum(b['avg_r'] for b in last_third) / len(last_third)
        avg_d_early = sum(b['avg_d'] for b in first_third) / len(first_third)
        avg_d_late = sum(b['avg_d'] for b in last_third) / len(last_third)

        print(f"  Avg reward  - early third: {avg_r_early:.1f}, late third: {avg_r_late:.1f}, delta: {avg_r_late - avg_r_early:+.1f}")
        print(f"  Avg distance - early third: {avg_d_early:.1f}, late third: {avg_d_late:.1f}, delta: {avg_d_late - avg_d_early:+.1f}")
        print(f"  Best bucket avg reward:   {max(avg_rewards_per_bucket):.1f} (bucket {avg_rewards_per_bucket.index(max(avg_rewards_per_bucket))})")
        print(f"  Best bucket avg distance: {max(avg_dists_per_bucket):.1f} (bucket {avg_dists_per_bucket.index(max(avg_dists_per_bucket))})")
        print(f"  Best bucket max distance: {max(max_dists_per_bucket)} (bucket {max_dists_per_bucket.index(max(max_dists_per_bucket))})")
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
            print(f"  Loss -> min: {min(nonzero_losses):.6f}, max: {max(nonzero_losses):.6f}, mean: {sum(nonzero_losses) / len(nonzero_losses):.6f}")
            print(f"          median: {sorted(nonzero_losses)[len(nonzero_losses) // 2]:.6f}")
            # Early vs late loss
            third = len(nonzero_losses) // 3
            if third > 0:
                early_loss = sum(nonzero_losses[:third]) / third
                late_loss = sum(nonzero_losses[-third:]) / third
                print(f"  Loss trend: early avg={early_loss:.6f}, late avg={late_loss:.6f}, delta={late_loss - early_loss:+.6f}")
        else:
            print(f"  *** CRITICAL: ALL LOSS VALUES ARE ZERO - NO LEARNING IS HAPPENING ***")

        print(f"Q-values: {len(nonzero_q):,} non-zero out of {len(q_means):,} total")
        if nonzero_q:
            print(f"  Q-val -> min: {min(nonzero_q):.4f}, max: {max(nonzero_q):.4f}, mean: {sum(nonzero_q) / len(nonzero_q):.4f}")
            print(f"          median: {sorted(nonzero_q)[len(nonzero_q) // 2]:.4f}")
            # Early vs late Q
            third = len(nonzero_q) // 3
            if third > 0:
                early_q = sum(nonzero_q[:third]) / third
                late_q = sum(nonzero_q[-third:]) / third
                print(f"  Q-val trend: early avg={early_q:.4f}, late avg={late_q:.4f}, delta={late_q - early_q:+.4f}")
        else:
            print(f"  *** CRITICAL: ALL Q-VALUES ARE ZERO - MODEL IS NOT PRODUCING MEANINGFUL OUTPUTS ***")

        epsilons = [t['epsilon'] for t in training]
        print(f"Epsilon: start={epsilons[0]:.4f}, end={epsilons[-1]:.4f}")

        buffer_sizes = [t['buffer_size'] for t in training]
        print(f"Buffer: start={buffer_sizes[0]:,}, end={buffer_sizes[-1]:,}, max={max(buffer_sizes):,}")

        learning_rates = [t['lr'] for t in training]
        print(f"Learning rate: start={learning_rates[0]:.6f}, end={learning_rates[-1]:.6f}")

        actions = defaultdict(int)
        for t in training:
            actions[t['action']] += 1

        ACTION_NAMES = {
            0: 'NOOP', 1: 'Right', 2: 'Right+A', 3: 'Right+B',
            4: 'Right+A+B', 5: 'A (Jump)', 6: 'Left', 7: 'Left+A',
            8: 'Left+B', 9: 'Right+B (Run)', 10: 'B (Run)', 11: 'Down'
        }

        print(f"\n--- ACTION DISTRIBUTION ---")
        for act, count in sorted(actions.items()):
            pct = 100 * count / len(training)
            name = ACTION_NAMES.get(act, f'Action {act}')
            bar = '#' * int(pct * 2)
            print(f"  {act:>2} {name:>14s}: {count:>8,} ({pct:>5.1f}%) {bar}")

        # Action diversity analysis
        total_actions = sum(actions.values())
        action_probs = [c / total_actions for c in actions.values()]
        entropy = -sum(p * np.log2(p) for p in action_probs if p > 0)
        max_entropy = np.log2(len(actions))
        print(f"\n  Action entropy: {entropy:.3f} / {max_entropy:.3f} (normalized: {entropy / max_entropy:.3f})")
        print(f"  Number of unique actions used: {len(actions)}")
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

    # Worst episodes (lowest reward)
    print(f"--- BOTTOM 10 EPISODES BY REWARD ---")
    sorted_by_reward_asc = sorted(episodes, key=lambda e: e['reward'])[:10]
    for e in sorted_by_reward_asc:
        print(f"  Ep {e['episode']:>6}: reward={e['reward']:>8.1f}, dist={e['mario_x_max']:>5}, steps={e['steps']:>5}, death={e['death_cause']}")
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
    print(f"  Deaths: {dict(sorted(dc.items(), key=lambda x: -x[1]))}")
    print()

    # Last 500 episodes
    recent500 = episodes[-500:]
    print(f"--- LAST 500 EPISODES ---")
    avg_r500 = sum(e['reward'] for e in recent500) / len(recent500)
    avg_d500 = sum(e['mario_x_max'] for e in recent500) / len(recent500)
    max_d500 = max(e['mario_x_max'] for e in recent500)
    avg_s500 = sum(e['steps'] for e in recent500) / len(recent500)
    print(f"  Avg reward:   {avg_r500:.1f}")
    print(f"  Avg distance: {avg_d500:.1f}")
    print(f"  Max distance: {max_d500}")
    print(f"  Avg steps:    {avg_s500:.1f}")
    print()

    # Distance distribution buckets
    print(f"--- DISTANCE DISTRIBUTION ---")
    dist_buckets = defaultdict(int)
    for d in distances:
        bucket = (d // 100) * 100
        dist_buckets[bucket] += 1
    print(f"  {'Range':>12s} | {'Count':>6s} | {'Pct':>6s} | Bar")
    for bucket in sorted(dist_buckets.keys()):
        count = dist_buckets[bucket]
        pct = 100 * count / len(distances)
        bar = '#' * int(pct)
        print(f"  {bucket:>5}-{bucket + 99:>5} | {count:>6} | {pct:>5.1f}% | {bar}")
    print()

    # Completion % distribution
    completion_pcts = [e['completion_pct'] for e in episodes]
    print(f"--- COMPLETION PERCENTAGE DISTRIBUTION ---")
    print(f"  Mean completion %: {sum(completion_pcts) / len(completion_pcts):.2f}%")
    print(f"  Max completion %:  {max(completion_pcts):.2f}%")
    pct_buckets = defaultdict(int)
    for p in completion_pcts:
        bucket = int(p // 5) * 5
        pct_buckets[bucket] += 1
    for bucket in sorted(pct_buckets.keys()):
        count = pct_buckets[bucket]
        pct = 100 * count / len(completion_pcts)
        bar = '#' * int(pct)
        print(f"  {bucket:>3}-{bucket + 4:>3}% | {count:>6} | {pct:>5.1f}% | {bar}")
    print()

    # Reward per step analysis
    rps = [e['avg_reward_per_step'] for e in episodes]
    print(f"--- REWARD PER STEP ---")
    print(f"  Mean:   {sum(rps) / len(rps):.4f}")
    print(f"  Median: {sorted(rps)[len(rps) // 2]:.4f}")
    print(f"  Min:    {min(rps):.4f}")
    print(f"  Max:    {max(rps):.4f}")
    print()

    # Q-value ranges from episodes
    max_qs = [e['max_q'] for e in episodes]
    min_qs = [e['min_q'] for e in episodes]
    print(f"--- Q-VALUE RANGES (from episode summaries) ---")
    print(f"  Max Q -> min: {min(max_qs):.4f}, max: {max(max_qs):.4f}, mean: {sum(max_qs) / len(max_qs):.4f}")
    print(f"  Min Q -> min: {min(min_qs):.4f}, max: {max(min_qs):.4f}, mean: {sum(min_qs) / len(min_qs):.4f}")
    print(f"  Avg Q spread (max-min per episode): {sum(mxq - mnq for mxq, mnq in zip(max_qs, min_qs)) / len(max_qs):.4f}")
    print()

    # Plateau detection
    if len(episodes) >= 6000:
        mid = episodes[len(episodes) // 2 - 500:len(episodes) // 2 + 500]
        late = episodes[-1000:]
        mid_avg = sum(e['mario_x_max'] for e in mid) / len(mid)
        late_avg = sum(e['mario_x_max'] for e in late) / len(late)
        mid_reward = sum(e['reward'] for e in mid) / len(mid)
        late_reward = sum(e['reward'] for e in late) / len(late)
        print(f"--- PLATEAU DETECTION ---")
        print(f"  Mid-training avg distance: {mid_avg:.1f}")
        print(f"  Late-training avg distance: {late_avg:.1f}")
        delta = late_avg - mid_avg
        print(f"  Delta: {delta:.1f} ({delta / max(1, mid_avg) * 100:.1f}%)")
        print(f"  Mid-training avg reward: {mid_reward:.1f}")
        print(f"  Late-training avg reward: {late_reward:.1f}")
        reward_delta = late_reward - mid_reward
        print(f"  Reward delta: {reward_delta:.1f} ({reward_delta / max(1, abs(mid_reward)) * 100:.1f}%)")
        if abs(delta) < mid_avg * 0.05:
            print(f"  >>> PLATEAU DETECTED - performance has stagnated <<<")
        elif delta > 0:
            print(f"  >>> IMPROVEMENT - performance is still growing <<<")
        else:
            print(f"  >>> REGRESSION - performance has declined <<<")
    elif len(episodes) >= 2000:
        early = episodes[:1000]
        late = episodes[-1000:]
        early_avg = sum(e['mario_x_max'] for e in early) / len(early)
        late_avg = sum(e['mario_x_max'] for e in late) / len(late)
        print(f"--- EARLY vs LATE COMPARISON ---")
        print(f"  First 1000 avg distance: {early_avg:.1f}")
        print(f"  Last 1000 avg distance:  {late_avg:.1f}")
        print(f"  Delta: {late_avg - early_avg:.1f}")

    print()
    print(f"{'=' * 100}")
    print(f"END OF ANALYSIS")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    analyze()
