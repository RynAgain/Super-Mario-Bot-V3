"""
Experience Replay Buffer for Super Mario Bros AI Training

This module implements a circular buffer for storing and sampling experiences
for DQN training. Supports both uniform and prioritized sampling strategies.

Features:
- Circular buffer for memory efficiency
- Priority sampling support (optional)
- Batch sampling for training
- Memory management for large datasets
- GPU-compatible tensor operations
"""

import torch
import numpy as np
from collections import namedtuple, deque
from typing import List, Tuple, Optional, Union
import random


# Experience tuple for storing transitions
Experience = namedtuple('Experience', [
    'state_frames',      # Current frame stack (8, 84, 84)
    'state_vector',      # Current game state vector (12,)
    'action',            # Action taken
    'reward',            # Reward received
    'next_state_frames', # Next frame stack (8, 84, 84)
    'next_state_vector', # Next game state vector (12,)
    'done',              # Episode termination flag
    'info'               # Additional information (optional)
])


class ReplayBuffer:
    """
    Circular buffer for storing and sampling experiences.
    
    This implementation uses a circular buffer to efficiently manage memory
    and provides both uniform and prioritized sampling capabilities.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        frame_stack_size: int = 8,
        frame_size: Tuple[int, int] = (84, 84),
        state_vector_size: int = 12,
        device: str = "cpu",
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            frame_stack_size: Number of frames in each stack (8)
            frame_size: Frame dimensions (84, 84)
            state_vector_size: Size of game state vector (12)
            device: Device for tensor operations ("cpu" or "cuda")
            prioritized: Enable prioritized experience replay
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: Beta increment per sampling step
        """
        self.capacity = capacity
        self.frame_stack_size = frame_stack_size
        self.frame_size = frame_size
        self.state_vector_size = state_vector_size
        self.device = torch.device(device)
        self.prioritized = prioritized
        
        # Prioritized replay parameters
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # Initialize storage
        self._initialize_storage()
        
        # Buffer state
        self.position = 0
        self.size = 0
        
        # Priority tree for prioritized sampling
        if self.prioritized:
            self._initialize_priority_tree()
    
    def _initialize_storage(self):
        """Initialize storage tensors for efficient memory usage."""
        # Pre-allocate tensors for better memory efficiency
        self.state_frames = torch.zeros(
            (self.capacity, self.frame_stack_size, *self.frame_size),
            dtype=torch.float32,
            device=self.device
        )
        
        self.state_vectors = torch.zeros(
            (self.capacity, self.state_vector_size),
            dtype=torch.float32,
            device=self.device
        )
        
        self.actions = torch.zeros(
            self.capacity,
            dtype=torch.long,
            device=self.device
        )
        
        self.rewards = torch.zeros(
            self.capacity,
            dtype=torch.float32,
            device=self.device
        )
        
        self.next_state_frames = torch.zeros(
            (self.capacity, self.frame_stack_size, *self.frame_size),
            dtype=torch.float32,
            device=self.device
        )
        
        self.next_state_vectors = torch.zeros(
            (self.capacity, self.state_vector_size),
            dtype=torch.float32,
            device=self.device
        )
        
        self.dones = torch.zeros(
            self.capacity,
            dtype=torch.bool,
            device=self.device
        )
    
    def _initialize_priority_tree(self):
        """Initialize sum tree for prioritized sampling."""
        # Sum tree for efficient priority sampling
        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2
        
        self.tree_capacity = tree_capacity
        self.tree = np.zeros(2 * tree_capacity - 1)
        self.min_tree = np.full(2 * tree_capacity - 1, float('inf'))
    
    def add(
        self,
        state_frames: Union[torch.Tensor, np.ndarray],
        state_vector: Union[torch.Tensor, np.ndarray],
        action: int,
        reward: float,
        next_state_frames: Union[torch.Tensor, np.ndarray],
        next_state_vector: Union[torch.Tensor, np.ndarray],
        done: bool,
        info: Optional[dict] = None
    ):
        """
        Add a new experience to the buffer.
        
        Args:
            state_frames: Current frame stack
            state_vector: Current game state vector
            action: Action taken
            reward: Reward received
            next_state_frames: Next frame stack
            next_state_vector: Next game state vector
            done: Episode termination flag
            info: Additional information (optional)
        """
        # Convert to tensors if necessary
        if isinstance(state_frames, np.ndarray):
            state_frames = torch.from_numpy(state_frames).float()
        if isinstance(state_vector, np.ndarray):
            state_vector = torch.from_numpy(state_vector).float()
        if isinstance(next_state_frames, np.ndarray):
            next_state_frames = torch.from_numpy(next_state_frames).float()
        if isinstance(next_state_vector, np.ndarray):
            next_state_vector = torch.from_numpy(next_state_vector).float()
        
        # Store experience
        self.state_frames[self.position] = state_frames.to(self.device)
        self.state_vectors[self.position] = state_vector.to(self.device)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_state_frames[self.position] = next_state_frames.to(self.device)
        self.next_state_vectors[self.position] = next_state_vector.to(self.device)
        self.dones[self.position] = done
        
        # Update priority for prioritized replay
        if self.prioritized:
            priority = self.max_priority ** self.alpha
            self._update_priority(self.position, priority)
        
        # Update buffer state
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of tensors: (state_frames, state_vectors, actions, rewards,
                             next_state_frames, next_state_vectors, dones, weights, indices)
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough experiences in buffer: {self.size} < {batch_size}")
        
        if self.prioritized:
            indices, weights = self._sample_prioritized(batch_size)
        else:
            indices = self._sample_uniform(batch_size)
            weights = torch.ones(batch_size, device=self.device)
        
        # Extract experiences
        state_frames = self.state_frames[indices]
        state_vectors = self.state_vectors[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_state_frames = self.next_state_frames[indices]
        next_state_vectors = self.next_state_vectors[indices]
        dones = self.dones[indices]
        
        return (
            state_frames,
            state_vectors,
            actions,
            rewards,
            next_state_frames,
            next_state_vectors,
            dones,
            weights,
            indices
        )
    
    def _sample_uniform(self, batch_size: int) -> torch.Tensor:
        """Sample indices uniformly."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return indices
    
    def _sample_prioritized(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample indices using prioritized sampling."""
        indices = []
        weights = []
        
        # Sample from priority tree
        segment = self.tree[0] / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx = self._retrieve(0, s)
            indices.append(idx)
            
            # Calculate importance sampling weight
            prob = self.tree[idx + self.tree_capacity - 1] / self.tree[0]
            weight = (self.size * prob) ** (-self.beta)
            weights.append(weight)
        
        # Normalize weights
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            torch.tensor(indices, device=self.device),
            torch.tensor(weights, dtype=torch.float32, device=self.device)
        )
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities for prioritized replay."""
        if not self.prioritized:
            return
        
        for idx, priority in zip(indices.cpu().numpy(), priorities.cpu().numpy()):
            priority = max(priority, 1e-6)  # Avoid zero priorities
            self.max_priority = max(self.max_priority, priority)
            self._update_priority(idx, priority ** self.alpha)
    
    def _update_priority(self, idx: int, priority: float):
        """Update priority in sum tree."""
        tree_idx = idx + self.tree_capacity - 1
        
        # Update sum tree
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
        
        # Update min tree
        tree_idx = idx + self.tree_capacity - 1
        self.min_tree[tree_idx] = priority
        
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.min_tree[tree_idx] = min(
                self.min_tree[2 * tree_idx + 1],
                self.min_tree[2 * tree_idx + 2]
            )
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample index from sum tree."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return self.size >= batch_size
    
    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0
        if self.prioritized:
            self.tree.fill(0)
            self.min_tree.fill(float('inf'))
    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics."""
        frame_memory = self.state_frames.element_size() * self.state_frames.numel()
        frame_memory += self.next_state_frames.element_size() * self.next_state_frames.numel()
        
        vector_memory = self.state_vectors.element_size() * self.state_vectors.numel()
        vector_memory += self.next_state_vectors.element_size() * self.next_state_vectors.numel()
        
        other_memory = (
            self.actions.element_size() * self.actions.numel() +
            self.rewards.element_size() * self.rewards.numel() +
            self.dones.element_size() * self.dones.numel()
        )
        
        total_memory = frame_memory + vector_memory + other_memory
        
        return {
            'total_mb': total_memory / (1024 * 1024),
            'frame_memory_mb': frame_memory / (1024 * 1024),
            'vector_memory_mb': vector_memory / (1024 * 1024),
            'other_memory_mb': other_memory / (1024 * 1024),
            'capacity': self.capacity,
            'size': self.size,
            'utilization': self.size / self.capacity
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """Convenience class for prioritized replay buffer."""
    
    def __init__(self, *args, **kwargs):
        kwargs['prioritized'] = True
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    # Test the replay buffer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = ReplayBuffer(capacity=1000, device=device)
    
    # Add some dummy experiences
    for i in range(100):
        state_frames = torch.randn(8, 84, 84)
        state_vector = torch.randn(12)
        action = np.random.randint(0, 12)
        reward = np.random.randn()
        next_state_frames = torch.randn(8, 84, 84)
        next_state_vector = torch.randn(12)
        done = np.random.random() < 0.1
        
        buffer.add(
            state_frames, state_vector, action, reward,
            next_state_frames, next_state_vector, done
        )
    
    # Test sampling
    if buffer.is_ready(32):
        batch = buffer.sample(32)
        print(f"Sampled batch shapes:")
        print(f"State frames: {batch[0].shape}")
        print(f"State vectors: {batch[1].shape}")
        print(f"Actions: {batch[2].shape}")
        print(f"Rewards: {batch[3].shape}")
        print(f"Next state frames: {batch[4].shape}")
        print(f"Next state vectors: {batch[5].shape}")
        print(f"Dones: {batch[6].shape}")
        print(f"Weights: {batch[7].shape}")
        print(f"Indices: {batch[8].shape}")
    
    # Test memory usage
    memory_stats = buffer.get_memory_usage()
    print(f"\nMemory usage: {memory_stats}")
    
    # Test prioritized buffer
    print("\nTesting prioritized buffer...")
    prioritized_buffer = PrioritizedReplayBuffer(capacity=1000, device=device)
    
    # Add experiences
    for i in range(100):
        state_frames = torch.randn(8, 84, 84)
        state_vector = torch.randn(12)
        action = np.random.randint(0, 12)
        reward = np.random.randn()
        next_state_frames = torch.randn(8, 84, 84)
        next_state_vector = torch.randn(12)
        done = np.random.random() < 0.1
        
        prioritized_buffer.add(
            state_frames, state_vector, action, reward,
            next_state_frames, next_state_vector, done
        )
    
    # Test prioritized sampling
    if prioritized_buffer.is_ready(32):
        batch = prioritized_buffer.sample(32)
        print(f"Prioritized sampling successful!")
        print(f"Weights range: {batch[7].min().item():.4f} - {batch[7].max().item():.4f}")