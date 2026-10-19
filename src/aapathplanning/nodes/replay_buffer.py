#! /usr/bin/env python
# Authors deitieslulces #

import os
import pickle
import random
import numpy as np

# =========================
# Simple Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds buffer size {len(self.buffer)}")
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        os.makedirs('save_model', exist_ok=True)  # 修正：目录名一致
        if save_path is None:
            save_path = f"save_model/sac_buffer_{env_name}_{suffix}"
        print(f"Saving buffer to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print(f"Loading buffer from {save_path}")
        with open(save_path, 'rb') as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


# =========================
# PER: SumTree
# =========================
class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = list(np.zeros(capacity, dtype=object))

    def add(self, p, transition):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = transition
        self.update(tree_idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[cl_idx]:
                parent_idx = cl_idx
            else:
                v -= self.tree[cl_idx]
                parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]


# =========================
# PER Buffer (SumTree-based)
# 兼容你 sac_per.SAC 的现有调用：
#   - store(transition)
#   - sample(batch_size) -> (idx, b_memory, ISWeights)
#   - batch_update(idx, td_error)
# 额外加入专家数据录制 (s,a,s') 到 .npy
# =========================
class Per_ReplayBuffer(object):
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.0

    def __init__(self, state_dim=28, action_dim=2, size=100000):
        self.tree = SumTree(size)
        self.memory_num = 0
        self.memory_size = size

        # —— 专家数据录制 —— #
        self.expert_data = []                 # [(s,a,sp), ...]
        self.max_expert_data = 10000
        self.record_path = os.path.join("data", "sac_expert_data.npy")
        os.makedirs(os.path.dirname(self.record_path), exist_ok=True)

    def store(self, transition):
        # 兼容现有 SAC：transition = (s, a, r, sp, done)
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)
        if self.memory_num < self.memory_size:
            self.memory_num += 1

        # 追加专家数据 (s,a,sp)
        try:
            s, a, _, sp, _ = transition
            self.expert_data.append((
                np.asarray(s,  dtype=np.float32),
                np.asarray(a,  dtype=np.float32),
                np.asarray(sp, dtype=np.float32),
            ))
            if len(self.expert_data) >= self.max_expert_data:
                self.save_expert_data()
        except Exception as e:
            # 如果 transition 不是期望格式，忽略录制但不影响训练
            pass

    def sample(self, batch_size=32):
        n = batch_size
        b_idx = np.empty((n,), dtype=np.int32)
        ISWeights = np.empty((n, 1))
        b_memory = []

        pri_seg = self.tree.total_p / n
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / (self.tree.total_p + 1e-12)
        if min_prob <= 0:
            min_prob = 1e-5

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / (self.tree.total_p + 1e-12)
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors = np.asarray(abs_errors, dtype=np.float32) + self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return self.memory_num

    # —— 专家数据落盘：只导出 (s,s') 给 NEAR，也可按需改成保存 (s,a,sp) —— #
    def save_expert_data(self):
        arr = np.asarray(self.expert_data, dtype=object)  # [(s,a,sp), ...]
        s  = np.stack(arr[:, 0], axis=0)  # (N, S)
        sp = np.stack(arr[:, 2], axis=0)  # (N, S)
        pairs = np.concatenate([s, sp], axis=1).astype(np.float32)  # (N, 2S)
        np.save(self.record_path, pairs)
        print(f"[PER] Expert pairs saved -> {self.record_path} ({len(pairs)} samples)")
        self.expert_data.clear()


# =========================
# Optional: vectorized PER
# =========================
class PrioritizedReplay(object):
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        N = len(self.buffer)
        prios = self.priorities if N == self.capacity else self.priorities[:self.pos]
        probs = prios ** self.alpha
        P = probs / probs.sum()

        indices = np.random.choice(N, batch_size, p=P)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        weights = (N * P[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)

    def __len__(self):
        return len(self.buffer)
