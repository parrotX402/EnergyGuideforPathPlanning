#!/usr/bin/env python3
import os
import sys
import copy
import numpy as np
import rospy
from std_msgs.msg import Float32

# ===== Path setup =====
sys.path.insert(0, "/home/zhosuiyuan/桌面/src")
dirPath = os.path.dirname(os.path.realpath(__file__))

# ===== Output dirs/files =====
reward_dir = os.path.join(dirPath, "reward")
os.makedirs(reward_dir, exist_ok=True)

# Ablation CSV (你原来就有，保留)
ablation_csv = os.path.join(reward_dir, "ablation.csv")
if not os.path.exists(ablation_csv):
    with open(ablation_csv, "w") as f:
        f.write("tag,episodes,success_rate,collision_rate,avg_steps,near_mean\n")

# NEW: train log csv (用于画训练曲线)
train_csv = os.path.join(reward_dir, "train_log.csv")
if not os.path.exists(train_csv):
    with open(train_csv, "w") as f:
        f.write("episode,success,collision,steps,ep_reward,near_mean\n")

# ===== Imports that depend on sys.path =====
from aapathplanning.environment.environment_sac import Env
from sac_atten import SAC, action_unnormalized
from near_infer import NearEnergyModel

# ===== NEAR model =====
CKPT_PATH = "/home/zhosuiyuan/桌面/near/isaacgymenvs/learning/motion_ncsn/run/logs/near_run1/checkpoint.pth"
near_model = NearEnergyModel(CKPT_PATH, device="cpu")

# ===== NEAR reward config =====
USE_NEAR_REWARD = True
NEAR_ALPHA = 0.02      # NEAR shaping 系数
NEAR_SCALE = 500.0     # 你现在测到 energy~50，用这个 scale
CLIP_MIN = -1.0
CLIP_MAX =  1.0

# ===== training counters =====
total_steps = 0
START_STEPS = 5000
NEAR_START_STEPS = 8000


def _normalize_state_for_near(s: np.ndarray) -> np.ndarray:
    """Align with NEAR input preprocessing. Currently only normalize obstacle_angle (last dim)."""
    s = s.astype(np.float32, copy=True)
    s[-1] = s[-1] / 23.0
    return s


def make_x56(state, next_state) -> np.ndarray:
    s  = _normalize_state_for_near(np.asarray(state))
    sp = _normalize_state_for_near(np.asarray(next_state))
    return np.concatenate([s, sp], axis=0).astype(np.float32)  # (56,)


if __name__ == '__main__':
    rospy.init_node('main')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    _ = pub_result  # 保持兼容，不使用也没关系

    # ===== Basic config =====
    action_dim = 2
    state_dim = 28
    max_steps = 300
    num_episodes = 600

    past_action = np.array([0.0, 0.0], dtype=np.float32)

    # ===== Expert / dataset collection options =====
    EXPERT_MODE = False
    SAVE_EXPERT_ONLY = False
    expert_save_dir = os.path.join(dirPath, "near_expert_dataset")
    os.makedirs(expert_save_dir, exist_ok=True)
    expert_episode = []
    expert_file_idx = 0
    expert_min_len = 20

    # ===== Reward logs (txt) =====
    train_reward = np.empty((0, 3), dtype=float)
    test_reward = np.empty((0, 3), dtype=float)

    # ===== Env / Agent =====
    env = Env(action_dim)
    agent = SAC(state_dim, action_dim)

    # 你指定加载 episode 30 的模型
    policy_path = os.path.join(dirPath, "model/30_policy_net.pth")
    critic_path = os.path.join(dirPath, "model/30_value_net.pth")
    agent.load_models_by_path(policy_path, critic_path, device="cpu")

    # IMPORTANT: 不要再调用 agent.load_models()，否则可能覆盖掉你上面指定的路径
    agent.load_models()

    # ===== NEAR dataset collector (可选) =====
    SAVE_NEAR_DATA = False
    near_save_dir = "/home/zhosuiyuan/桌面/src/aapathplanning/nodes/v2_more_clean"
    os.makedirs(near_save_dir, exist_ok=True)

    near_chunk = []
    near_chunk_size = 10000
    near_file_idx = 0

    obstacle_min_range_list = []

    # ===== rolling success count for printing every 10 eps =====
    success_count_10 = 0

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        goalflag = False

        collision_count = 0      # 本回合是否撞了
        near_list = []           # 本回合 near_r 序列
        rewards_current_episode = 0.0

        for step in range(max_steps):
            state = np.float32(state)

            # ===== choose action =====
            if EXPERT_MODE:
                action = agent.select_action(state, eval=True)
            else:
                # 你原逻辑：每 10 轮用 eval=True
                if not ep % 10 == 0:
                    action = agent.select_action(state)
                else:
                    action = agent.select_action(state, eval=True)

            # ===== unnormalize action =====
            unnorm_action = np.array([
                action_unnormalized(action[0], 0.3, 0.0),
                action_unnormalized(action[1], 2.0, -1.0),
            ], dtype=np.float32)

            # ===== step env =====
            next_state, reward, done, goalflag, obstacle_min_range = env.step(unnorm_action, past_action)
            obstacle_min_range_list.append(obstacle_min_range)

            # ===== NEAR shaping =====
            near_r = 0.0
            if USE_NEAR_REWARD and (not done) and (not goalflag):
                x56 = make_x56(state, next_state)
                near_r = near_model.shaped_reward(
                    x56,
                    scale=NEAR_SCALE,
                    clip_min=CLIP_MIN,
                    clip_max=CLIP_MAX
                )
                reward = reward + NEAR_ALPHA * near_r

            near_list.append(float(near_r))

            # ===== accumulate episode reward (关键：你之前没加) =====
            rewards_current_episode += float(reward)

            # ===== collect NEAR training data (optional) =====
            if SAVE_NEAR_DATA:
                s_near = _normalize_state_for_near(state)
                sp_near = _normalize_state_for_near(next_state)
                near_chunk.append(np.concatenate([s_near, sp_near], axis=0))

                # Expert collector
                if EXPERT_MODE:
                    expert_episode.append(np.concatenate([s_near, sp_near], axis=0))

                if len(near_chunk) >= near_chunk_size:
                    out_path = os.path.join(near_save_dir, f"near_X_{near_file_idx:04d}.npy")
                    np.save(out_path, np.stack(near_chunk, axis=0))
                    print(f"[NEAR] saved {len(near_chunk)} transitions -> {out_path}")
                    near_chunk.clear()
                    near_file_idx += 1

            # ===== store transition and update =====
            transition = (state, action, reward, next_state, done)
            agent.store_transition(transition)

            bs = agent.buffer_size()
            if bs % 100 == 0:
                print(f"[CHECK] buffer_size = {bs}")

            # 你原本写了两次 update，这里保留一次即可
            if bs >= 256:
                agent.update_parameters(agent.batch_size)
                if total_steps % 200 == 0:
                    print("[TRAIN] update called, buffer =", bs)

            # ===== update state =====
            state = copy.deepcopy(next_state)
            total_steps += 1

            # ===== termination =====
            if done or goalflag:
                if goalflag:
                    success_count_10 += 1
                else:
                    collision_count = 1  # 修复：done 且非 goal 就算 collision
                break

        # ===== per-episode stats =====
        near_mean = float(np.mean(near_list)) if len(near_list) > 0 else 0.0
        ep_steps = step + 1

        print(
            f"[STAT] ep={ep} "
            f"success={int(goalflag)} "
            f"collision={int(collision_count)} "
            f"steps={ep_steps} "
            f"ep_reward={rewards_current_episode:.3f} "
            f"near_mean={near_mean:.3f}"
        )

        # ===== write training log csv (关键) =====
        with open(train_csv, "a") as f:
            f.write(f"{ep},{int(goalflag)},{int(collision_count)},{ep_steps},{rewards_current_episode:.3f},{near_mean:.3f}\n")

        # ===== Expert episode save =====
        if EXPERT_MODE:
            success = bool(goalflag)
            if (not SAVE_EXPERT_ONLY) or success:
                if len(expert_episode) >= expert_min_len:
                    out_path = os.path.join(expert_save_dir, f"expert_X_{expert_file_idx:04d}.npy")
                    np.save(out_path, np.stack(expert_episode, axis=0))
                    print(f"[EXPERT] saved episode len={len(expert_episode)} success={success} -> {out_path}")
                    expert_file_idx += 1
        expert_episode.clear()

        # ===== print success rate every 10 eps =====
        if (ep + 1) % 10 == 0:
            success_rate = (success_count_10 / 10.0) * 100.0
            print(f"Success Rate after {ep + 1} episodes: {success_rate:.1f}%")
            success_count_10 = 0

        # ===== save reward txt (你原本的格式：ep, ep_reward, step) =====
        train_reward = np.append(train_reward, np.asarray([[ep, rewards_current_episode, ep_steps]]), axis=0)
        np.savetxt(os.path.join(reward_dir, "sac_reward.txt"), train_reward)

        # ===== save model every 10 eps (你的逻辑) =====
        if (ep + 1) % 200 == 0:
            print(f"Saving model at episode {ep + 1}")
            agent.save_models(ep + 1)
        
        if (ep + 1) % 50 == 0:
            print(f"[INFO] ep={ep+1} buffer={agent.buffer_size()}")


        # ===== flush remaining NEAR data chunk (optional) =====
        if SAVE_NEAR_DATA and len(near_chunk) > 0:
            out_path = os.path.join(near_save_dir, f"near_X_{near_file_idx:04d}.npy")
            np.save(out_path, np.stack(near_chunk, axis=0))
            print(f"[NEAR] final save {len(near_chunk)} transitions -> {out_path}")
            near_chunk.clear()
            near_file_idx += 1

        # ===== optional reward debug =====
        if ep_steps % 50 == 0:
            # r_env = reward - NEAR_ALPHA * near_r  # 这里 reward 是最后一步 reward，不代表回合平均，仅做参考
            print(f"[INFO] ep={ep} steps={ep_steps} buf={agent.buffer_size()}")

    print("[INFO] Training finished. Logs saved to:")
    print("  -", train_csv)
    print("  -", os.path.join(reward_dir, "sac_reward.txt"))
