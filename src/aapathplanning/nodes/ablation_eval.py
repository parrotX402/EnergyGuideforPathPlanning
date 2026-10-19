#!/usr/bin/env python3
import os
import sys
import copy
import numpy as np
import rospy
from std_msgs.msg import Float32

# ====== 1) Path setup ======
sys.path.insert(0, "/home/zhosuiyuan/桌面/src")
dirPath = os.path.dirname(os.path.realpath(__file__))

# ====== 2) Output csv setup ======
reward_dir = os.path.join(dirPath, "reward")
os.makedirs(reward_dir, exist_ok=True)

# 建议新建一个文件，避免和旧表头不一致
ablation_csv = os.path.join(reward_dir, "复杂地图_alpha_sweep.csv")
if not os.path.exists(ablation_csv):
    with open(ablation_csv, "w") as f:
        # 表头：不保留 tag，只保留 alpha 数值列
        f.write("alpha,episodes,success_rate,collision_rate,avg_steps_success,near_mean\n")

# ====== 3) Imports that depend on sys.path ======
from aapathplanning.environment.environment_sac import Env
from sac_atten import SAC, action_unnormalized
from near_infer import NearEnergyModel

# ====== 4) NEAR model ======
CKPT_PATH = "/home/zhosuiyuan/桌面/near/isaacgymenvs/learning/motion_ncsn/run/logs/near_run1/checkpoint.pth"
near_model = NearEnergyModel(CKPT_PATH, device="cpu")

# ====== 5) Ablation configs: alpha sweep ======
ALPHAS = [0.006, 0.007, 0.008, 0.009, 0.010]  # 你要跑的 alpha
NEAR_SCALE = 3.0
CLIP = (-0.2, 0.0)

ABLATIONS = []
for a in ALPHAS:
    if a == 0.0:
        ABLATIONS.append(dict(tag="A_no_near", use_near=False, alpha=0.0, scale=NEAR_SCALE, clip=CLIP))
    else:
        tag = f"B_a{int(round(a * 1000)):04d}"  # 0.002->a0002, 0.010->a0010
        ABLATIONS.append(dict(tag=tag, use_near=True, alpha=float(a), scale=NEAR_SCALE, clip=CLIP))

EVAL_EPISODES = 20
MAX_STEPS = 200

# ====== 6) NEAR input normalisation ======
def _normalize_state_for_near(s: np.ndarray) -> np.ndarray:
    """Align with NEAR training preprocessing. Currently only normalize obstacle_angle (last dim)."""
    s = s.astype(np.float32, copy=True)
    s[-1] = s[-1] / 23.0
    return s

def make_x56(state, next_state) -> np.ndarray:
    s  = _normalize_state_for_near(np.asarray(state))
    sp = _normalize_state_for_near(np.asarray(next_state))
    return np.concatenate([s, sp], axis=0).astype(np.float32)  # (56,)

def main():
    rospy.init_node("ablation_eval")
    pub_result = rospy.Publisher("result", Float32, queue_size=5)
    _ = pub_result  # not used, keep for compatibility

    action_dim = 2
    state_dim = 28
    past_action = np.array([0.0, 0.0], dtype=np.float32)

    # ====== Env / Agent ======
    env = Env(action_dim)
    agent = SAC(state_dim, action_dim)

    # ====== Load your trained model (episode 30) ======
    policy_path = os.path.join(dirPath, "model/30_policy_net.pth")
    critic_path = os.path.join(dirPath, "model/30_value_net.pth")

    # IMPORTANT: do NOT call agent.load_models() (it loads Tsac/1500 and overrides)
    agent.load_models_by_path(policy_path, critic_path, device="cpu")

    print("\n[INFO] Loaded model for ablation eval:")
    print("  policy:", policy_path)
    print("  critic:", critic_path)
    print("  csv   :", ablation_csv)

    # ====== Run ablations ======
    for cfg in ABLATIONS:
        USE_NEAR_REWARD = bool(cfg["use_near"])
        NEAR_ALPHA = float(cfg["alpha"])
        NEAR_SCALE = float(cfg["scale"])
        CLIP_MIN, CLIP_MAX = cfg["clip"]

        success_count = 0
        collision_count = 0
        steps_success_list = []
        near_mean_list = []

        print("\n" + "=" * 90)
        print(f"[ABLATION] {cfg['tag']} | alpha={NEAR_ALPHA:.3f} use_near={USE_NEAR_REWARD} scale={NEAR_SCALE} clip={cfg['clip']}")
        print("=" * 90)

        for ep in range(EVAL_EPISODES):
            state = env.reset()
            done = False
            goalflag = False
            near_list = []

            steps_this_ep = 0
            collided_this_ep = 0

            for step in range(MAX_STEPS):
                state = np.float32(state)

                # eval-only action
                action = agent.select_action(state, eval=True)

                unnorm_action = np.array([
                    action_unnormalized(action[0], 0.3, 0.0),
                    action_unnormalized(action[1], 2.0, -1.0),
                ], dtype=np.float32)

                next_state, reward, done, goalflag, obstacle_min_range = env.step(unnorm_action, past_action)

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

                state = copy.deepcopy(next_state)
                steps_this_ep = step + 1

                if done or goalflag:
                    if goalflag:
                        success_count += 1
                        steps_success_list.append(steps_this_ep)
                    else:
                        collided_this_ep = 1
                    break

            collision_count += collided_this_ep
            near_mean = float(np.mean(near_list)) if len(near_list) > 0 else 0.0
            near_mean_list.append(near_mean)

            print(f"[STAT] alpha={NEAR_ALPHA:.3f} ep={ep:03d} success={int(goalflag)} collision={collided_this_ep} steps={steps_this_ep} near_mean={near_mean:.3f}")

        # ====== Summarize ======
        episodes = EVAL_EPISODES
        success_rate = success_count / episodes
        collision_rate = collision_count / episodes
        avg_steps_success = float(np.mean(steps_success_list)) if len(steps_success_list) > 0 else -1.0
        near_mean_all = float(np.mean(near_mean_list)) if len(near_mean_list) > 0 else 0.0

        print("\n[RESULT]")
        print(f"  alpha            : {NEAR_ALPHA:.3f}")
        print(f"  episodes         : {episodes}")
        print(f"  success_rate     : {success_rate:.3f}")
        print(f"  collision_rate   : {collision_rate:.3f}")
        print(f"  avg_steps_success: {avg_steps_success:.2f}")
        print(f"  near_mean        : {near_mean_all:.3f}")

        # ====== Append to CSV ======
        with open(ablation_csv, "a") as f:
            f.write(f"{NEAR_ALPHA:.3f},{episodes},{success_rate:.3f},{collision_rate:.3f},{avg_steps_success:.2f},{near_mean_all:.3f}\n")

        print(f"[CSV] appended -> {ablation_csv}")

    print("\n[DONE] All ablations finished.")

if __name__ == "__main__":
    main()