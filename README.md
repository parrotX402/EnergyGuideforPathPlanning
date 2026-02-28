# Energy_Guided_for_PathPlanning
# Energy-Guided Reward Shaping (EG) for Robust & Generalizable DRL Path Planning

Code release for the paper:

**Energy-Guided Reward Shaping for Robust and Generalizable Deep Reinforcement Learning in Dynamic Robot Path Planning** :contentReference[oaicite:1]{index=1}

EG is an IRL-inspired, **plug-and-play reward shaping** framework. We train a **transition-level energy model** from expert demonstrations, then **freeze** it and convert energy into dense shaping rewards:
- local features: ξ = [v, ω, θ_rel, d_obs] :contentReference[oaicite:2]{index=2}  
- shaping reward: r_eg = exp( -E_ϕ(s,a) / σ ) :contentReference[oaicite:3]{index=3}  
- final reward: r_total = r_env + α · r_eg :contentReference[oaicite:4]{index=4}  

EG is integrated into **SAC / TD3 / GAP-SAC** and improves convergence, success rate, collision rate, and trajectory smoothness, with **zero-shot generalization** to unseen complex maps. :contentReference[oaicite:5]{index=5}

---

## Highlights

- **Pre-train & freeze**: energy model is trained offline and remains fixed during RL training, avoiding adversarial instability. :contentReference[oaicite:6]{index=6}  
- **Environment-agnostic priors**: reward depends only on local transition features (not global map layout), supporting transfer/generalization. :contentReference[oaicite:7]{index=7}  
- **Zero-shot evaluation**: energy is trained from expert data collected only in Map A; no additional demos are used in Map B testing. :contentReference[oaicite:8]{index=8}  

---

## Environment

Experiments are conducted in **ROS–Gazebo** with a TurtleBot3 differential-drive model. :contentReference[oaicite:9]{index=9}

- OS: Ubuntu 20.04
- ROS: (e.g., Noetic)
- Python: 3.8+
- GPU: recommended for energy model training

---

## Installation

```bash
# clone
git clone https://github.com/<your-org-or-username>/<repo>.git
cd <repo>

# (recommended) create env
conda create -n eg-nav python=3.8 -y
conda activate eg-nav

# install python deps
pip install -r requirements.txt
