"""Integration reference for the original ROS1 loop; no ROS imports or auto-start.

Call after loading the existing agent/environment yourself. The supplied env can
also be a fake environment for a unit/integration check. This is not a new policy
optimizer: agent.replay_buffer.push/update follow the original SAC API.
"""

from energyguide.integration import RewardAdapter, bootstrap_terminal
from energyguide.model import FrozenReward


def reward_components(checkpoint, state, physical_action, next_state, env_reward,
                      collision=False, success=False):
    adapter = RewardAdapter(FrozenReward(checkpoint), alpha=0.002)
    return adapter.transition(state, physical_action, next_state, env_reward,
                              terminated=bootstrap_terminal(collision, success))


# In the real training loop, create FrozenReward/RewardAdapter ONCE, outside the loop:
#
# recorder = DemonstrationRecorder(velocity_source="command")  # Map A collection
# state = env.reset()
# normalized_action = agent.select_action(state)
# physical_action = action_unnormalized(normalized_action, high, low)
# next_state, r_env, collided, reached_goal, _ = env.step(physical_action, past_action)
# recorder.append(physical_action[0], physical_action[1], state[24], state[26])
# terminal = bootstrap_terminal(collided, reached_goal)
# r_total, r_eg = adapter.transition(state, physical_action, next_state, r_env, terminal)
# agent.replay_buffer.push(state, normalized_action, r_total, next_state, terminal)
#
# On success, collision, or 300-step timeout:
# recorder.finish_episode(success=reached_goal, collision=collided)
# After enough completed Map A episodes, including successes and failures:
# recorder.save("demonstrations_map_a.npz")
#
# Note: the initial 20,000-episode SAC expert uses environmental rewards only.
# Collection, energy pretraining, and policy training are distinct stages.
