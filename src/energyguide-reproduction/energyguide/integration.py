"""ROS-independent reward integration; no velocity publisher or device actions."""

import math

import numpy as np

from .data import feature4, legacy56


class RewardAdapter:
    def __init__(self, frozen_reward, alpha=0.002, terminal_shaping=False):
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.frozen_reward = frozen_reward
        self.alpha = alpha
        # Old implementation skipped terminal shaping. Explicit implementation choice.
        self.terminal_shaping = terminal_shaping

    def transition(self, state, action_physical, next_state, environment_reward,
                   terminated=False, measured_velocity=None):
        """States follow old ROS ordering: 24 lidar + heading,distance,clearance,index.

        paper4 uses pre-action local geometry. By default action commands approximate
        induced velocity; supply synchronized measured_velocity=(v,omega) if available.
        Actions are physical m/s and rad/s, never normalized policy outputs.
        """
        state, next_state = np.asarray(state), np.asarray(next_state)
        if state.shape != (28,) or next_state.shape != (28,):
            raise ValueError("Expected 28D states")
        if self.alpha == 0 or (terminated and not self.terminal_shaping):
            return float(environment_reward), 0.0
        representation = self.frozen_reward.model.config.representation
        if representation == "paper4":
            v, w = action_physical if measured_velocity is None else measured_velocity
            features = feature4(v, w, state[24], state[26])
        else:
            features = legacy56(state, next_state)
        shaping = self.frozen_reward(features)
        return float(environment_reward + self.alpha * shaping), shaping


def bootstrap_terminal(collision, success):
    """Use this replay mask. A 300-step time limit alone is a truncation."""
    return bool(collision or success)
