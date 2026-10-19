"""Collect the missing four features at source; preserve failed trajectories too."""

from pathlib import Path

import numpy as np

from .data import feature4


class DemonstrationRecorder:
    def __init__(self, velocity_source="command"):
        if velocity_source not in ("command", "measured"):
            raise ValueError("velocity_source must be command or measured")
        self.velocity_source = velocity_source
        self.features, self.labels, self.episodes = [], [], []
        self.pending = []
        self.episode_count = 0

    def append(self, v, omega, heading, clearance):
        self.pending.append(feature4(v, omega, heading, clearance))

    def finish_episode(self, success, collision=False):
        if success and collision:
            raise ValueError("Episode outcome is ambiguous")
        if not self.pending:
            raise ValueError("No transitions recorded")
        # Failure includes both collision and timeout, as stated by the manuscript.
        self.features.extend(self.pending)
        self.labels.extend([int(bool(success))] * len(self.pending))
        self.episodes.extend([self.episode_count] * len(self.pending))
        self.episode_count += 1
        self.pending = []

    def save(self, path):
        if self.pending:
            raise ValueError("Finish the current episode before saving")
        if not self.features:
            raise ValueError("No completed demonstrations")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create: never overwrite an existing demonstration collection.
        with path.open("xb") as handle:
            np.savez_compressed(handle, features=np.asarray(self.features, dtype=np.float32),
                                success=np.asarray(self.labels, dtype=np.int64),
                                episode_id=np.asarray(self.episodes, dtype=np.int64),
                                map_id=np.full(len(self.features), "A"),
                                velocity_source=np.array(self.velocity_source))
