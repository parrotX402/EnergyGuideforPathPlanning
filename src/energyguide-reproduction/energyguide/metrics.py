"""Raw episode aggregates; no paper numbers are copied into measured outputs."""

import numpy as np


def summarize(episodes):
    if not episodes:
        raise ValueError("No evaluation episodes")
    for e in episodes:
        if e["success"] and e["collision"]:
            raise ValueError("Episode cannot be both a success and collision")
        if e["steps"] <= 0:
            raise ValueError("steps must be positive")
    successful = [e for e in episodes if e["success"]]
    return {
        "episodes": len(episodes),
        "success_rate": sum(bool(e["success"]) for e in episodes) / len(episodes),
        "collision_rate": sum(bool(e["collision"]) for e in episodes) / len(episodes),
        "timeout_rate": sum(not e["success"] and not e["collision"] for e in episodes) / len(episodes),
        "mean_steps_success": float(np.mean([e["steps"] for e in successful])) if successful else None,
    }
