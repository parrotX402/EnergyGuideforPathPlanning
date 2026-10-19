"""Tests target scientific failure modes rather than implementation snapshots."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from energyguide.data import feature4, group_split, legacy56, load_dataset, sample_indices
from energyguide.integration import RewardAdapter, bootstrap_terminal
from energyguide.metrics import summarize
from energyguide.model import EnergyConfig, EnergyMLP, FrozenReward, dsm_loss, save_checkpoint
from energyguide.recording import DemonstrationRecorder


class PointMassPotential(nn.Module):
    """Exact -log p for clean x=0 convolved with Gaussian sigma."""
    def __init__(self, literal=False):
        super().__init__()
        self.config = EnergyConfig(hidden=(8,), convention="paper_literal" if literal else "negative_log_density")

    def forward(self, x, level):
        sigma = x.new_tensor(self.config.sigmas)[level]
        energy = x.square().sum(-1) / (2 * sigma.square())
        return -energy if self.config.convention == "paper_literal" else energy


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dsm_sign_against_analytic_gaussian(self):
        clean = torch.zeros(12, 4)
        levels = torch.arange(3).repeat(4)
        noise = torch.linspace(-1, 1, 48).reshape(12, 4)
        for literal in (False, True):
            self.assertLess(dsm_loss(PointMassPotential(literal), clean, levels, noise).item(), 1e-8)
        # Using positive energy with the displayed '+' sign must fail this oracle.
        wrong = PointMassPotential()
        wrong.config.convention = "paper_literal"
        wrong.forward = lambda x, level: x.square().sum(-1) / (2 * x.new_tensor(wrong.config.sigmas)[level].square())
        self.assertGreater(dsm_loss(wrong, clean, levels, noise).item(), 0.1)

    def test_optimizer_receives_second_derivative_gradients(self):
        model = EnergyMLP(EnergyConfig(hidden=(16, 16)))
        original = model.layers[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = dsm_loss(model, torch.randn(32, 4))
        optimizer.zero_grad()
        loss.backward()
        self.assertTrue(torch.isfinite(model.layers[0].weight.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(original, model.layers[0].weight))

    def test_input_refuses_56_to_4_invention(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "states.npy"
            np.save(p, np.zeros((5, 56), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "velocities"):
                load_dataset(p, "paper4", allow_unlabelled=True)

    def test_recorder_preserves_failed_episodes_and_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = DemonstrationRecorder()
            recorder.append(0.15, 0.1, 0.2, 0.5)
            with self.assertRaises(ValueError):
                recorder.save(Path(folder) / "unfinished.npz")
            recorder.finish_episode(success=True)
            recorder.append(0.0, 0.0, 0.0, 0.1)
            recorder.finish_episode(success=False, collision=True)
            path = Path(folder) / "map_a.npz"
            recorder.save(path)
            x, labels, groups, _ = load_dataset(path, "paper4")
            self.assertEqual(x.shape, (2, 4))
            np.testing.assert_array_equal(labels, [1, 0])
            self.assertEqual(len(np.unique(groups)), 2)
            with self.assertRaises(FileExistsError):
                recorder.save(path)

    def test_map_b_is_rejected_for_energy_training(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "data.npz"
            np.savez(p, features=np.zeros((2, 4)), success=[0, 1], map_id=["A", "B"])
            with self.assertRaisesRegex(ValueError, "Map A"):
                load_dataset(p, "paper4")

    def test_group_holdout_prevents_episode_leakage(self):
        groups = np.repeat(np.arange(10), 10)
        train, held = group_split(groups, seed=7)
        self.assertFalse(set(groups[train]) & set(groups[held]))
        self.assertEqual(len(train) + len(held), len(groups))

    def test_success_failure_resampling(self):
        labels = np.array([1] * 90 + [0] * 10)
        idx = sample_indices(labels, 100000, np.random.default_rng(0))
        self.assertAlmostEqual(float(labels[idx].mean()), 0.7, delta=0.005)
        with self.assertRaises(ValueError):
            sample_indices(np.ones(20), 10, np.random.default_rng(0))

    def test_features_use_physical_actions(self):
        f = feature4(0.2, -0.5, 3 * np.pi, 0.3)
        np.testing.assert_allclose(f, [0.2, -0.5, -np.pi, 0.3], atol=1e-6)
        with self.assertRaises(ValueError):
            feature4(1.0, 0, 0, 0.3)
        s = np.arange(28, dtype=np.float32)
        self.assertAlmostEqual(float(legacy56(s, s)[27]), 27 / 23, places=6)
        self.assertEqual(s[-1], 27)

    def test_checkpoint_roundtrip_freezing_reward_and_terminal_handling(self):
        with tempfile.TemporaryDirectory() as folder:
            model = EnergyMLP(EnergyConfig(hidden=(8, 8)))
            path = Path(folder) / "energy.pt"
            save_checkpoint(path, model, energy_reference=0.0)
            reward = FrozenReward(path)
            before = {k: v.clone() for k, v in reward.model.state_dict().items()}
            state = np.zeros(28, dtype=np.float32)
            state[26] = 0.5
            adapter = RewardAdapter(reward, alpha=0.002)
            total, shaped = adapter.transition(state, [0.2, 0.1], state, -10)
            self.assertGreater(shaped, 0)
            self.assertLessEqual(shaped, 1)
            self.assertAlmostEqual(total, -10 + 0.002 * shaped)
            self.assertEqual(adapter.transition(state, [0.2, 0.1], state, -650, True), (-650, 0))
            self.assertFalse(any(p.requires_grad for p in reward.model.parameters()))
            for name, value in before.items():
                self.assertTrue(torch.equal(value, reward.model.state_dict()[name]))
            self.assertTrue(bootstrap_terminal(True, False))
            self.assertTrue(bootstrap_terminal(False, True))
            self.assertFalse(bootstrap_terminal(False, False))  # time-limit bootstrap

    def test_literal_mapping_does_not_pretend_to_be_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            model = EnergyMLP(EnergyConfig(hidden=(8,)))
            for p in model.parameters():
                nn.init.zeros_(p)
            model.layers[-1].bias.data.fill_(-2)
            path = Path(folder) / "energy.pt"
            save_checkpoint(path, model)
            x = np.array([0.1, 0, 0, 0.5], dtype=np.float32)
            self.assertGreater(FrozenReward(path, mapping="literal")(x), 1)
            self.assertLessEqual(FrozenReward(path)(x), 1)
            with self.assertRaises(ValueError):
                FrozenReward(path, beta=0)

    def test_metrics_use_all_episodes_and_null_when_no_success(self):
        rows = [dict(success=True, collision=False, steps=10),
                dict(success=False, collision=True, steps=20),
                dict(success=False, collision=False, steps=300)]
        result = summarize(rows)
        self.assertAlmostEqual(result["success_rate"], 1 / 3)
        self.assertAlmostEqual(result["collision_rate"], 1 / 3)
        self.assertEqual(result["mean_steps_success"], 10)
        self.assertIsNone(summarize(rows[1:])["mean_steps_success"])


if __name__ == "__main__":
    unittest.main()
