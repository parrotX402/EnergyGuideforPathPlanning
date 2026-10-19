"""CPU-capable offline reconstruction CLI. Full paper budget is opt-in via pretrain."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .data import group_split, load_dataset, sample_indices
from .integration import RewardAdapter
from .model import EnergyConfig, EnergyMLP, FrozenReward, dsm_loss, save_checkpoint


def pretrain(args):
    if args.iterations < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        raise ValueError("iterations, batch size, and checkpoint frequency must be positive")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x, labels, groups, sources = load_dataset(args.data, args.representation, args.allow_unlabelled)
    train_idx, validation_idx = group_split(groups, args.seed, args.holdout)
    train_x, train_y = x[train_idx], labels[train_idx]
    # Check outcome coverage before creating output files.
    sample_indices(train_y, 1, rng)
    config = EnergyConfig(representation=args.representation, convention=args.convention,
                          conditioned=not args.unconditioned)
    model = EnergyMLP(config).to(args.device)
    tensor = torch.from_numpy(train_x).to(args.device)
    with torch.no_grad():
        model.center.copy_(tensor.mean(0))
        model.scale.copy_(tensor.std(0, unbiased=False).clamp_min(1e-3))
    normalized = model.normalize(tensor).detach()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999),
                                 weight_decay=0.0, amsgrad=False)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "synthetic_test_only" if args.synthetic else "reconstruction_training",
        "source_files": sources,
        "seed": args.seed,
        "train_rows": len(train_idx), "validation_rows": len(validation_idx),
        "train_groups": sorted(np.unique(groups[train_idx]).tolist()),
        "validation_groups": sorted(np.unique(groups[validation_idx]).tolist()),
        "success_probability": 0.7 if np.all(train_y >= 0) else None,
        "batch_size": args.batch_size,
        "learning_rate": 1e-4,
        "noise_conditioning": "additive_embedding_first_hidden" if config.conditioned else "none",
        "normalization": "training_mean_std_with_1e-3_floor",
        "reward_mapping_default": "exp(-softplus(density_energy-reference)/beta)",
        "reward_beta_default": 1.0,
        "reference_policy": "training_median_at_smallest_sigma",
        "representation_warning": "legacy56 is not the paper's 4D method" if args.representation == "legacy56" else None,
    }

    def checkpoint(step, loss, filename):
        with torch.no_grad():
            # A fixed training-only reference; never use Map B to calibrate energy.
            calibration = normalized[:min(len(normalized), 4096)]
            level = int(np.argmin(config.sigmas))
            index = torch.full((len(calibration),), level, dtype=torch.long, device=args.device)
            reference = float(model.density_energy(calibration, index).median().item())
        save_checkpoint(out / filename, model, optimizer, **metadata,
                        iterations=step, final_training_loss=loss, energy_reference=reference)

    with (out / "loss.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "dsm_training_loss"])
        model.train()
        for step in range(1, args.iterations + 1):
            batch = sample_indices(train_y, args.batch_size, rng)
            loss = dsm_loss(model, normalized[torch.as_tensor(batch, device=args.device)])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite DSM loss at iteration {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            writer.writerow([step, loss_value])
            if step == 1 or step % 1000 == 0 or step == args.iterations:
                print(json.dumps({"iteration": step, "dsm_loss": loss_value}), flush=True)
            if step % args.checkpoint_every == 0:
                checkpoint(step, loss_value, f"iteration_{step:06d}.pt")
    checkpoint(args.iterations, loss_value, "checkpoint.pt")
    validation_loss = None
    if len(validation_idx):
        held = torch.from_numpy(x[validation_idx[:1024]]).to(args.device)
        validation_loss = float(dsm_loss(model, model.normalize(held).detach()).detach().cpu())
    summary = {**metadata, "iterations": args.iterations,
               "final_training_loss": loss_value,
               "validation_dsm_loss_single_noise_sample": validation_loss,
               "checkpoint": str((out / "checkpoint.pt").resolve()),
               "navigation_evaluation": "not_run"}
    (out / "run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"checkpoint": summary["checkpoint"], "navigation_evaluation": "not_run"}))
    return summary


def smoke(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty smoke output directory")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    # Deliberately synthetic. No claim of expert quality, smoothness, or navigation.
    n = 192
    features = np.column_stack((rng.uniform(0, 0.3, n), rng.uniform(-1, 1, n),
                                rng.uniform(-np.pi, np.pi, n), rng.uniform(0.13, 3.5, n))).astype(np.float32)
    episode = np.repeat(np.arange(12), 16)
    success = (episode % 3 != 0).astype(np.int64)
    source = out / "SYNTHETIC_INPUT.npz"
    np.savez(source, features=features, success=success, episode_id=episode, map_id=np.full(n, "A"))
    train_args = argparse.Namespace(data=str(source), out=str(out / "training"), representation="paper4",
                                    allow_unlabelled=False, iterations=args.iterations, batch_size=64,
                                    checkpoint_every=5000, threads=2, seed=0, holdout=0.2,
                                    convention="negative_log_density", unconditioned=False,
                                    device="cpu", synthetic=True)
    pretrain(train_args)
    frozen = FrozenReward(out / "training/checkpoint.pt")
    state = np.zeros(28, dtype=np.float32)
    state[24], state[26] = 0.2, 0.5
    before = {k: v.clone() for k, v in frozen.model.state_dict().items()}
    rewards = frozen(features)
    assert np.isfinite(rewards).all() and (rewards > 0).all() and (rewards <= 1).all()
    combined, shaping = RewardAdapter(frozen).transition(state, [0.15, 0.1], state, 2.0)
    assert all(not p.requires_grad for p in frozen.model.parameters())
    assert all(torch.equal(v, frozen.model.state_dict()[k]) for k, v in before.items())
    verification = {"status": "synthetic_test_only", "finite_rewards": True,
                    "frozen_parameters_unchanged": True, "rows": n,
                    "reward_min": float(rewards.min()), "reward_max": float(rewards.max()),
                    "example_combined_reward": combined, "example_shaping_reward": shaping,
                    "navigation_evaluation": "not_run"}
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("pretrain", help="Offline DSM; default budget 200001 iterations")
    train.add_argument("--data", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--representation", choices=["paper4", "legacy56"], default="paper4")
    train.add_argument("--convention", choices=["negative_log_density", "paper_literal"], default="negative_log_density")
    train.add_argument("--iterations", type=int, default=200001)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--checkpoint-every", type=int, default=5000)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--holdout", type=float, default=0.2)
    train.add_argument("--device", default="cpu")
    train.add_argument("--allow-unlabelled", action="store_true")
    train.add_argument("--unconditioned", action="store_true")
    train.set_defaults(func=pretrain, synthetic=False)
    check = sub.add_parser("smoke", help="Synthetic integration check, no navigation training")
    check.add_argument("--out", required=True)
    check.add_argument("--iterations", type=int, default=8)
    check.set_defaults(func=smoke)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
