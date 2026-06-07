"""Merge two or more local LeRobot datasets into one and push it to the Hub.

Usage:
    uv run python merge_datasets.py [out_name] [src_name_1 src_name_2 ...]

With no args, merges ``so101-pickup-s2`` + ``so101-s4`` into ``so101-pickup-s2-s4`` (all
under your HF user). Sources must share fps, robot_type, and features (same cameras) —
``aggregate_datasets`` validates this and raises otherwise. The merged dataset is built in
the local LeRobot cache, then pushed private to the Hub.
"""

import sys

import phone_camera  # noqa: F401 - import side effect loads .env + .env.local (HF_TOKEN)
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from recording import resolve_hf_username

user = resolve_hf_username()
args = sys.argv[1:]
out_name = args[0] if args else "so101-pickup-s2-s4"
src_names = args[1:] if len(args) > 1 else ["so101-pickup-s2", "so101-s4"]

out_repo = f"{user}/{out_name}"
src_repos = [f"{user}/{name}" for name in src_names]

print(f"Merging {src_repos} -> {out_repo}")
aggregate_datasets(repo_ids=src_repos, aggr_repo_id=out_repo)

merged = LeRobotDataset(out_repo)
print(
    f"Merged: episodes={merged.meta.total_episodes} frames={merged.meta.total_frames}"
)

print(f"Pushing to https://huggingface.co/datasets/{out_repo} (private)...")
merged.push_to_hub(private=True)
print("Done.")
