"""One-off: push an already-recorded LeRobot dataset from local cache to the Hub.

Used to recover a recording whose process crashed before it could push (e.g. the
dashboard died mid-session). Usage: uv run python upload_cached_dataset.py <name>
"""

import sys
import phone_camera  # noqa: F401 - import side effect loads .env + .env.local (HF_TOKEN)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from recording import resolve_hf_username

name = sys.argv[1] if len(sys.argv) > 1 else "so101-s4"
repo_id = f"{resolve_hf_username()}/{name}"

print(f"Loading {repo_id} from local cache...")
dataset = LeRobotDataset(repo_id)  # root defaults to LEROBOT_HOME; loads from disk
print(f"  episodes={dataset.meta.total_episodes} frames={dataset.meta.total_frames}")

print(f"Pushing to https://huggingface.co/datasets/{repo_id} (private)...")
dataset.push_to_hub(private=True)
print("Done.")
