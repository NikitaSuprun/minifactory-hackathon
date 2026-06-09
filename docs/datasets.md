# Datasets — record, merge, and publish

Demonstrations are recorded as [LeRobot](https://github.com/huggingface/lerobot)
datasets (observations from all three cameras + the follower's joint actions), then
merged and pushed to the Hugging Face Hub for training.

📦 Our published dataset:
**[`nsuprun/so101-pickup-merged`](https://huggingface.co/datasets/nsuprun/so101-pickup-merged)**
→ trained into the ACT model
[`nsuprun/merged-so101-49904152`](https://huggingface.co/nsuprun/merged-so101-49904152)
(see **[inference.md](inference.md)**).

## Record demonstrations

You record by teleoperating: the **leader** arm drives the **follower**, and every
frame (3 camera images + follower joint targets) is saved as an episode. First get the
arms and cameras working — see **[arms.md](arms.md)** and **[cameras.md](cameras.md)**.

- **From the dashboard:** the **Record** tab creates a dataset, runs episodes while you
  teleoperate, and pushes to the Hub when you stop. Simplest path.
- **From the CLI** (`record_dataset.py`):
  ```bash
  uv run python record_dataset.py --name so101-pickup --task "Pick up the cube" --episodes 30
  ```
  Episodes are written to the local LeRobot cache
  (`~/.cache/huggingface/lerobot/<user>/<name>/`) and pushed to a **private** HF repo.

Recording uses streaming video encoding (`RECORD_VCODEC=auto` → `h264_videotoolbox`
on the Mac, or `libsvtav1`). Shared building blocks live in `recording.py`.

## Merge several datasets

We recorded in multiple sessions, then combined them. `merge_datasets.py` validates that
the sources share `fps` / `robot_type` / `features`, aggregates them, and pushes the
result:

```bash
uv run python merge_datasets.py <out_name> <src_1> <src_2> ...
```

## Harmonise video codecs

Aggregation requires matching codecs. If sessions were recorded with different encoders,
re-encode a dataset's videos in place first:

```bash
uv run python reencode_dataset_videos.py <dataset_name> [h264|av1]
```

## Recover an interrupted upload

If the dashboard captured a dataset but crashed before pushing, push it from the local
cache after the fact:

```bash
uv run python upload_cached_dataset.py <name>
```

## Inspect an episode

Build a standalone HTML viewer (frames + joint state) for one episode in the local
cache:

```bash
uv run python make_viewer.py <repo_id> [episode_index]
```
