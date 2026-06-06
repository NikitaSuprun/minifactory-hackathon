#!/usr/bin/env python
"""Build a self-contained HTML viewer for a local LeRobotDataset episode.

Plays all camera streams in sync and plots action / observation.state signals
with a playhead tied to video playback. No rerun / torch / GPU needed — just a
browser. AV1-in-mp4 videos play best in Chrome (Safari 17+ also works).

Usage:
    uv run python make_viewer.py --repo-id nsuprun/so101-pick-cube --episode-index 0
    # then open the printed viewer.html path in a browser
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_ROOT = Path(os.path.expanduser("~/.cache/huggingface/lerobot"))


def build_viewer_html(
    repo_id: str,
    episode_index: int = 0,
    root: Path = DEFAULT_ROOT,
    video_url_prefix: str | None = None,
) -> str:
    """Return a self-contained HTML viewer for one episode of a local LeRobotDataset.

    ``video_url_prefix`` makes the ``<video>`` ``src`` absolute (e.g. when the dashboard
    serves the mp4s from a route); left ``None`` the paths stay relative to the dataset
    dir, which is what the standalone ``viewer.html`` next to the data needs.
    """
    ds_dir = root / repo_id
    info = json.loads((ds_dir / "meta" / "info.json").read_text())
    fps = info["fps"]

    # Camera (video) feature keys, e.g. observation.images.camera1
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    # Load the episode's rows from the single data parquet (this dataset has one chunk).
    data_file = ds_dir / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    df = table.to_pylist()
    rows = [r for r in df if r["episode_index"] == episode_index]
    rows.sort(key=lambda r: r["frame_index"])
    if not rows:
        raise ValueError(f"No frames found for episode {episode_index}")

    action_names = info["features"]["action"]["names"]
    state_names = info["features"]["observation.state"]["names"]
    timestamps = [r["timestamp"] for r in rows]
    actions = [list(r["action"]) for r in rows]
    states = [list(r["observation.state"]) for r in rows]

    # Resolve video file paths relative to where the HTML will live (ds_dir).
    videos = []
    for vk in video_keys:
        rel = info["video_path"].format(video_key=vk, chunk_index=0, file_index=0)
        if (ds_dir / rel).exists():
            src = rel if video_url_prefix is None else f"{video_url_prefix}/{rel}"
            videos.append({"key": vk, "src": src})

    payload = {
        "repo_id": repo_id,
        "episode": episode_index,
        "fps": fps,
        "num_frames": len(rows),
        "duration": timestamps[-1] if timestamps else 0,
        "videos": videos,
        "action_names": action_names,
        "state_names": state_names,
        "timestamps": timestamps,
        "actions": actions,
        "states": states,
    }

    return _HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="LeRobot cache root (default ~/.cache/huggingface/lerobot)",
    )
    args = ap.parse_args()

    root = args.root or DEFAULT_ROOT
    try:
        html = build_viewer_html(args.repo_id, args.episode_index, root)
    except ValueError as e:
        raise SystemExit(str(e))
    out = root / args.repo_id / "viewer.html"
    out.write_text(html)
    print(f"Viewer written -> {out}")
    print(f"Open it with:   open '{out}'")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LeRobot dataset viewer</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background:#111; color:#eee; }
  header { padding: 12px 18px; background:#1b1b1b; border-bottom:1px solid #333; }
  header h1 { font-size: 15px; margin: 0 0 4px; font-weight: 600; }
  header .meta { font-size: 12px; color:#9aa; }
  .videos { display:flex; flex-wrap:wrap; gap:10px; padding:14px 18px; }
  .vid { background:#000; border:1px solid #333; border-radius:6px; overflow:hidden; }
  .vid .cap { font-size:11px; color:#9aa; padding:4px 8px; background:#1b1b1b; }
  video { display:block; width:340px; height:255px; background:#000; }
  .controls { padding: 6px 18px 14px; display:flex; align-items:center; gap:12px; }
  .controls button { background:#2a2a2a; color:#eee; border:1px solid #444; border-radius:5px; padding:6px 12px; cursor:pointer; }
  #scrub { flex:1; }
  #tlabel { font-variant-numeric: tabular-nums; font-size:12px; color:#9aa; min-width:150px; text-align:right;}
  .charts { padding: 0 18px 24px; }
  .chartblock h2 { font-size:13px; margin:14px 0 6px; color:#cdd; }
  canvas.plot { width:100%; height:220px; background:#161616; border:1px solid #333; border-radius:6px; }
  .legend { font-size:11px; color:#9aa; margin-top:4px; display:flex; flex-wrap:wrap; gap:10px;}
  .legend span { display:inline-flex; align-items:center; gap:4px;}
  .legend i { width:10px; height:10px; border-radius:2px; display:inline-block; }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="meta" id="metaline"></div>
</header>

<div class="videos" id="videos"></div>

<div class="controls">
  <button id="play">▶ Play</button>
  <input type="range" id="scrub" min="0" max="1000" value="0">
  <span id="tlabel"></span>
</div>

<div class="charts" id="charts"></div>

<script>
const DATA = __DATA__;
const COLORS = ["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#46f0f0","#f032e6"];

document.getElementById("title").textContent =
  `${DATA.repo_id} — episode ${DATA.episode}`;
document.getElementById("metaline").textContent =
  `${DATA.num_frames} frames · ${DATA.fps} fps · ${DATA.duration.toFixed(2)}s · ${DATA.videos.length} cameras`;

// ---- videos ----
const vidWrap = document.getElementById("videos");
const videoEls = [];
DATA.videos.forEach(v => {
  const box = document.createElement("div"); box.className = "vid";
  const cap = document.createElement("div"); cap.className = "cap";
  cap.textContent = v.key.replace("observation.images.", "");
  const el = document.createElement("video");
  el.src = v.src; el.muted = true; el.playsInline = true; el.preload = "auto";
  box.appendChild(cap); box.appendChild(el); vidWrap.appendChild(box);
  videoEls.push(el);
});
const master = videoEls[0];   // drive everything off the first camera's clock

// ---- charts ----
function makeChart(title, names, series) {
  const block = document.createElement("div"); block.className = "chartblock";
  const h = document.createElement("h2"); h.textContent = title; block.appendChild(h);
  const cv = document.createElement("canvas"); cv.className = "plot"; block.appendChild(cv);
  const leg = document.createElement("div"); leg.className = "legend";
  names.forEach((n,i)=>{ const s=document.createElement("span");
    s.innerHTML = `<i style="background:${COLORS[i%COLORS.length]}"></i>${n}`; leg.appendChild(s); });
  block.appendChild(leg);
  document.getElementById("charts").appendChild(block);

  // precompute min/max across all dims
  let mn=Infinity, mx=-Infinity;
  series.forEach(row=>row.forEach(val=>{ if(val<mn)mn=val; if(val>mx)mx=val; }));
  if(mn===mx){mn-=1;mx+=1;}
  const T = DATA.timestamps, tMax = T[T.length-1] || 1;

  function draw(playT){
    const dpr = window.devicePixelRatio||1;
    const w = cv.clientWidth, h = cv.clientHeight;
    cv.width = w*dpr; cv.height = h*dpr;
    const ctx = cv.getContext("2d"); ctx.scale(dpr,dpr);
    ctx.clearRect(0,0,w,h);
    const pad=6;
    const X = t => pad + (t/tMax)*(w-2*pad);
    const Y = v => (h-pad) - ((v-mn)/(mx-mn))*(h-2*pad);
    // lines
    const dims = names.length;
    for(let d=0; d<dims; d++){
      ctx.beginPath(); ctx.strokeStyle = COLORS[d%COLORS.length]; ctx.lineWidth=1;
      for(let i=0;i<series.length;i++){
        const x=X(T[i]), y=Y(series[i][d]);
        i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
      }
      ctx.stroke();
    }
    // playhead
    const px = X(playT);
    ctx.strokeStyle="#fff"; ctx.lineWidth=1; ctx.beginPath();
    ctx.moveTo(px,0); ctx.lineTo(px,h); ctx.stroke();
  }
  return draw;
}

const drawAction = makeChart("action", DATA.action_names, DATA.actions);
const drawState  = makeChart("observation.state", DATA.state_names, DATA.states);

// ---- sync loop ----
const scrub = document.getElementById("scrub");
const tlabel = document.getElementById("tlabel");
const playBtn = document.getElementById("play");
const dur = DATA.duration || (master.duration||1);

function syncOthers(){ const t=master.currentTime;
  videoEls.forEach((v,i)=>{ if(i>0 && Math.abs(v.currentTime-t)>0.08) v.currentTime=t; }); }

function tick(){
  const t = master.currentTime;
  scrub.value = String(Math.round((t/dur)*1000));
  tlabel.textContent = `${t.toFixed(2)}s / ${dur.toFixed(2)}s`;
  drawAction(t); drawState(t);
  syncOthers();
  requestAnimationFrame(tick);
}
playBtn.onclick = ()=>{
  if(master.paused){ videoEls.forEach(v=>v.play()); playBtn.textContent="⏸ Pause"; }
  else { videoEls.forEach(v=>v.pause()); playBtn.textContent="▶ Play"; }
};
scrub.oninput = ()=>{ const t=(scrub.value/1000)*dur;
  videoEls.forEach(v=>{ try{v.currentTime=t;}catch(e){} }); drawAction(t); drawState(t); };
master.addEventListener("loadedmetadata", ()=>{ drawAction(0); drawState(0); });
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
