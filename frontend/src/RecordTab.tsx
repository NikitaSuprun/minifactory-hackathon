import { useEffect, useState } from "react";
import { CircleDot, Square, SkipForward, RotateCcw } from "lucide-react";
import type { Status } from "./types";
import { getLog } from "./api";
import { Btn, Card, CameraTile, LogPanel } from "./components";

const ACTIVE = new Set(["starting", "recording", "resetting", "finalizing", "pushing"]);

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-slate-400">{label}</span>
      {children}
    </label>
  );
}

const inputCls =
  "rounded-xl border border-white/10 bg-black/30 px-3 py-2 text-sm outline-none focus:border-indigo-400/60 disabled:opacity-50";

export default function RecordTab({
  s,
  act,
}: {
  s: Status | null;
  act: (path: string, body?: unknown) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [task, setTask] = useState("Pick up the cube");
  const [episodes, setEpisodes] = useState(5);
  const [episodeTime, setEpisodeTime] = useState(60);
  const [resetTime, setResetTime] = useState(15);
  const [fps, setFps] = useState(30);
  const [recLog, setRecLog] = useState("…");
  const [tick, setTick] = useState(0);

  const running = s?.recording_running ?? false;
  const phase = s?.record_status ?? "idle";
  const prog = s?.record_progress ?? {};

  // Poll the recorder log while a run is active (and briefly after).
  useEffect(() => {
    const load = async () => setRecLog(await getLog("record"));
    load();
    const id = setInterval(load, 2000);
    return () => clearInterval(id);
  }, []);

  // Local clock so the in-episode progress bar advances smoothly between status polls.
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 500);
    return () => clearInterval(id);
  }, []);

  const start = () => {
    if (!name.trim()) {
      alert("Dataset name is required.");
      return;
    }
    if (
      !confirm(
        "Recording will disconnect the arms from the dashboard and take over all " +
          "3 cameras. You teleoperate with the leader; episodes auto-advance. Continue?",
      )
    )
      return;
    void act("/record/start", {
      name: name.trim(),
      task,
      episodes,
      episode_time: episodeTime,
      reset_time: resetTime,
      fps,
    });
  };

  // In-episode progress (recording/resetting phases stamp episode_started_at).
  let episodePct = 0;
  if (
    (phase === "recording" || phase === "resetting") &&
    prog.episode_started_at &&
    prog.episode_time_s
  ) {
    const elapsed = Date.now() / 1000 - prog.episode_started_at;
    episodePct = Math.max(0, Math.min(100, (elapsed / prog.episode_time_s) * 100));
  }
  // Reference tick so the smooth-clock interval is not flagged as unused.
  void tick;

  const curEp = prog.current_episode ?? 0;
  const totEp = prog.total_episodes ?? episodes;
  const overallPct = totEp ? Math.min(100, (Math.max(curEp - 1, 0) / totEp) * 100) : 0;
  const url = phase === "done" ? prog.message : undefined;

  return (
    <div className="grid gap-4">
      <Card title="Record a dataset">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="dataset name">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={running}
              placeholder="e.g. so101-pick-cube"
              className={inputCls}
            />
          </Field>
          <Field label="task prompt">
            <input
              value={task}
              onChange={(e) => setTask(e.target.value)}
              disabled={running}
              className={inputCls}
            />
          </Field>
          <Field label="episodes">
            <input
              type="number"
              min={1}
              value={episodes}
              onChange={(e) => setEpisodes(+e.target.value)}
              disabled={running}
              className={inputCls}
            />
          </Field>
          <Field label="episode time (s)">
            <input
              type="number"
              min={1}
              value={episodeTime}
              onChange={(e) => setEpisodeTime(+e.target.value)}
              disabled={running}
              className={inputCls}
            />
          </Field>
          <Field label="reset time (s)">
            <input
              type="number"
              min={0}
              value={resetTime}
              onChange={(e) => setResetTime(+e.target.value)}
              disabled={running}
              className={inputCls}
            />
          </Field>
          <Field label="fps">
            <input
              type="number"
              min={1}
              value={fps}
              onChange={(e) => setFps(+e.target.value)}
              disabled={running}
              className={inputCls}
            />
          </Field>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          <Btn
            onClick={start}
            disabled={running || (s?.inference_running ?? false)}
            icon={<CircleDot size={16} />}
          >
            Start recording
          </Btn>
          <Btn
            onClick={() => act("/record/stop")}
            disabled={!running}
            danger
            icon={<Square size={16} />}
          >
            Stop &amp; save
          </Btn>
          <Btn
            onClick={() => act("/record/event", { event: "end_episode" })}
            disabled={!running}
            icon={<SkipForward size={16} />}
          >
            End episode
          </Btn>
          <Btn
            onClick={() => act("/record/event", { event: "rerecord" })}
            disabled={!running}
            icon={<RotateCcw size={16} />}
          >
            Re-record
          </Btn>
        </div>

        {phase !== "idle" && (
          <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="font-mono uppercase tracking-wider">{phase}</span>
              <span className="font-mono">
                episode {curEp || "–"} / {totEp}
              </span>
            </div>
            <div className="h-2 rounded-full bg-slate-700/60">
              <div
                className="h-2 rounded-full bg-indigo-400/80 transition-[width] duration-300"
                style={{ width: `${overallPct}%` }}
              />
            </div>
            {(phase === "recording" || phase === "resetting") && (
              <div className="h-1.5 rounded-full bg-slate-700/40">
                <div
                  className={`h-1.5 rounded-full ${phase === "recording" ? "bg-emerald-400/80" : "bg-amber-400/70"}`}
                  style={{ width: `${episodePct}%` }}
                />
              </div>
            )}
            {url && (
              <p className="text-sm text-emerald-300">
                ✓ pushed to{" "}
                <a className="underline" href={url} target="_blank" rel="noreferrer">
                  {prog.repo_id}
                </a>
              </p>
            )}
          </div>
        )}
      </Card>

      <div className="grid gap-4 sm:grid-cols-3">
        <CameraTile title="Phone cam" src="/camera.mjpeg" fps={s?.camera_fps.phone ?? 0} />
        <CameraTile title="Wrist cam" src="/wrist.mjpeg" fps={s?.camera_fps.wrist ?? 0} />
        <CameraTile title="Camera 3" src="/camera3.mjpeg" fps={s?.camera_fps.camera3 ?? 0} />
      </div>
      {ACTIVE.has(phase) && (
        <p className="-mt-2 text-xs text-slate-500">
          Cameras are held by the recorder — previews resume when recording ends.
        </p>
      )}

      <LogPanel title="Recorder log" text={recLog} />
    </div>
  );
}
