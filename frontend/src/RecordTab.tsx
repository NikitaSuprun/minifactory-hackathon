import { useEffect, useState } from "react";
import {
  Cable,
  Hand,
  CircleDot,
  Square,
  SkipForward,
  RotateCcw,
} from "lucide-react";
import type { Status } from "./types";
import { Btn, Card, CameraTile } from "./components";

const ACTIVE = new Set(["starting", "recording", "resetting", "finalizing", "pushing"]);

type Params = {
  name: string;
  task: string;
  episodes: number;
  episode_time: number;
  reset_time: number;
  fps: number;
};

const inputCls =
  "rounded-xl border border-white/10 bg-black/30 px-3 py-2 text-sm outline-none focus:border-indigo-400/60";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-slate-400">{label}</span>
      {children}
    </label>
  );
}

function fmt(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export default function RecordTab({
  s,
  act,
}: {
  s: Status | null;
  act: (path: string, body?: unknown) => Promise<void>;
}) {
  const [params, setParams] = useState<Params>({
    name: "",
    task: "Pick up the cube",
    episodes: 5,
    episode_time: 60,
    reset_time: 15,
    fps: 30,
  });
  const [modalOpen, setModalOpen] = useState(false);
  const [countdown, setCountdown] = useState<number | "go" | null>(null);
  const [nowMs, setNowMs] = useState(() => Date.now());

  // Ticking clock so the elapsed timer + phase bar advance between status polls.
  useEffect(() => {
    const id = setInterval(() => setNowMs(Date.now()), 250);
    return () => clearInterval(id);
  }, []);

  const con = s?.connected ?? false;
  const tel = s?.teleop_running ?? false;
  const rec = s?.recording_running ?? false;
  const inf = s?.inference_running ?? false;
  const phase = s?.record_status ?? "idle";
  const counting = countdown !== null;
  const canRecord = con && tel && !rec && !inf && !counting;

  const set = (patch: Partial<Params>) => setParams((p) => ({ ...p, ...patch }));

  const beginCountdown = () => {
    if (!params.name.trim()) {
      alert("Dataset name is required.");
      return;
    }
    setModalOpen(false);
    let n = 3;
    setCountdown(3);
    const tick = () => {
      n -= 1;
      if (n > 0) {
        setCountdown(n);
        setTimeout(tick, 800);
      } else if (n === 0) {
        setCountdown("go");
        setTimeout(() => {
          setCountdown(null);
          void act("/record/start", { ...params, name: params.name.trim() });
        }, 700);
      }
    };
    setTimeout(tick, 800);
  };

  // Overall + in-phase progress.
  const curEp = s?.record_current_episode ?? 0;
  const totEp = s?.record_total_episodes || params.episodes;
  const overallPct = totEp ? Math.min(100, (Math.max(curEp - 1, 0) / totEp) * 100) : 0;
  const nowS = nowMs / 1000;
  let phasePct = 0;
  if (
    (phase === "recording" || phase === "resetting") &&
    s?.record_phase_started_at &&
    s.record_phase_time_s
  ) {
    phasePct = Math.min(
      100,
      ((nowS - s.record_phase_started_at) / s.record_phase_time_s) * 100,
    );
  }
  const elapsed = s?.record_started_at ? nowS - s.record_started_at : 0;
  const repo = s?.record_repo_id ?? s?.record_last_done_repo ?? null;

  return (
    <div className="grid gap-4">
      {/* Countdown overlay */}
      {counting && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="text-center">
            <div className="text-[8rem] leading-none font-bold tabular-nums text-white">
              {countdown === "go" ? "Go!" : countdown}
            </div>
            <p className="mt-2 text-slate-300">get ready — teleoperate the leader arm</p>
          </div>
        </div>
      )}

      {/* Questions modal */}
      {modalOpen && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4">
          <Card title="New recording" className="w-full max-w-lg">
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="dataset name">
                <input
                  autoFocus
                  value={params.name}
                  onChange={(e) => set({ name: e.target.value })}
                  placeholder="e.g. so101-pick-cube"
                  className={inputCls}
                />
              </Field>
              <Field label="task prompt">
                <input
                  value={params.task}
                  onChange={(e) => set({ task: e.target.value })}
                  className={inputCls}
                />
              </Field>
              <Field label="episodes">
                <input
                  type="number"
                  min={1}
                  value={params.episodes}
                  onChange={(e) => set({ episodes: +e.target.value })}
                  className={inputCls}
                />
              </Field>
              <Field label="fps">
                <input
                  type="number"
                  min={1}
                  value={params.fps}
                  onChange={(e) => set({ fps: +e.target.value })}
                  className={inputCls}
                />
              </Field>
              <Field label="episode time (s)">
                <input
                  type="number"
                  min={1}
                  value={params.episode_time}
                  onChange={(e) => set({ episode_time: +e.target.value })}
                  className={inputCls}
                />
              </Field>
              <Field label="reset time (s)">
                <input
                  type="number"
                  min={0}
                  value={params.reset_time}
                  onChange={(e) => set({ reset_time: +e.target.value })}
                  className={inputCls}
                />
              </Field>
            </div>
            <div className="mt-4 flex justify-end gap-2">
              <Btn onClick={() => setModalOpen(false)} danger>
                Cancel
              </Btn>
              <Btn onClick={beginCountdown} icon={<CircleDot size={16} />}>
                Start (3-2-1)
              </Btn>
            </div>
          </Card>
        </div>
      )}

      <Card title="Record a dataset">
        <ol className="mb-4 flex flex-wrap items-center gap-3 text-sm">
          <li className="flex items-center gap-2">
            <span
              className={`flex h-5 w-5 items-center justify-center rounded-full text-xs ${con ? "bg-emerald-600 text-white" : "bg-slate-700 text-slate-300"}`}
            >
              1
            </span>
            <Btn
              onClick={() => act("/connect")}
              disabled={con || inf || rec}
              icon={<Cable size={16} />}
            >
              Connect
            </Btn>
          </li>
          <li className="flex items-center gap-2">
            <span
              className={`flex h-5 w-5 items-center justify-center rounded-full text-xs ${tel ? "bg-emerald-600 text-white" : "bg-slate-700 text-slate-300"}`}
            >
              2
            </span>
            <Btn
              onClick={() => act("/teleop/start")}
              disabled={!con || tel || inf || rec}
              icon={<Hand size={16} />}
            >
              Start teleop
            </Btn>
          </li>
          <li className="flex items-center gap-2">
            <span
              className={`flex h-5 w-5 items-center justify-center rounded-full text-xs ${rec ? "bg-rose-600 text-white" : "bg-slate-700 text-slate-300"}`}
            >
              3
            </span>
            <Btn
              onClick={() => setModalOpen(true)}
              disabled={!canRecord}
              icon={<CircleDot size={16} />}
            >
              Record
            </Btn>
          </li>
        </ol>

        {!con && (
          <p className="text-xs text-slate-500">
            Connect the arms and start teleoperation, then Record opens the questions and
            counts you in.
          </p>
        )}

        {rec && (
          <div className="flex flex-wrap gap-2">
            <Btn
              onClick={() => act("/record/stop")}
              danger
              icon={<Square size={16} />}
            >
              Stop &amp; save
            </Btn>
            <Btn
              onClick={() => act("/record/event", { event: "end_episode" })}
              icon={<SkipForward size={16} />}
            >
              End episode
            </Btn>
            <Btn
              onClick={() => act("/record/event", { event: "rerecord" })}
              icon={<RotateCcw size={16} />}
            >
              Re-record
            </Btn>
          </div>
        )}

        {phase !== "idle" && (
          <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2 font-medium">
                {ACTIVE.has(phase) && (
                  <span className="h-3 w-3 animate-pulse rounded-full bg-rose-500 shadow-[0_0_8px_2px] shadow-rose-500/60" />
                )}
                <span className="font-mono tracking-wider uppercase">
                  {phase === "recording"
                    ? "● REC"
                    : phase === "resetting"
                      ? "reset window"
                      : phase}
                </span>
              </span>
              <span className="font-mono tabular-nums text-slate-300">
                {ACTIVE.has(phase) && <>elapsed {fmt(elapsed)} · </>}
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
                  className={`h-1.5 rounded-full ${phase === "recording" ? "bg-rose-400/80" : "bg-amber-400/70"}`}
                  style={{ width: `${phasePct}%` }}
                />
              </div>
            )}
            {phase === "done" && repo && (
              <p className="text-sm text-emerald-300">
                ✓ pushed to{" "}
                <a
                  className="underline"
                  href={`https://huggingface.co/datasets/${repo}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  {repo}
                </a>{" "}
                — verify it in the Datasets tab.
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
    </div>
  );
}
