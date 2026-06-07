import { useCallback, useEffect, useRef, useState } from "react";
import {
  Cable,
  Hand,
  Play,
  Pause,
  Square,
  Unplug,
  Cpu,
  Zap,
  Check,
  AlertTriangle,
  Gamepad2,
  CircleDot,
  Database,
} from "lucide-react";
import type { Status } from "./types";
import { getStatus, getLog, post } from "./api";
import {
  Btn,
  Card,
  CameraTile,
  JointTable,
  LogPanel,
  Pill,
  StatusPills,
} from "./components";
import RecordTab from "./RecordTab";
import DatasetsTab from "./DatasetsTab";

type Tab = "control" | "record" | "datasets";

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: "control", label: "Control", icon: <Gamepad2 size={15} /> },
  { id: "record", label: "Record", icon: <CircleDot size={15} /> },
  { id: "datasets", label: "Datasets", icon: <Database size={15} /> },
];

export default function App() {
  const [s, setS] = useState<Status | null>(null);
  const [tab, setTab] = useState<Tab>("control");
  const [task, setTask] = useState("Pick up the cube");
  const [taskEdited, setTaskEdited] = useState(false);
  const [srvLog, setSrvLog] = useState("…");
  const [cliLog, setCliLog] = useState("…");
  const prevRecord = useRef<string>("idle");

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const st = await getStatus();
        if (!alive) return;
        setS(st);
        if (!taskEdited) setTask(st.task);
        // When a recording finishes, jump to the Datasets tab to verify it.
        if (st.record_status === "done" && prevRecord.current !== "done") {
          setTab("datasets");
        }
        prevRecord.current = st.record_status;
      } catch {
        /* keep last */
      }
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [taskEdited]);

  useEffect(() => {
    const tick = async () => {
      setSrvLog(await getLog("server"));
      setCliLog(await getLog("client"));
    };
    tick();
    const id = setInterval(tick, 4000);
    return () => clearInterval(id);
  }, []);

  const act = useCallback(async (path: string, body?: unknown) => {
    try {
      await post(path, body);
    } catch (e) {
      alert(String(e instanceof Error ? e.message : e));
    }
  }, []);

  const runInference = useCallback(() => {
    if (
      !confirm(
        "This frees the arm to the remote client and will MOVE it. Continue?",
      )
    )
      return;
    void act("/inference/start", { task });
  }, [act, task]);

  const setLiveTask = useCallback(() => {
    void act("/inference/task", { task });
  }, [act, task]);

  const con = s?.connected ?? false;
  const tel = s?.teleop_running ?? false;
  const inf = s?.inference_running ?? false; // subprocess owns the hardware (loading/warm/running)
  const rec = s?.recording_running ?? false;
  const istatus = s?.inference_status ?? "idle";
  const running = istatus === "running"; // following the policy (arm moving)
  const ready = istatus === "ready"; // warm + holding (model loaded, arm still)
  const prewarming = istatus === "prewarming"; // loading
  const warm = running || ready; // subprocess alive with model loaded
  const serverTask = s?.task ?? "";

  return (
    <div className="mx-auto max-w-6xl px-5 py-6">
      <header className="mb-5 flex items-center justify-between">
        <h1 className="text-xl font-semibold tracking-tight">
          SO-101 <span className="text-slate-400">Control</span>
        </h1>
        <span className="font-mono text-xs text-slate-500">
          {s?.follower_port ?? "—"}
        </span>
      </header>

      <nav className="mb-5 flex gap-1 rounded-xl border border-white/5 bg-slate-900/40 p-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`inline-flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition ${
              tab === t.id
                ? "bg-indigo-600 text-white"
                : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
            }`}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </nav>

      {s?.error && (
        <div className="mb-4 flex items-center gap-2 rounded-xl border border-rose-500/40 bg-rose-600/15 px-4 py-2.5 text-sm text-rose-200">
          <AlertTriangle size={16} /> {s.error}
        </div>
      )}

      {tab === "control" && (
        <>
          <div className="mb-4 grid gap-4 lg:grid-cols-3">
            <Card title="Status" className="lg:col-span-2">
              {s ? (
                <StatusPills s={s} />
              ) : (
                <p className="text-sm text-slate-500">connecting…</p>
              )}
              <div className="mt-4">
                <JointTable joints={s?.joints ?? {}} />
              </div>
            </Card>

            <Card title="Control">
              <div className="flex flex-wrap gap-2">
                <Btn
                  onClick={() => act("/connect")}
                  disabled={con || inf || rec}
                  icon={<Cable size={16} />}
                >
                  Connect
                </Btn>
                <Btn
                  onClick={() => act("/disconnect")}
                  disabled={!con || inf || rec}
                  danger
                  icon={<Unplug size={16} />}
                >
                  Disconnect
                </Btn>
                <Btn
                  onClick={() => act("/teleop/start")}
                  disabled={!con || tel || inf || rec}
                  icon={<Hand size={16} />}
                >
                  Start teleop
                </Btn>
                <Btn
                  onClick={() => act("/teleop/stop")}
                  disabled={!tel}
                  danger
                  icon={<Square size={16} />}
                >
                  Stop teleop
                </Btn>
              </div>
            </Card>
          </div>

          <Card title="Inference · remote (GPU box)" className="mb-4">
            <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
              <span className="inline-flex items-center gap-1 rounded-md bg-slate-700/50 px-2 py-1 font-mono text-slate-300">
                <Cpu size={13} /> {s?.policy ?? "lerobot/smolvla_base"}
              </span>
              <span className="rounded-md bg-emerald-700/30 px-2 py-1 font-mono text-emerald-300">
                {s?.device ?? "cuda"}
              </span>
            </div>
            <label className="text-xs text-slate-400">
              task prompt{" "}
              {warm && (
                <span className="text-slate-500">
                  · changing it applies live, no reload
                </span>
              )}
            </label>
            <div className="mt-1 flex flex-wrap items-center gap-2">
              <input
                value={task}
                onChange={(e) => {
                  setTask(e.target.value);
                  setTaskEdited(true);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && warm && task !== serverTask)
                    setLiveTask();
                }}
                disabled={rec}
                className="min-w-72 flex-1 rounded-xl border border-white/10 bg-black/30 px-3 py-2 text-sm outline-none focus:border-indigo-400/60 disabled:opacity-50"
              />
              <Btn
                onClick={setLiveTask}
                disabled={!warm || task === serverTask}
                icon={<Check size={16} />}
              >
                Set task
              </Btn>
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Btn
                onClick={() => act("/inference/prewarm")}
                disabled={prewarming || warm || rec}
                icon={<Zap size={16} />}
              >
                Prewarm
              </Btn>
              <Btn
                onClick={runInference}
                disabled={running || rec}
                icon={<Play size={16} />}
              >
                {ready ? "Resume" : "Run inference"}
              </Btn>
              <Btn
                onClick={() => act("/inference/pause")}
                disabled={!running}
                icon={<Pause size={16} />}
              >
                Pause
              </Btn>
              <Btn
                onClick={() => act("/inference/stop")}
                disabled={!inf}
                danger
                icon={<Square size={16} />}
              >
                Stop
              </Btn>
              {prewarming && (
                <Pill tone="warn" pulse>
                  Loading…
                </Pill>
              )}
              {running && (
                <Pill tone="on" pulse>
                  Following
                </Pill>
              )}
              {ready && <Pill tone="warn">Paused · warm</Pill>}
            </div>
          </Card>

          <div className="mb-4 grid gap-4 sm:grid-cols-3">
            <CameraTile
              title="Phone cam"
              src="/camera.mjpeg"
              fps={s?.camera_fps.phone ?? 0}
            />
            <CameraTile
              title="Wrist cam"
              src="/wrist.mjpeg"
              fps={s?.camera_fps.wrist ?? 0}
            />
            <CameraTile
              title="Camera 3"
              src="/camera3.mjpeg"
              fps={s?.camera_fps.camera3 ?? 0}
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <LogPanel title="GPU-box server log" text={srvLog} />
            <LogPanel title="Client log" text={cliLog} />
          </div>
        </>
      )}

      {tab === "record" && <RecordTab s={s} act={act} />}

      {tab === "datasets" && (
        <DatasetsTab preselect={s?.record_last_done_repo} />
      )}
    </div>
  );
}
