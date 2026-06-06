import { useCallback, useEffect, useState } from "react";
import {
  Cable,
  Hand,
  Play,
  Square,
  Unplug,
  Cpu,
  AlertTriangle,
} from "lucide-react";
import type { Status } from "./types";
import { getStatus, getLog, post } from "./api";
import {
  Btn,
  Card,
  CameraTile,
  JointTable,
  LogPanel,
  StatusPills,
} from "./components";

export default function App() {
  const [s, setS] = useState<Status | null>(null);
  const [task, setTask] = useState("Pick up the cube");
  const [taskEdited, setTaskEdited] = useState(false);
  const [srvLog, setSrvLog] = useState("…");
  const [cliLog, setCliLog] = useState("…");

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const st = await getStatus();
        if (!alive) return;
        setS(st);
        if (!taskEdited) setTask(st.task);
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
        "Run inference frees the arm to the remote client and will MOVE it. Continue?",
      )
    )
      return;
    void act("/inference/start", { task });
  }, [act, task]);

  const con = s?.connected ?? false;
  const tel = s?.teleop_running ?? false;
  const inf = s?.inference_running ?? false;

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

      {s?.error && (
        <div className="mb-4 flex items-center gap-2 rounded-xl border border-rose-500/40 bg-rose-600/15 px-4 py-2.5 text-sm text-rose-200">
          <AlertTriangle size={16} /> {s.error}
        </div>
      )}

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
              disabled={con || inf}
              icon={<Cable size={16} />}
            >
              Connect
            </Btn>
            <Btn
              onClick={() => act("/disconnect")}
              disabled={!con || inf}
              danger
              icon={<Unplug size={16} />}
            >
              Disconnect
            </Btn>
            <Btn
              onClick={() => act("/teleop/start")}
              disabled={!con || tel || inf}
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
        <label className="text-xs text-slate-400">task prompt</label>
        <div className="mt-1 flex flex-wrap gap-2">
          <input
            value={task}
            onChange={(e) => {
              setTask(e.target.value);
              setTaskEdited(true);
            }}
            disabled={inf}
            className="min-w-72 flex-1 rounded-xl border border-white/10 bg-black/30 px-3 py-2 text-sm outline-none focus:border-indigo-400/60 disabled:opacity-50"
          />
          <Btn onClick={runInference} disabled={inf} icon={<Play size={16} />}>
            Run inference
          </Btn>
          <Btn
            onClick={() => act("/inference/stop")}
            disabled={!inf}
            danger
            icon={<Square size={16} />}
          >
            Stop
          </Btn>
        </div>
      </Card>

      <div className="mb-4 grid gap-4 sm:grid-cols-2">
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
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <LogPanel title="GPU-box server log" text={srvLog} />
        <LogPanel title="Client log" text={cliLog} />
      </div>
    </div>
  );
}
