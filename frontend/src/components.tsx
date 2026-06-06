import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import type { Status } from "./types";

type Tone = "on" | "off" | "warn" | "err";

const TONE: Record<Tone, string> = {
  on: "bg-emerald-600/20 text-emerald-300 ring-emerald-500/40",
  off: "bg-slate-600/20 text-slate-300 ring-slate-500/30",
  warn: "bg-amber-600/20 text-amber-300 ring-amber-500/40",
  err: "bg-rose-600/20 text-rose-300 ring-rose-500/40",
};

export function Pill({
  tone,
  pulse,
  children,
}: {
  tone: Tone;
  pulse?: boolean;
  children: ReactNode;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm ring-1 ${TONE[tone]}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full bg-current ${pulse ? "animate-pulse" : ""}`}
      />
      {children}
    </span>
  );
}

export function Card({
  title,
  right,
  children,
  className = "",
}: {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-2xl border border-white/5 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur ${className}`}
    >
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold tracking-widest text-slate-400 uppercase">
          {title}
        </h3>
        {right}
      </div>
      {children}
    </div>
  );
}

export function Btn({
  onClick,
  disabled,
  danger,
  icon,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  danger?: boolean;
  icon?: ReactNode;
  children: ReactNode;
}) {
  const base = danger
    ? "bg-rose-600 hover:bg-rose-500"
    : "bg-indigo-600 hover:bg-indigo-500";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-medium text-white transition active:scale-95 disabled:cursor-not-allowed disabled:opacity-30 ${base}`}
    >
      {icon}
      {children}
    </button>
  );
}

export function JointTable({ joints }: { joints: Record<string, number> }) {
  const keys = Object.keys(joints);
  if (keys.length === 0)
    return <p className="text-sm text-slate-500">No joint data (start teleop).</p>;
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 sm:grid-cols-3">
      {keys.map((k) => {
        // SO-101 normalized positions are roughly [-100, 100] (gripper 0..100).
        const v = joints[k];
        const pct = Math.max(0, Math.min(100, (v + 100) / 2));
        return (
          <div key={k}>
            <div className="flex justify-between text-xs text-slate-400">
              <span>{k.replace(".pos", "")}</span>
              <span className="font-mono text-slate-200 tabular-nums">
                {v.toFixed(1)}
              </span>
            </div>
            <div className="mt-1 h-1.5 rounded-full bg-slate-700/60">
              <div
                className="h-1.5 rounded-full bg-indigo-400/80 transition-[width] duration-200"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function CameraTile({
  title,
  src,
  fps,
}: {
  title: string;
  src: string;
  fps: number;
}) {
  return (
    <Card
      title={title}
      right={
        fps > 0 ? (
          <span className="rounded-md bg-black/40 px-2 py-0.5 font-mono text-xs text-emerald-300">
            {fps.toFixed(0)} fps
          </span>
        ) : (
          <span className="font-mono text-xs text-slate-500">—</span>
        )
      }
    >
      <img
        src={src}
        alt={title}
        className="aspect-[4/3] w-full rounded-lg bg-black object-cover"
        onError={(e) => (e.currentTarget.style.opacity = "0.25")}
      />
    </Card>
  );
}

export function LogPanel({ title, text }: { title: string; text: string }) {
  const ref = useRef<HTMLPreElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [text]);
  return (
    <Card title={title} className="flex-1">
      <pre
        ref={ref}
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        }}
        className="scroll-thin max-h-64 overflow-auto rounded-lg bg-black/40 p-3 text-xs leading-relaxed whitespace-pre-wrap text-slate-300"
      >
        {text}
      </pre>
    </Card>
  );
}

export function StatusPills({ s }: { s: Status }) {
  return (
    <div className="flex flex-wrap gap-2">
      <Pill tone={s.connected ? "on" : "off"}>
        arms {s.connected ? "connected" : "disconnected"}
      </Pill>
      <Pill tone={s.teleop_running ? "on" : "off"} pulse={s.teleop_running}>
        teleop {s.teleop_running ? `on · ${s.control_fps} Hz` : "off"}
      </Pill>
      <Pill
        tone={
          s.inference_status === "running"
            ? "on"
            : s.inference_status === "error"
              ? "err"
              : "off"
        }
        pulse={s.inference_running}
      >
        inference {s.inference_status}
      </Pill>
      {s.record_status !== "idle" && (
        <Pill
          tone={
            s.record_status === "error"
              ? "err"
              : s.record_status === "done"
                ? "on"
                : s.recording_running
                  ? "warn"
                  : "off"
          }
          pulse={s.recording_running}
        >
          recording {s.record_status}
        </Pill>
      )}
      <Pill tone={s.server_reachable ? "on" : "err"}>
        server {s.server_reachable ? "reachable" : "down"}
      </Pill>
    </div>
  );
}
