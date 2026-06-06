import { useCallback, useEffect, useState } from "react";
import { RefreshCw, BadgeCheck } from "lucide-react";
import type { DatasetInfo, VerifyResult } from "./types";
import { getDatasets, verifyDataset } from "./api";
import { Btn, Card, Pill } from "./components";

export default function DatasetsTab({ preselect }: { preselect?: string | null }) {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [repo, setRepo] = useState<string>("");
  const [episode, setEpisode] = useState(0);
  const [verify, setVerify] = useState<VerifyResult | null>(null);
  const [verifying, setVerifying] = useState(false);

  const load = useCallback(async () => {
    try {
      const ds = await getDatasets();
      setDatasets(ds);
      setRepo((cur) => {
        if (cur && ds.some((d) => d.repo_id === cur)) return cur;
        const wanted =
          preselect && ds.some((d) => d.repo_id === preselect) ? preselect : null;
        return wanted ?? ds[0]?.repo_id ?? "";
      });
    } catch {
      /* keep last */
    }
  }, [preselect]);

  useEffect(() => {
    void load();
  }, [load]);

  // Reset per-dataset state when the selection changes.
  useEffect(() => {
    setEpisode(0);
    setVerify(null);
  }, [repo]);

  const selected = datasets.find((d) => d.repo_id === repo);
  const totalEpisodes = selected?.total_episodes ?? 0;

  const runVerify = async () => {
    if (!repo) return;
    setVerifying(true);
    try {
      setVerify(await verifyDataset(repo));
    } catch (e) {
      alert(String(e instanceof Error ? e.message : e));
    } finally {
      setVerifying(false);
    }
  };

  const selCls =
    "rounded-xl border border-white/10 bg-black/30 px-3 py-2 text-sm outline-none focus:border-indigo-400/60";

  return (
    <div className="grid gap-4">
      <Card
        title="Datasets"
        right={
          <button
            onClick={() => void load()}
            className="inline-flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200"
          >
            <RefreshCw size={13} /> refresh
          </button>
        }
      >
        {datasets.length === 0 ? (
          <p className="text-sm text-slate-500">
            No local datasets found under the LeRobot cache. Record one in the Record
            tab.
          </p>
        ) : (
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-400">dataset</span>
              <select
                value={repo}
                onChange={(e) => setRepo(e.target.value)}
                className={selCls}
              >
                {datasets.map((d) => (
                  <option key={d.repo_id} value={d.repo_id}>
                    {d.repo_id} ({d.total_episodes} ep)
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-400">episode</span>
              <select
                value={episode}
                onChange={(e) => setEpisode(+e.target.value)}
                className={selCls}
                disabled={totalEpisodes === 0}
              >
                {Array.from({ length: Math.max(totalEpisodes, 1) }, (_, i) => (
                  <option key={i} value={i}>
                    {i}
                  </option>
                ))}
              </select>
            </label>
            <Btn
              onClick={runVerify}
              disabled={!repo || verifying}
              icon={<BadgeCheck size={16} />}
            >
              {verifying ? "Verifying…" : "Verify upload"}
            </Btn>
          </div>
        )}

        {selected && (
          <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-400">
            <span className="rounded-md bg-slate-700/40 px-2 py-1 font-mono">
              {selected.total_episodes} episodes
            </span>
            <span className="rounded-md bg-slate-700/40 px-2 py-1 font-mono">
              {selected.total_frames} frames
            </span>
            <span className="rounded-md bg-slate-700/40 px-2 py-1 font-mono">
              {selected.fps} fps
            </span>
            <span className="rounded-md bg-slate-700/40 px-2 py-1 font-mono">
              {selected.cameras.join(" · ") || "no cameras"}
            </span>
          </div>
        )}

        {verify && (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Pill tone={verify.match ? "on" : "err"}>
              {verify.match ? "upload verified" : "mismatch"}
            </Pill>
            <span className="font-mono text-xs text-slate-400">
              local: {verify.local.total_episodes} ep · {verify.local.video_files}{" "}
              videos
            </span>
            <span className="font-mono text-xs text-slate-400">
              hub:{" "}
              {verify.hub.exists
                ? `${verify.hub.video_files} videos${verify.hub.has_info ? " · info ✓" : ""}`
                : `unavailable (${verify.hub.error ?? "not found"})`}
            </span>
          </div>
        )}
      </Card>

      {repo && totalEpisodes > 0 && (
        <Card title={`Replay · ${repo} · episode ${episode}`}>
          <iframe
            key={`${repo}#${episode}`}
            src={`/datasets/${repo}/viewer?episode=${episode}`}
            className="h-[78vh] w-full rounded-lg border border-white/5 bg-black"
            title="dataset viewer"
          />
        </Card>
      )}
    </div>
  );
}
