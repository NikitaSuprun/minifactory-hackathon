import type { Status } from "./types";

export async function getStatus(): Promise<Status> {
  const r = await fetch("/status");
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

export async function getLog(which: "server" | "client"): Promise<string> {
  const r = await fetch(`/logs/${which}`);
  if (!r.ok) return `(log ${which} unavailable)`;
  return (await r.json()).text as string;
}

/** POST a control endpoint; surface the API's error detail on failure. */
export async function post(path: string, body?: unknown): Promise<void> {
  const r = await fetch(path, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
}
