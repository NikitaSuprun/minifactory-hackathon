export interface Status {
  connected: boolean;
  teleop_running: boolean;
  inference_running: boolean;
  inference_status: "idle" | "prewarming" | "ready" | "running" | "error";
  inference_following: boolean;
  control_fps: number;
  camera_fps: { phone: number; wrist: number; camera3: number };
  joints: Record<string, number>;
  error: string | null;
  device: string;
  policy: string;
  task: string;
  server_reachable: boolean;
  follower_port: string | null;
  leader_port: string | null;
  recording_running: boolean;
  record_status: string;
  record_repo_id: string | null;
  record_last_done_repo: string | null;
  record_current_episode: number;
  record_total_episodes: number;
  record_started_at: number; // epoch seconds; 0 when idle
  record_phase_started_at: number;
  record_phase_time_s: number;
}

export interface DatasetInfo {
  repo_id: string;
  total_episodes: number;
  total_frames: number;
  fps: number;
  cameras: string[];
}

export interface VerifyResult {
  local: { total_episodes: number; total_frames: number; video_files: number };
  hub: {
    exists: boolean;
    video_files?: number;
    has_info?: boolean;
    error?: string;
  };
  match: boolean;
}
