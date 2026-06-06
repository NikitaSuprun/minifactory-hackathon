export interface Status {
  connected: boolean;
  teleop_running: boolean;
  inference_running: boolean;
  inference_status: string;
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
}
