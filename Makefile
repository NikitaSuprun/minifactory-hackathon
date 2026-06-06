# Minifactory SO-101 — common commands. Run `make` to list targets.
.DEFAULT_GOAL := help
.PHONY: help dashboard find-port calibrate-follower calibrate-leader \
        server-deploy server-logs tunnel client check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## --- Mac (arm host) ---
dashboard: ## Run the operator dashboard (http://localhost:8041)
	uv run python arm_dashboard.py

find-port: ## Discover SO-101 USB serial ports (set them in .env)
	uv run lerobot-find-port

calibrate-follower: ## Calibrate the follower arm (interactive)
	uv run lerobot-calibrate --robot.type=so101_follower \
	  --robot.port=$$(grep '^FOLLOWER_PORT=' .env | cut -d= -f2) --robot.id=so101_follower

calibrate-leader: ## Calibrate the leader arm (interactive)
	uv run lerobot-calibrate --teleop.type=so101_leader \
	  --teleop.port=$$(grep '^LEADER_PORT=' .env | cut -d= -f2) --teleop.id=so101_leader

client: ## Run the remote-inference robot client directly (dashboard does this for you)
	uv run python run_robot_client.py

tunnel: ## SSH tunnel to the GPU box (only if not using Tailscale)
	uv run python run_tunnel.py

## --- GPU box (inference) ---
server-deploy: ## Deploy + (re)start the policy server on the GPU box (needs .env.local HF_TOKEN)
	HF_TOKEN=$$(grep '^HF_TOKEN=' .env.local | cut -d= -f2-) \
	  uvx --from ansible-core ansible-playbook -i deploy/inventory.ini deploy/playbook.yml

server-logs: ## Tail the policy server log on the GPU box
	ssh $$(grep '^GPU_SSH_HOST=' .env | cut -d= -f2) \
	  'tail -n 60 ~/minifactory-hackathon/policy_server.out'

## --- dev ---
check: ## Lint + type-check the Python code
	uv run ruff check . && uv run ruff format --check . && uv run pyright
