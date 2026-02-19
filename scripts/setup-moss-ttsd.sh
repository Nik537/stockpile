#!/usr/bin/env bash
# setup-moss-ttsd.sh
#
# End-to-end setup script for the MOSS-TTSD RunPod serverless worker.
# Builds the slim Docker image, pushes it to Docker Hub, prints RunPod
# endpoint creation instructions, and writes the endpoint ID back into .env.
#
# Usage: bash scripts/setup-moss-ttsd.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
die()     { error "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Locate repo root (works wherever the script is called from)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKER_DIR="$REPO_ROOT/runpod-moss-ttsd-worker"
DOCKERFILE="$WORKER_DIR/Dockerfile.slim"
ENV_FILE="$REPO_ROOT/.env"
IMAGE="techtawn/moss-ttsd-runpod:latest"

echo ""
echo -e "${BOLD}=== MOSS-TTSD RunPod Setup ===${RESET}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Prerequisites check
# ---------------------------------------------------------------------------
info "Checking prerequisites..."

# Docker daemon
if ! docker info >/dev/null 2>&1; then
    die "Docker is not running. Please start Docker Desktop and try again."
fi
success "Docker is running."

# gh CLI
if ! command -v gh >/dev/null 2>&1; then
    die "GitHub CLI (gh) is not installed. Install from https://cli.github.com/"
fi
success "GitHub CLI found."

# Logged in to Docker Hub as techtawn
DOCKER_USER="$(docker system info --format '{{.Name}}' 2>/dev/null || true)"
# docker info Name is the daemon name, not the login user — check via config
DOCKER_LOGIN_USER="$(docker system info 2>/dev/null | grep 'Username' | awk '{print $2}' || true)"
if [[ "$DOCKER_LOGIN_USER" != "techtawn" ]]; then
    warn "Docker Hub login user is '${DOCKER_LOGIN_USER:-<not logged in>}', expected 'techtawn'."
    info "Attempting 'docker login' now..."
    docker login || die "Docker login failed. Please run 'docker login' manually."
fi
success "Docker Hub authentication OK."

# Slim Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
    die "Slim Dockerfile not found at $DOCKERFILE"
fi
success "Slim Dockerfile found."

echo ""

# ---------------------------------------------------------------------------
# Step 2: Build
# ---------------------------------------------------------------------------
info "Building slim image: ${IMAGE}"
info "Platform: linux/amd64"
info "Context:  ${WORKER_DIR}"
echo ""

docker build \
    --platform linux/amd64 \
    -f "$DOCKERFILE" \
    -t "$IMAGE" \
    "$WORKER_DIR"

success "Image built successfully: ${IMAGE}"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Push
# ---------------------------------------------------------------------------
info "Pushing image to Docker Hub..."
docker push "$IMAGE"
success "Image pushed: ${IMAGE}"
echo ""

# ---------------------------------------------------------------------------
# Step 4: RunPod endpoint creation instructions
# ---------------------------------------------------------------------------
echo -e "${BOLD}=== Create RunPod Serverless Endpoint ===${RESET}"
echo ""
echo -e "  1. Go to ${CYAN}https://runpod.io/console/serverless${RESET}"
echo ""
echo -e "  2. Click ${BOLD}+ New Endpoint${RESET} and use these settings:"
echo ""
echo -e "     ${YELLOW}Container Image:${RESET}    techtawn/moss-ttsd-runpod:latest"
echo -e "     ${YELLOW}GPU Type:${RESET}           A5000 (24GB) or L40S (48GB) or A100 (40/80GB)"
echo -e "     ${YELLOW}Min Workers:${RESET}        0  (scale to zero)"
echo -e "     ${YELLOW}Max Workers:${RESET}        3  (adjust to your load)"
echo -e "     ${YELLOW}Idle Timeout:${RESET}       60 seconds"
echo -e "     ${YELLOW}Execution Timeout:${RESET}  600 seconds (10 minutes)"
echo -e "     ${YELLOW}Container Disk:${RESET}     50 GB  (needed for model cache)"
echo ""
echo -e "  3. After the endpoint is created, copy its ${BOLD}Endpoint ID${RESET}"
echo -e "     (looks like: abc123def456)"
echo ""

# ---------------------------------------------------------------------------
# Step 5: Collect endpoint ID from user
# ---------------------------------------------------------------------------
read -rp "$(echo -e ${BOLD}"Enter your RunPod Endpoint ID (or press Enter to skip): "${RESET})" ENDPOINT_ID

if [[ -z "$ENDPOINT_ID" ]]; then
    warn "No endpoint ID entered. Skipping .env update and commit."
    echo ""
    info "You can set it manually later:"
    echo "  RUNPOD_MOSS_TTSD_ENDPOINT_ID=<your-id>"
    echo ""
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 6: Update .env
# ---------------------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    warn ".env file not found at $ENV_FILE — creating it."
    touch "$ENV_FILE"
fi

if grep -q "^RUNPOD_MOSS_TTSD_ENDPOINT_ID=" "$ENV_FILE" 2>/dev/null; then
    # Replace existing line (macOS-compatible sed in-place)
    sed -i '' "s/^RUNPOD_MOSS_TTSD_ENDPOINT_ID=.*/RUNPOD_MOSS_TTSD_ENDPOINT_ID=${ENDPOINT_ID}/" "$ENV_FILE"
    success "Updated RUNPOD_MOSS_TTSD_ENDPOINT_ID in .env"
else
    echo "RUNPOD_MOSS_TTSD_ENDPOINT_ID=${ENDPOINT_ID}" >> "$ENV_FILE"
    success "Appended RUNPOD_MOSS_TTSD_ENDPOINT_ID to .env"
fi

echo ""

# ---------------------------------------------------------------------------
# Step 7: Commit and push to trigger Render redeploy
# ---------------------------------------------------------------------------
info "Committing and pushing to trigger Render redeploy..."

cd "$REPO_ROOT"

# Stage only relevant files (never commit .env)
git add \
    runpod-moss-ttsd-worker/Dockerfile.slim \
    render.yaml \
    .github/workflows/build-moss.yml \
    scripts/setup-moss-ttsd.sh 2>/dev/null || true

# Commit if there are staged changes
if git diff --cached --quiet; then
    info "No new file changes to commit (files already staged/committed)."
else
    git commit -m "feat: configure MOSS-TTSD RunPod endpoint (${ENDPOINT_ID})"
    success "Committed changes."
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git push origin "$CURRENT_BRANCH"
success "Pushed to origin/${CURRENT_BRANCH} — Render will redeploy automatically."

echo ""
echo -e "${BOLD}${GREEN}=== Setup Complete ===${RESET}"
echo ""
echo -e "  Endpoint ID : ${CYAN}${ENDPOINT_ID}${RESET}"
echo -e "  Docker image: ${CYAN}${IMAGE}${RESET}"
echo ""
echo -e "  ${YELLOW}Reminder:${RESET} Add RUNPOD_MOSS_TTSD_ENDPOINT_ID to your Render"
echo -e "  dashboard environment variables if not already present."
echo ""
