#!/usr/bin/env bash
# Set up the design log for a data-loader-helper session.
#
# Ensures projects/<project_name>/ exists (creating it if needed) and seeds
# projects/<project_name>/design_log.md from the bundled template. Run this only
# once the project name has been decided. Existing design_log.md is left
# untouched (resume-safe), so re-running is harmless.
#
# Usage: setup_design_log.sh <project_name>
set -euo pipefail

PROJECT_NAME="${1:?usage: setup_design_log.sh <project_name>}"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SKILL_DIR" rev-parse --show-toplevel)"
PROJECT_DIR="$REPO_ROOT/projects/$PROJECT_NAME"
DEST="$PROJECT_DIR/design_log.md"

mkdir -p "$PROJECT_DIR"
if [[ -e "$DEST" ]]; then
  echo "exists: $DEST (left unchanged)"
else
  cp "$SKILL_DIR/design_log_template.md" "$DEST"
  echo "created: $DEST"
fi
