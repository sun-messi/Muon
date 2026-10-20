#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 ORIGINAL_PROJECT_ROOT USER@HOST REMOTE_ABSOLUTE_PROJECT_ROOT" >&2
  exit 2
fi
source_root=$1
remote=$2
target=$3
[[ "$remote" =~ ^[A-Za-z0-9_.@-]+$ ]] || { echo "Invalid SSH destination" >&2; exit 2; }
[[ "$target" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo "Use an absolute destination without shell metacharacters" >&2; exit 2; }
[[ "$target" != / ]] || { echo "Refusing filesystem-root destination" >&2; exit 2; }
for id in 2 3 4; do
  cache="$source_root/runs/real_task_eight_theme_cpt/run$id/cache"
  [[ -d "$cache" ]] || { echo "Missing source cache: $cache" >&2; exit 2; }
done
ssh "$remote" "mkdir -p -- '$target'"
for id in 2 3 4; do
  # -L is essential: the original cache directories are absolute symlinks.
  rsync -aL --relative --info=progress2 \
    --include='*/' --include='*.npy' --include='manifest.json' \
    --include='document_ids.json' --exclude='*' \
    "$source_root/./runs/real_task_eight_theme_cpt/run$id/cache/" "$remote:$target/"
  comparison=comparison_v2
  if [[ "$id" == 2 ]]; then comparison=comparison_v3; fi
  rsync -a --relative \
    "$source_root/./runs/real_task_eight_theme_cpt/run$id/$comparison/endpoints.csv" \
    "$remote:$target/"
done
echo "Transfer complete. On the destination run: python scripts/verify_reference_caches.py"
