#!/usr/bin/env zsh
# Wrap Nous Hermes or OpenCode inside AgFS transactional guardrails (protects tests/** and caps budget)
set -euo pipefail

agfs wrap \
  --db .agfs/state.agdb \
  --workspace . \
  --protect "tests/**" \
  --max-budget 2.50 \
  -- "$@"
