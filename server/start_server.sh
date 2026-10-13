#!/usr/bin/env bash
# Human Generator server on Linux (untested). Run install steps from the README by hand first.
cd "$(dirname "$0")/.."
unset PYTHONHOME PYTHONPATH
export TEXT_ENCODER_DEVICE="${TEXT_ENCODER_DEVICE:-cpu}"
[ -f cache/.models_ready ] && export HF_HUB_OFFLINE=1
exec .venv/bin/python -u server/hg_server.py "$@"
