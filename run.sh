#!/usr/bin/bash

# edited run.sh script to run mount.sh

if [ "$LOCAL_RANK" = "0" ] && [ -z "$DISABLE_MOUNT" ]; then
    source /packages/torchx_conda_mount/mount.sh
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR &&
python3 "$@"
