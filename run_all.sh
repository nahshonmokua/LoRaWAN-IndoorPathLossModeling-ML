#!/bin/bash
# Runs the notebooks in order, each on its kernel, and writes the executed cells in place, as for the reported runs.
# Notebooks 06 and 14 need the RAPIDS kernel (see README, Environments). The whole chain takes about 40 h on 32 cores and one GPU.
set -e
cd "$(dirname "$0")"
for nb in [0-9][0-9]_*.ipynb; do
    case "$nb" in 06_*|14_*) kernel=lorawan-pathloss-rapids ;; *) kernel=lorawan-pathloss ;; esac
    echo "$(date '+%F %T')  $nb  ($kernel)"
    python - "$nb" "$kernel" <<'PY'
import sys, nbformat
from nbclient import NotebookClient
path, kernel = sys.argv[1], sys.argv[2]
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, kernel_name=kernel, timeout=None, resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, path)
PY
done
