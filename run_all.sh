#!/bin/bash
# Runs the notebooks in dependency order, each on its kernel, and writes the executed cells in place, as for the reported runs. The numbers follow
# the topics: notebook 09 reads the residuals of the history models that 13 and 14 write, so it runs after them.
# Notebooks 06 and 14 need the RAPIDS kernel (see README, Environments). The whole chain takes about 50 h on 32 cores and one GPU.
set -e
cd "$(dirname "$0")"
for n in 00 01 02 03 04 05 06 07 08 10 11 12 13 14 09 15 16 17 18 19 20 21 22 23; do
    nb=$(ls "${n}"_*.ipynb)
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
