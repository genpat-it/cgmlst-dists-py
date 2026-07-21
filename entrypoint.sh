#!/bin/bash
# Make sure this script is executable: chmod +x entrypoint.sh

# Numba's @jit(cache=True) needs a writable directory to store the compiled
# cache. When the container is run as an arbitrary user (e.g. `docker run -u $UID`)
# neither /app nor $HOME are writable, so numba aborts at import time with
# "cannot cache function ...: no locator available for file '/app/cgmlst-dists.py'".
# Point the cache at a writable temp dir unless the caller already provided one.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$(mktemp -d)}"

# Execute the Python script with any arguments passed to the Docker command
exec python cgmlst-dists.py "$@"
