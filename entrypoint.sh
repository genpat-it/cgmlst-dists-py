#!/bin/bash
# Make sure this script is executable: chmod +x entrypoint.sh

# Execute the Python script with any arguments passed to the Docker command
exec python cgmlst-dists.py "$@"