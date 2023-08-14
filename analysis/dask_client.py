"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-29
Description: This script is used to start a dask client on a local cluster.
The script should be called from the command line as a background process
with the PID saved to a variable as
    python dask_client.py &
    pid=$!
This will allow the client to run in the background while the main script
executes. The PID can be used to kill the client with
    kill $pid
"""
# Standard library
import sys
import time

# External dependencies
import dask
from dask.distributed import Client, LocalCluster

# add local src directory to path
sys.path.append("./../../src")

if __name__ == "__main__":
    # local cluster parameters
    N_JOBS = 24

    # see if dask client exists, if not, create one
    try:
        client = Client(
            "localhost:8786", timeout=2, name="Dask Client from existing cluster"
        )
    except OSError:
        dask.config.set(
            {
                "distributed.worker.memory.target": 0.6,
                "distributed.worker.memory.spill": 0.7,
                "distributed.worker.memory.pause": 1,
                "distributed.worker.memory.terminate": 1,
            }
        )
        cluster = LocalCluster(
            n_workers=N_JOBS,
            threads_per_worker=1,
            processes=True,
            memory_limit="60GB",
            scheduler_port=8786,
        )
        client = Client(cluster, name="Dask Client from new cluster")

    # sleep forever
    while True:
        time.sleep(10)
