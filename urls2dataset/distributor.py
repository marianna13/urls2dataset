"""distributor defines the distribution strategies for img2dataset"""
import os
import time
import subprocess
import yaml
from datetime import datetime
from contextlib import contextmanager
from multiprocessing import get_context
from itertools import islice, chain

import fsspec
from tqdm import tqdm


def retrier(runf, failed_shards, max_shard_retry):
    # retry failed shards max_shard_retry times
    for i in range(max_shard_retry):
        if len(failed_shards) == 0:
            break
        print(f"Retrying {len(failed_shards)} shards, try {i+1}")
        failed_shards = runf(failed_shards)
    if len(failed_shards) != 0:
        print(
            f"Retried {max_shard_retry} times, but {len(failed_shards)} shards "
            "still failed. You may restart the same command to retry again."
        )


def multiprocessing_distributor(processes_count, worker, input_sharder, _, max_shard_retry):
    """Distribute the work to the processes using multiprocessing"""
    ctx = get_context("spawn")
    with ctx.Pool(processes_count, maxtasksperchild=5) as process_pool:

        def run(gen):
            failed_shards = []
            for (status, row) in tqdm(process_pool.imap_unordered(worker, gen)):
                if status is False:
                    failed_shards.append(row)
            return failed_shards

        failed_shards = run(input_sharder)

        retrier(run, failed_shards, max_shard_retry)

        process_pool.terminate()
        process_pool.join()
        del process_pool
