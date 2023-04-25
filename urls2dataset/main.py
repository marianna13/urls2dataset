import fsspec
from .input_sharder import InputSharder
from .download_worker import DownloadWorker
from typing import List, Optional
import os
from .data_writer import (
    WebDatasetSampleWriter,
    ParquetSampleWriter,
)
from .distributor import (
    multiprocessing_distributor,
)


def identity(x):
    return x


def urls2dataset(
    url_list: str,
    output_folder: str = "data",
    processes_count: int = 1,
    thread_count: int = 16,
    output_format: str = "files",
    input_format: str = "parquet",
    url_col: str = "url",
    save_additional_columns: Optional[List[str]] = None,
    caption_col: Optional[str] = None,
    number_sample_per_shard: int = 10000,
    incremental_mode: str = "incremental",
    timeout: int = 3,
    oom_shard_count: int = 5,
    tmp_dir: str = "/tmp",
    subjob_size: int = 1000,
    max_shard_retry: int = 1,
    config={},
):
    """
    extract text from webpage links
    """

    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    output_folder = make_path_absolute(output_folder)
    url_list = make_path_absolute(url_list)

    tmp_path = output_folder + "/_tmp"
    fs, run_tmp_dir = fsspec.core.url_to_fs(tmp_path)
    if not fs.exists(run_tmp_dir):
        fs.mkdir(run_tmp_dir)

    sampler = identity

    fs, output_path = fsspec.core.url_to_fs(output_folder)

    if not fs.exists(output_path):
        fs.mkdir(output_path)
        done_shards = set()
    else:
        if incremental_mode == "incremental":
            done_shards = set(int(x.split("/")[-1].split("_")[0]) for x in fs.glob(output_path + "/*.json"))
        elif incremental_mode == "overwrite":
            fs.rm(output_path, recursive=True)
            fs.mkdir(output_path)
            done_shards = set()
        else:
            raise ValueError(f"Unknown incremental mode {incremental_mode}")

    if output_format == "webdataset":
        sample_writer_class = WebDatasetSampleWriter
    elif output_format == "parquet":
        sample_writer_class = ParquetSampleWriter  # type: ignore
    else:
        raise ValueError(f"Invalid output format {output_format}")

    save_caption = caption_col is not None

    shard_iterator = InputSharder(
        url_list,
        input_format,
        url_col,
        caption_col,
        save_additional_columns,
        number_sample_per_shard,
        done_shards,
        tmp_path,
        sampler,
    )

    worker = DownloadWorker(
        sample_writer_class=sample_writer_class,
        save_caption=save_caption,
        output_folder=output_folder,
        column_list=shard_iterator.column_list,
        thread_count=thread_count,
        timeout=timeout,
        number_sample_per_shard=number_sample_per_shard,
        oom_shard_count=oom_shard_count,
        tmp_dir=tmp_dir,
        config=config,
        common_crawl=input_format == "cc",
    )

    distributor_fn = multiprocessing_distributor

    distributor_fn(
        processes_count,
        worker,
        shard_iterator,
        subjob_size,
        max_shard_retry,
    )


def main():
    fire.Fire(video2dataset)


if __name__ == "__main__":
    main()
