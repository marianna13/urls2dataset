"""the downloader module handles the downloading"""

import math
import time
import pyarrow as pa
import traceback

import fsspec

from multiprocessing.pool import ThreadPool
from threading import Semaphore
from typing import List, Any
import numpy as np

from .data_reader import DataReader
from .logger import CappedCounter
from .logger import write_stats
from .subsamplers import Subsampler
from .filters import Filter


def compute_key(key, shard_id, oom_sample_per_shard, oom_shard_count):
    true_key = (10**oom_sample_per_shard) * shard_id + key
    key_format = oom_sample_per_shard + oom_shard_count
    str_key = "{true_key:0{key_format}d}".format(  # pylint: disable=consider-using-f-string
        key_format=key_format, true_key=true_key
    )
    return str_key


class TextCleaner:
    def __init__(self):
        from pii_transform.api.e2e.multilang import MultiPiiTextProcessor
        configfile ='piisa-config.yml'
        self.proc = MultiPiiTextProcessor(lang=["en", "es", "de", 'fr', "ru", "zh", "po", "pt", "it", "no", "ua", "hi", "se", "ne", "tr", "ar", 'jp', "ko"], config=configfile, 
                            keep_piic=False, debug=None)
    def __call__(self, text, lang):
        try:
            cleaned_text = self.proc(text, lang)
        except:
            cleaned_text = text
        return cleaned_text


class DownloadWorker:
    """The downloader class gets calls with shards, download them then call the writer to write them down"""

    def __init__(
        self,
        sample_writer_class,
        save_caption,
        output_folder,
        column_list,
        thread_count,
        timeout,
        number_sample_per_shard,
        oom_shard_count,
        tmp_dir,
        config,
        postprocess_func,
        common_crawl,
        filters_config,
        clean_text
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.save_caption = save_caption
        self.output_folder = output_folder
        self.column_list = column_list
        self.number_sample_per_shard = number_sample_per_shard
        self.oom_shard_count = oom_shard_count
        self.thread_count = thread_count
        self.common_crawl = common_crawl
        self.config = config
        self.postprocess_func = postprocess_func
        self.filters_config = filters_config
        self.data_reader = DataReader(timeout, tmp_dir=tmp_dir, config=config, common_crawl=common_crawl)
        self.clean_text = clean_text
        if clean_text:
            self.proc_text = TextCleaner()


    def __call__(
        self,
        row,
    ):
        try:
            self.download_shard(row)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def download_shard(
        self,
        row,
    ):
        """Function to start an video downloading in one process"""

        shard_id, shard_file = row
        start_time = time.time()

        fs, shard_path = fsspec.core.url_to_fs(shard_file)
        with fs.open(shard_path, "rb") as f:
            df = pa.ipc.open_file(f).read_all()
        schema = df.schema
        schema = (
            schema.append(pa.field("key", pa.string()))
            .append(pa.field("status", pa.string()))
            .append(pa.field("error_message", pa.string()))
        )

        if self.config.get("media_elems"):
            schema = schema.append(pa.field("media", pa.binary()))
        if self.postprocess_func is not None:
            schema = schema.append(pa.field("postproc_value", pa.binary()))
        schema = schema.append(pa.field("language", pa.string()))

        pydict = df.select(self.column_list).to_pydict()
        shard_to_dl = list(enumerate(zip(*(pydict[col] for col in self.column_list))))
        del pydict
        del df

        status_dict = CappedCounter()

        count = len(shard_to_dl)
        successes = 0
        failed_to_download = 0
        failed_to_subsample = 0
        bytes_downloaded = 0
        url_indice = self.column_list.index("url")
        caption_indice = self.column_list.index("caption") if "caption" in self.column_list else None
        key_url_list = [(key, x[url_indice]) for key, x in shard_to_dl]

        semaphore = Semaphore(self.thread_count)

        if self.common_crawl:
            pass

        def data_generator():
            for e in key_url_list:
                # semaphore.acquire()  # pylint: disable=(consider-using-with)
                yield e

        loader = data_generator()

        subsampler = Subsampler(func=self.postprocess_func)
        try:
            _filter = Filter(**self.filters_config)
        except:
            _filter = lambda x: x

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id,
            self.output_folder,
            self.save_caption,
            self.oom_shard_count,
            schema,
        )
        oom_sample_per_shard = math.ceil(math.log10(self.number_sample_per_shard))

        with ThreadPool(self.thread_count) as thread_pool:
            for key, texts, media, error_message in thread_pool.imap_unordered(
                self.data_reader,  # pylint: disable=(unnecessary-lambda)
                loader,
            ):
                try:
                    _, sample_data = shard_to_dl[key]
                    str_key = compute_key(key, shard_id, oom_sample_per_shard, self.oom_shard_count)
                    meta = {
                        **{self.column_list[i]: sample_data[i] for i in range(len(self.column_list))},
                        "media": media,
                        "key": str_key,
                        "status": None,
                        "error_message": error_message,
                    }
                    if error_message is not None:

                        failed_to_download += 1
                        status = "failed_to_download"
                        status_dict.increment(error_message)
                        meta["status"] = status
                        sample_writer.write(
                            {},
                            str_key,
                            sample_data[caption_indice] if caption_indice is not None else None,
                            meta,
                        )
                        semaphore.release()
                        continue

                    if self.postprocess_func is not None:
                        value, error_message = subsampler(texts)
                        if error_message is None:
                            meta["postproc_value"] = value
                        else:
                            meta["postproc_value"] = None

                    bytes_downloaded += len(texts)

                    metas = [meta]

                    # if error_message is not None:
                    #     failed_to_subsample += 1
                    #     status = "failed_to_subsample"
                    #     status_dict.increment(error_message)
                    #     meta["status"] = status
                    #     meta["error_message"] = error_message

                    #     sample_writer.write(
                    #         {},
                    #         key,
                    #         caption,
                    #         meta,
                    #     )
                    # continue
                    sample = {**meta, "text": texts}
                    if not _filter(sample):
                        continue

                    del sample

                    successes += 1
                    status = "success"
                    status_dict.increment(status)

                    meta["status"] = status

                    if self.clean_text:
                        texts = self.proc_text(texts, lang=media['language'])

                    text_caption = sample_data[caption_indice] if caption_indice is not None else None
                    sample_writer.write(
                        texts,
                        meta["key"],
                        text_caption,
                        meta,
                    )
                except Exception as err:  # pylint: disable=broad-except
                    traceback.print_exc()
                    print(f"Sample {key} failed to download: {err}")
                semaphore.release()

            sample_writer.close()
            thread_pool.terminate()
            thread_pool.join()
            del thread_pool

        end_time = time.time()
        write_stats(
            self.output_folder,
            shard_id,
            count,
            successes,
            failed_to_download,
            failed_to_subsample,
            bytes_downloaded,
            start_time,
            end_time,
            status_dict,
            self.oom_shard_count,
        )
        fs.rm(shard_path)
