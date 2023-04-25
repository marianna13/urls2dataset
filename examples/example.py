from urls2dataset import urls2dataset


if __name__ == "__main__":

    urls2dataset(
        url_list="test_cc.txt",
        input_format="txt",
        output_format="parquet",
        output_folder="data",
        processes_count=16,
        number_sample_per_shard=1000,
        thread_count=16,
    )

