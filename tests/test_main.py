import pytest
from urls2dataset import urls2dataset
import os


@pytest.mark.parametrize("url_list", ["test-files/urls.txt"])
def test_urls2dataset(url_list):
    output_folder = "test_output"
    urls2dataset(
        url_list=url_list,
        input_format="txt",
        output_format="parquet",
        output_folder=output_folder,
        processes_count=1,
        number_sample_per_shard=100,
        thread_count=1,
    )

    assert len(os.listdir(output_folder)) > 0
