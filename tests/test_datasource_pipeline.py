import tempfile

import luigi

from convml_data.examples import ExampleDatasource, fetch_example_datasource
from convml_data.pipeline import GenerateRegriddedScenes, GenerateTiles


def test_make_triplets():
    with tempfile.TemporaryDirectory() as tmpdir:
        datasource_path = fetch_example_datasource(
            ExampleDatasource.EUREC4A_SMALL, data_dir=tmpdir
        )
        task_rect_data = GenerateTiles(
            data_path=datasource_path,
            tiles_kind="triplets",
        )
        luigi.build([task_rect_data], local_scheduler=True)


def test_make_regridded_domain_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        datasource_path = fetch_example_datasource(
            ExampleDatasource.EUREC4A_SMALL, data_dir=tmpdir
        )
        task_rect_data = GenerateRegriddedScenes(
            data_path=datasource_path,
        )
        luigi.build([task_rect_data], local_scheduler=True)
