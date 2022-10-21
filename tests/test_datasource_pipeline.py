import functools
import socket
from pathlib import Path

import luigi

from convml_data import DataSource
from convml_data.pipeline import GenerateRegriddedScenes, GenerateTiles

EXAMPLE_FILEPATH = str(Path(__file__).parent / "example")
HAS_JASMIN_ACCESS = socket.getfqdn() in ["thixo"]


def test_make_triplets():
    datasource = DataSource.load(EXAMPLE_FILEPATH)
    TileTask = functools.partial(
        GenerateTiles,
        data_path=EXAMPLE_FILEPATH,
        tiles_kind="triplets",
    )

    tasks = [TileTask()]

    aux_products = list(datasource.aux_products.keys())
    for aux_product_name in aux_products:
        if (
            datasource.aux_products[aux_product_name]["source"] == "era5"
            and not HAS_JASMIN_ACCESS
        ):
            continue

        tasks.append(
            TileTask(
                aux_name=aux_product_name,
            )
        )

    assert luigi.build(tasks, local_scheduler=True)


def test_make_regridded_domain_data():
    datasource = DataSource.load(EXAMPLE_FILEPATH)

    tasks = [
        GenerateRegriddedScenes(
            data_path=EXAMPLE_FILEPATH,
        )
    ]

    aux_products = list(datasource.aux_products.keys())
    for aux_product_name in aux_products:
        if (
            datasource.aux_products[aux_product_name]["source"] == "era5"
            and not HAS_JASMIN_ACCESS
        ):
            continue

        tasks.append(
            GenerateRegriddedScenes(
                data_path=EXAMPLE_FILEPATH,
                aux_name=aux_product_name,
            )
        )

    assert luigi.build(tasks, local_scheduler=True)
