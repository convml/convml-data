import functools
import socket
from pathlib import Path

import luigi
import pytest

from convml_data import DataSource
from convml_data.pipeline import (
    GenerateRegriddedScenes,
    GenerateSceneIDs,
    GenerateTiles,
)
from convml_data.pipeline.aux_sources import CheckForAuxiliaryFiles
from convml_data.sources.ceres_syn1deg_modis.earthaccess_auth import auth as ea_auth

# from convml_data.pipeline.embeddings.sampling import DatasetScenesTileEmbeddings

EXAMPLE_FILEPATH = str(Path(__file__).parent / "example")
HAS_JASMIN_ACCESS = socket.getfqdn() in ["thixo"]

ea = ea_auth.Auth().login(strategy="netrc")
HAS_EARTHDATA_ACCESS = ea_auth.Auth().login(strategy="netrc").authenticated


def _data_source_available(source_name):
    if source_name == "era5" and not HAS_JASMIN_ACCESS:
        return False

    if source_name == "ceres_syn1deg_modis" and not HAS_EARTHDATA_ACCESS:
        return False

    return True


def test_make_triplets():
    datasource = DataSource.load(EXAMPLE_FILEPATH)
    TileTask = functools.partial(  # noqa
        GenerateTiles,
        data_path=EXAMPLE_FILEPATH,
        tiles_kind="triplets",
    )

    tasks = [TileTask()]

    aux_products = list(datasource.aux_products.keys())
    for aux_product_name in aux_products:
        product_source = datasource.aux_products[aux_product_name]["source"]
        if not _data_source_available(product_source):
            continue

        tasks.append(
            TileTask(
                aux_name=aux_product_name,
            )
        )

    assert luigi.build(tasks, local_scheduler=True)

    # task_embs = DatasetScenesTileEmbeddings(data_path=EXAMPLE_FILEPATH)
    # assert luigi.build(task_embs, local_scheduler=True)


def _parse_example_aux_products():
    datasource = DataSource.load(EXAMPLE_FILEPATH)
    aux_product_names = []

    for aux_product_name, kwargs in datasource.aux_products.items():
        product_source = kwargs["source"]
        if not _data_source_available(product_source):
            continue

        aux_product_names.append(aux_product_name)
    return aux_product_names


AUX_NAMES = [
    None,
] + _parse_example_aux_products()


def _delete_if_exists(path):
    if Path(path).exists():
        Path(path).unlink()


@pytest.mark.parametrize("aux_product_name", AUX_NAMES)
def test_find_scene_source_data(aux_product_name):
    if aux_product_name is None:
        task = GenerateSceneIDs(data_path=EXAMPLE_FILEPATH)
    else:
        task = CheckForAuxiliaryFiles(
            data_path=EXAMPLE_FILEPATH,
            aux_name=aux_product_name,
        )

        # uncomment to also remove result of queries for data
        # for _, reqs in task.requires()["product"].items():
        #     if not isinstance(reqs, list):
        #         reqs = [reqs]
        #     for req in reqs:
        #         _delete_if_exists(req.output().path)

    _delete_if_exists(task.output().path)

    assert luigi.build([task], local_scheduler=True)


@pytest.mark.parametrize("aux_product_name", AUX_NAMES)
def test_make_regridded_domain_data(aux_product_name):
    task = GenerateRegriddedScenes(
        data_path=EXAMPLE_FILEPATH,
        aux_name=aux_product_name,
    )

    assert luigi.build([task], local_scheduler=True)
