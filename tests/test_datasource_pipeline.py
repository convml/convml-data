import functools
import socket
from pathlib import Path

import luigi

from convml_data import DataSource
from convml_data.pipeline import GenerateRegriddedScenes, GenerateTiles
from convml_data.pipeline.embeddings.sampling import DatasetScenesTileEmbeddings

try:
    from convml_tt.data.examples import PretrainedModel, fetch_pretrained_model

    HAS_CONVML_TT = True
except ImportError:
    HAS_CONVML_TT = False

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

    if HAS_CONVML_TT:
        model_path = fetch_pretrained_model(
            pretrained_model=PretrainedModel.FIXED_NORM_STAGE2,
            data_dir=Path(EXAMPLE_FILEPATH) / "embeddings" / "models",
        )
        task_embs = DatasetScenesTileEmbeddings(
            data_path=EXAMPLE_FILEPATH,
            tiles_kind="triplets",
            model_path=model_path,
            # model_path=Path(model_path).relative_to(EXAMPLE_FILEPATH),
        )
        assert luigi.build([task_embs], local_scheduler=True)


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
