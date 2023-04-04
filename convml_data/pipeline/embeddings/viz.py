from pathlib import Path

import luigi
import matplotlib_aximgcache as mpl_aic
from convml_tt.interpretation.plots import manifold2d as manifold2d_plot

from .sampling import (
    AggregatedDatasetScenesTileEmbeddings,
    TransformedAggregatedDatasetScenesTileEmbeddings,
    make_embedding_name,
)


class TripletEmbeddingsManifoldPlot2D(luigi.Task):
    data_path = luigi.Parameter(default=".")
    dataset_stage = luigi.Parameter()

    tile_size = luigi.OptionalFloatParameter(default=0.04)
    dl_sampling = luigi.OptionalFloatParameter(default=0.08)

    model_name = luigi.Parameter()
    transform_method = luigi.Parameter()
    cache_ax_image = luigi.BoolParameter(default=False)

    def requires(self):
        kwargs = dict(
            tiles_kind="triplets",
            data_path=self.data_path,
            model_name=self.model_name,
            model_args=dict(dataset_stage=self.dataset_stage),
        )

        tasks = {}
        tasks["triplet_embeddings"] = AggregatedDatasetScenesTileEmbeddings(
            **kwargs,
        )

        kwargs["model_args"].update(dict(tile_type="anchor"))
        tasks[
            "triplet_anchor_embeddings_manifold"
        ] = TransformedAggregatedDatasetScenesTileEmbeddings(
            embedding_transform=self.transform_method,
            **kwargs,
        )

        return tasks

    def run(self):
        inputs = self.input()

        da_triplet_embs = (
            inputs["triplet_embeddings"]
            .open()
            .set_index(triplet_tile_id=("tile_id", "tile_type"))
            .unstack("triplet_tile_id")
            .drop("scene_id")
        )

        da_anchor_manifold_embs = (
            inputs["triplet_anchor_embeddings_manifold"]
            .open()
            .swap_dims(dict(triplet_tile_id="tile_id"))
            .sortby("tile_id")
        )

        fig, ax, _ = manifold2d_plot.make_manifold_reference_plot(
            da_embs=da_triplet_embs,
            method=self.transform_method,
            da_embs_manifold=da_anchor_manifold_embs,
            tile_size=self.tile_size,
            dl=self.dl_sampling,
        )
        fig.set_dpi(300)

        if self.cache_ax_image:
            fp_aximgcache = Path(self.output()["aximgcache"].path)

            if not fp_aximgcache.exists():
                mpl_aic.save_ax_to_image(ax=ax, fpath=fp_aximgcache)

        fig.savefig(self.output()["image"].path)

    def output(self):
        emb_name = make_embedding_name(
            kind="triplets", model_name=self.model_name, transform=self.transform_method
        )

        name_parts = [emb_name, self.transform_method, "png"]

        fn = ".".join(name_parts)

        outputs = dict(image=luigi.LocalTarget(str(Path(self.data_path) / fn)))

        if not self.cache_ax_image:
            name_parts_aximgcache = list(name_parts)
            name_parts_aximgcache.insert(3, "aximgcache")
            fn_aximgcache = ".".join(name_parts_aximgcache)
            outputs["aximgcache"] = luigi.LocalTarget(
                str(Path(self.data_path) / fn_aximgcache)
            )

        return outputs
