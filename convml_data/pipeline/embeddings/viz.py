from pathlib import Path

import luigi
from convml_tt.interpretation.plots import isomap2d as isomap2d_plot

from .sampling import AggregatedDatasetScenesTileEmbeddings, make_embedding_name


class TripletEmbeddingsManifoldPlot2D(luigi.Task):
    data_path = luigi.Parameter(default=".")

    tile_size = luigi.OptionalFloatParameter(default=0.05)
    dl_sampling = luigi.OptionalFloatParameter(default=0.1)

    model_path = luigi.Parameter()
    transform_method = luigi.Parameter(default="isomap")

    def requires(self):
        kwargs = dict(
            tiles_kind="triplets",
            data_path=self.data_path,
            model_path=self.model_path,
        )

        TaskClass = AggregatedDatasetScenesTileEmbeddings
        tasks = {}
        tasks["triplet_embeddings"] = TaskClass(
            **kwargs,
        )

        tasks["triplet_anchor_embeddings_manifold"] = TaskClass(
            embedding_transform=self.transform_method,
            model_args=dict(tile_type="anchor"),
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

        fig, _, _ = isomap2d_plot.make_manifold_reference_plot(
            da_embs=da_triplet_embs,
            method=self.transform_method,
            da_embs_manifold=da_anchor_manifold_embs,
            tile_size=self.tile_size,
            dl=self.dl_sampling,
        )
        fig.savefig(self.output().path)

    def output(self):
        emb_name = make_embedding_name(
            kind="triplets", model_path=self.model_path, transform=self.transform_method
        )

        name_parts = [emb_name, self.transform_method, "png"]

        fn = ".".join(name_parts)

        return luigi.LocalTarget(str(Path(self.data_path) / fn))
