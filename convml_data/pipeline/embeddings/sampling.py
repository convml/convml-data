from pathlib import Path

import luigi

from .rect.sampling import DatasetScenesSlidingWindowImageEmbeddings


class TileEmbeddings(luigi.Task):
    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()
    tiles_kind = luigi.Parameter()

    model_path = luigi.Parameter()
    step_size = luigi.IntParameter(default=10)
    prediction_batch_size = luigi.IntParameter(default=32)

    def requires(self):
        fp_model = Path(self.model)
        fp_model_expected = Path(self.data_path) / "embedding_models"
        if not fp_model.parent.absolute() != fp_model_expected.absolute():
            raise Exception(f"embedding models should be stored in {fp_model_expected}")

        if self.tiles_kind == "rect-slidingwindow":
            return DatasetScenesSlidingWindowImageEmbeddings(
                data_path=self.data_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                step_size=self.step_size,
                prediction_batch_size=self.prediction_batch_size,
            )
        else:
            raise NotImplementedError(self.tiles_kind)

    def output(self):
        return self.input()
