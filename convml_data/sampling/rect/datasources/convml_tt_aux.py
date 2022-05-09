"""
Module for ingesting cropped auxiliary products produced by convml_tt source
data pipeline
"""
from pathlib import Path

import luigi

from ..utils import BBoxParameter, XArrayTarget

try:
    from convml_tt.data.sources.pipeline.sampling import CropSceneSourceFiles

    HAS_CONVML = True
except ImportError:
    HAS_CONVML = False


class CroppedAuxProduct(luigi.Task):
    datapath = luigi.Parameter(default=".")
    scene_id = luigi.Parameter()
    var_name = luigi.Parameter()
    bbox = BBoxParameter()

    @property
    def product_name(self):
        assert self.var_name.startswith("goes16_")
        var_name = self.var_name.replace("goes16_", "")
        return var_name

    def requires(self):
        if HAS_CONVML:
            return CropSceneSourceFiles(
                scene_id=self.scene_id,
                data_path=self.datapath,
                aux_product=self.product_name,
            )

    def output(self):
        if HAS_CONVML:
            return self.requires().output()["data"]
        else:
            fn = f"{self.scene_id}.nc"
            fp = (
                Path(self.datapath)
                / "source_data"
                / "goes16"
                / "aux"
                / self.product_name
                / "cropped"
                / fn
            )
            return XArrayTarget(str(fp))
