import luigi

from .. import utils
from . import ceres, convml_tt_aux

try:
    from . import era5
except ImportError:
    era5 = None


class FieldCropped(luigi.WrapperTask):
    scene_id = luigi.Parameter()
    var_name = luigi.Parameter()
    bbox = utils.BBoxParameter()
    datapath = luigi.Parameter()

    def requires(self):
        rad_vars = [
            "broadband_longwave_flux",
            "broadband_shortwave_albedo",
            "cloud_top_temperature",
        ]

        var_name = self.var_name

        if any(var_name.startswith(rad_var) for rad_var in rad_vars):
            TaskClass = ceres.CeresFieldCropped
        elif var_name.startswith("ceres_"):
            TaskClass = ceres.CeresFieldCropped
            var_name = var_name.replace("ceres_", "")
        elif self.var_name.startswith("goes16_"):
            TaskClass = convml_tt_aux.CroppedAuxProduct
        else:
            TaskClass = era5.ERA5FieldCropped

        return TaskClass(
            scene_id=self.scene_id,
            bbox=self.bbox,
            var_name=var_name,
            datapath=self.datapath,
        )

    def output(self):
        return self.input()
