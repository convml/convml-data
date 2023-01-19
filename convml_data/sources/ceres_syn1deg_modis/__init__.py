"""
Module for fetching and work with the CERES-Syn1deg-1Hour radiation dataset
from
https://asdc.larc.nasa.gov/project/CERES/CER_SYN1deg-1Hour_Terra-Aqua-MODIS_Edition4A
"""

from .extract import extract_variable  # noqa
from .pipeline import QueryForData  # noqa
