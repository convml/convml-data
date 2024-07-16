# CER_SYN1deg-1Hour_Terra-Aqua-MODIS_Edition4A_409407.20220401.hdf
TIME_FORMAT = "%Y%m%d"
FILENAME_FORMAT = "CER_SYN1deg-1Hour_Terra-Aqua-MODIS_Edition4A_{version}.{time}.hdf"

URL_MONTH_LISTING_FORMAT = (
    "https://asdc.larc.nasa.gov/data/CERES/SYN1deg-1Hour/Terra-Aqua-MODIS_Edition4A/"
    "{time:%Y}/{time:%m}/"
)

URL_FILE_FORMAT = URL_MONTH_LISTING_FORMAT + FILENAME_FORMAT.replace(
    "{time}", "{time:" + TIME_FORMAT + "}"
)
