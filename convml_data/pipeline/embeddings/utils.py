import os
from pathlib import Path


def symlink_external_models(data_path, model_filename, external_data_path):
    """
    Create symlink to enable use of model trained on one dataset (in
    `data_path`) to be used with a different dataset (in `external_data_path`).

    For two datasets:

    - `eurec4a-2018-grl-triplets` (stored in `data_path`) on which we've
      trained a model (with `model_filename`), and

    - `eurec4a-2018202-winter-midday` (stored in `external_data_path`)

    we want to use the trained model from the former dataset in the former.
    This is achieved by creating a symlink in
    `embeddings/models/{dataset_name}__{model_identifier}` where `dataset_name`
    is inferred from the relative path names.

    data
    ├── eurec4a-20182020-winter-midday
    │   └── embeddings
    │       └── models
    │           └── eurec4a-2018-grl-triplets__fixednorm-stage-2.triplets.stage__train.tile_type__anchor.isomap.joblib
    │               -> ../../../eurec4a-2018-grl-triplets/embeddings/models/fixednorm-stage-2.triplets.stage__train.tile_type__anchor.isomap.joblib
    └── eurec4a-2018-grl-triplets
        └── embeddings
            └── models
                └── fixednorm-stage-2.triplets.stage__train.tile_type__anchor.isomap.joblib


    """

    # find commont prefix for the two data paths, we usually store all datasets
    # in a common root
    prefix = os.path.commonpath([data_path, external_data_path])
    data_path_rel = Path(data_path).relative_to(prefix)
    external_data_path_rel = Path(external_data_path).relative_to(prefix)

    dataset_identifier = str(data_path_rel).replace("/", "__")
    local_model_filename = f"{dataset_identifier}__{model_filename}"

    model_src_path = (
        Path(prefix) / data_path_rel / "embeddings" / "models" / model_filename
    )
    assert model_src_path.exists()

    models_dst_path = Path(prefix) / external_data_path_rel / "embeddings" / "models"
    models_dst_path.mkdir(exist_ok=True, parents=True)

    ext_model_link_path = models_dst_path / local_model_filename

    if not ext_model_link_path.is_symlink():
        ext_model_link_path.parent.mkdir(exist_ok=True, parents=True)
        # for some reason os.path.relpath adds an extra `../` in front of the
        # path, so we need to snip that off. Can't use
        # pathlib.Path.relative_to() here because that requires the provided
        # paths to be subpaths
        ext_model_link_path.symlink_to(
            os.path.relpath(model_src_path, ext_model_link_path)[3:]
        )
    elif ext_model_link_path.is_symlink():
        pass
    elif ext_model_link_path.exists():
        raise Exception(
            f"{ext_model_link_path} exists but it is not a symlink, which it should be (!)"
        )
    return local_model_filename
