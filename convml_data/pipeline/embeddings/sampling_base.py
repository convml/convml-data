from pathlib import Path

import luigi

from .defaults import TRIPLET_EMBEDDINGS_KWARGS
from .rect.defaults import SLIDING_WINDOW_EMBEDDINGS_DEFAULT_KWARGS


def model_identifier_from_filename(fn):
    return fn.replace(".torch.pkl", "").replace(".ckpt", "")


def make_embedding_name(kind, model_filename, **model_args):
    if kind == "rect-slidingwindow":
        full_args = dict(SLIDING_WINDOW_EMBEDDINGS_DEFAULT_KWARGS)
    elif kind == "triplets":
        full_args = dict(TRIPLET_EMBEDDINGS_KWARGS)
    else:
        raise NotImplementedError(kind)

    full_args.update(model_args)

    skip_args = ["prediction_batch_size"]
    name_parts = [model_identifier_from_filename(model_filename), kind]
    name_parts += [f"{k}__{v}" for (k, v) in full_args.items() if k not in skip_args]

    return ".".join(name_parts)


class _EmbeddingModelMixin(object):
    model_filename = luigi.Parameter()

    @property
    def model_path(self):
        fp_model = Path(self.data_path) / "embeddings" / "models" / self.model_filename
        return fp_model
