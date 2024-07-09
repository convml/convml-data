from pathlib import Path

import luigi

from .rect.defaults import SLIDING_WINDOW_EMBEDDINGS_DEFAULT_KWARGS


def make_embedding_name(kind, model_name, **model_args):
    if kind == "rect-slidingwindow":
        full_args = dict(SLIDING_WINDOW_EMBEDDINGS_DEFAULT_KWARGS)
    elif kind == "triplets":
        full_args = {}
    else:
        raise NotImplementedError(kind)

    full_args.update(model_args)

    skip_args = ["prediction_batch_size"]
    name_parts = []
    name_parts += [f"{k}__{v}" for (k, v) in full_args.items() if k not in skip_args]
    name_parts += ["using", model_name]

    if model_name is None:
        raise Exception("`model_name` can't be None")

    return ".".join(name_parts)


class _EmbeddingModelMixin(object):
    model_name = luigi.Parameter()

    @property
    def model_path(self):
        model_filename = f"{self.model_name}.torch.pkl"
        fp_model = Path(self.data_path) / "embeddings" / "models" / model_filename
        return fp_model
