"""
Contains the main triplet-trainer architecture (Tile2Vec) and the datamodule to
load triplet-datasets (TripletTrainerDataModule)
"""
import pathlib

import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

from .data.dataset import ImageTripletDataset
from .data.transforms import get_transforms
from .external.nn_layers import AdaptiveConcatPool2d
from . import backbones


class HeadFineTuner(pl.callbacks.BaseFinetuning):
    """
    Freezes the backbone during training.
    """

    def __init__(self, train_bn: bool = True):
        super().__init__()
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        self.freeze(modules=[pl_module.backbone], train_bn=self.train_bn)

    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
    ):
        # here we could unfreeze at a specific epoch if we wanted to for
        # example, but we'll just keep the backbone frozen for now
        pass


class Tile2Vec(pl.LightningModule):
    def __init__(
        self,
        base_arch="resnet18",
        pretrained=False,
        margin=1.0,
        lr=1.0e-5,
        l2_regularisation=None,
        n_input_channels=3,
        n_embedding_dims=100,
    ):
        super().__init__()

        if l2_regularisation not in [None, "unknown"]:
            raise NotImplementedError()

        self.lr = lr
        self.margin = margin
        self.l2_regularisation = l2_regularisation
        self.n_embedding_dims = n_embedding_dims
        self.n_input_channels = n_input_channels
        self.base_arch = base_arch
        self.pretrained = pretrained
        self.save_hyperparameters()
        self.__build_model()

    def __build_model(self):
        if self.hparams.base_arch == "unknown":
            # special value allowing loading of weights directly to produce an encoder network
            pass
        else:
            self.backbone, n_features_backbone = self._create_backbone_layers(
                n_input_channels=self.hparams.n_input_channels,
                base_arch=self.hparams.base_arch,
                pretrained=self.hparams.pretrained,
            )

            self.head = self._create_head_layers(
                n_features_backbone=n_features_backbone
            )

    def _create_head_layers(self, n_features_backbone, head_type="orig_fastai"):

        if head_type == "linear":
            # make "head" block which takes features of the encoder (resnet18 has
            # 512), uses adaptive pooling to reduce the x- and y-dimension, and
            # then uses a fully-connected layer to make the desired output size
            head = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),  # -> (batch_size, n_features, 1, 1)
                torch.nn.Flatten(),  # -> (batch_size, n_features)
                torch.nn.Linear(
                    in_features=n_features_backbone, out_features=self.n_embedding_dims
                ),  # -> (batch_size, n_embedding_dims)
            )
        elif head_type == "orig_fastai":
            # make "head" block which takes features of the encoder (resnet18 has
            # 512), uses adaptive pooling to reduce the x- and y-dimension, and
            # then uses two blocks of batchnorm, dropout and linear layers with
            # a ReLU activation layer inbetween to finally make the output the
            # size of the embedding dimensions requested. This architecture
            # replicates the original architecture generated with fastai and
            # used in L. Denby (2020) GRL

            n_intermediate_layer_features = 512
            head = torch.nn.Sequential(
                AdaptiveConcatPool2d(1),  # -> (batch_size, 2*n_features, 1, 1)
                nn.Flatten(),  # -> (batch_size, 2*n_features)
                # first batchnorm, dropout, linear block (only the linear block
                # affects the shape)
                nn.BatchNorm1d(
                    num_features=2 * n_features_backbone, eps=1.0e-5, momentum=0.1
                ),
                nn.Dropout(p=0.25),
                nn.Linear(  # -> (batch_size, n_intermediate_layer_features)
                    in_features=2 * n_features_backbone,
                    out_features=n_intermediate_layer_features,
                ),
                # ReLU activation
                nn.ReLU(inplace=True),
                # second batchnorm, dropout, linear block
                nn.BatchNorm1d(
                    num_features=n_intermediate_layer_features, eps=1.0e-5, momentum=0.1
                ),
                nn.Dropout(p=0.5),
                nn.Linear(
                    in_features=n_intermediate_layer_features,
                    out_features=self.n_embedding_dims,
                ),  # -> (batch_size, n_embedding_dims)
            )
        else:
            raise NotImplementedError(head_type)

        return head

    def _create_backbone_layers(
        self, n_input_channels, base_arch=None, pretrained=False
    ):
        backbone, n_features_backbone = backbones.backbone_and_num_features(
            model_name=base_arch, pretrained=pretrained
        )

        # We need to ensure the number of input channels matches the number of
        # channels in our data. If it doesn't we replace that layer here
        # (loosing the pretrained weights!)
        # NOTE: at least for the resnet models the convention is that the first
        # (the input) layer is `.conv1`.
        layers = list(backbone.children())
        if not isinstance(layers[0], torch.nn.Conv2d):
            raise Exception(f"Recognised type for input layer {layers[0]}")

        input_conv_orig = layers[0]

        if input_conv_orig.in_channels != n_input_channels:
            layers[0] = torch.nn.Conv2d(
                n_input_channels,
                input_conv_orig.out_channels,
                kernel_size=input_conv_orig.kernel_size,
                stride=input_conv_orig.stride,
                padding=input_conv_orig.padding,
                bias=input_conv_orig.bias,
            )

            # make a new backbone with the right number of input layers
            backbone = torch.nn.Sequential(*layers)

        return backbone, n_features_backbone

    def encoder(self, x):
        # when working with a pretrained model we've frozen the backbone layers
        # and so we don't want to calculate gradients through this layer
        if self.pretrained:
            with torch.no_grad():
                z = self.backbone(x)
        else:
            z = self.backbone(x)
        return self.head(z)

    def _loss(self, batch):
        y_anchor, y_neighbour, y_distant = [self.encoder(x) for x in batch]

        # per-batch distance between anchor and neighbour tile in embedding
        # space, shape: (batch_size, )
        l_near = ((y_anchor - y_neighbour) ** 2.0).sum(dim=1)
        # per-batch distance between anchor and distant tile in embedding
        # space, shape: (batch_size, )
        l_distant = ((y_anchor - y_distant) ** 2.0).sum(dim=1)

        # total loss goes through a ReLU with the margin added, and we take
        # mean across the batch
        loss = torch.mean(F.relu(l_near - l_distant + self.margin))
        self.log("loss", loss)
        self.log("l_near_mean", torch.mean(l_near.detach()))
        self.log("l_distant_mean", torch.mean(l_distant.detach()))

        return loss

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch=batch, batch_idx=batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--base-arch", type=str, default="resnet18", help="Backbone architecture"
        )
        parser.add_argument(
            "--pretrained",
            type=bool,
            default=True,
            help="Use a pretrained backbone, only 'head' layers are trained",
        )
        parser.add_argument(
            "--margin", type=float, default=1.0, help="margin to distant tile"
        )
        parser.add_argument("--lr", type=float, default=1.0e-5, help="learning rate")
        return parser


class TripletTrainerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        normalize_for_arch=None,
        train_val_fraction=0.9,
        batch_size=32,
        num_dataloader_workers=0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_val_fraction = train_val_fraction
        self.batch_size = batch_size
        self._train_dataset = None
        self._test_dataset = None
        self.num_dataloader_workers = num_dataloader_workers

        self._train_transforms = get_transforms(
            step="train", normalize_for_arch=normalize_for_arch
        )
        self._predict_transforms = get_transforms(
            step="predict", normalize_for_arch=normalize_for_arch
        )

    def get_dataset(self, stage):
        if stage == "fit":
            return ImageTripletDataset(
                data_dir=self.data_dir, stage="train", transform=self._train_transforms
            )
        elif stage == "predict":
            return ImageTripletDataset(
                data_dir=self.data_dir,
                stage="study",
                transform=self._predict_transforms,
            )
        else:
            raise NotImplementedError(stage)

    def setup(self, stage=None):
        if stage == "fit":
            full_dataset = self.get_dataset(stage=stage)
            n_samples = len(full_dataset)
            n_train = int(n_samples * self.train_val_fraction)
            n_valid = n_samples - n_train
            self._train_dataset, self._val_dataset = random_split(
                full_dataset, [n_train, n_valid]
            )
        else:
            raise NotImplementedError(stage)

    def train_dataloader(self):
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self._val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("data_dir", type=pathlib.Path)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--train-valid-fraction", type=float, default=0.9)
        parser.add_argument("--num-dataloader-workers", type=int, default=0)
        return parser
