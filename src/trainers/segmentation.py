"""Trainer for image segmentation."""

from typing import Any, Optional, Sequence

import torch
import numpy as np
import PIL
import matplotlib.pyplot as plt
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassJaccardIndex,
)
import torchgeo
import torchgeo.trainers

import src.models
import src.models_segmentation
import src.utils

src.utils.set_resources(num_threads=4)


class SegmentationTrainer(torchgeo.trainers.SemanticSegmentationTask):
    def __init__(
        self,
        segmentation_model,
        model,
        model_type="",
        weights=None,
        feature_map_indices=(5, 11, 17, 23),
        aux_loss_factor=0.5,
        input_size=224,
        patch_size=16,
        in_channels: int = 3,
        num_classes: int = 1000,
        num_filters: int = 64,
        loss: str = "ce",
        pretrained=True,
        input_res=10,
        adapter=False,
        adapter_trainable=True,
        adapter_shared=False,
        adapter_scale=1.0,
        adapter_type="lora",
        adapter_hidden_dim=16,
        norm_trainable=True,
        fixed_output_size=0,
        use_mask_token=False,
        train_patch_embed=False,
        patch_embed_adapter=False,
        patch_embed_adapter_scale=1.0,
        train_all_params=False,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        lr: float = 1e-3,
        patience: int = 10,
        train_cls_mask_tokens=False,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        callbacks=None,
        only_scaler_trainable=False,
        only_bias_trainable=False,
        class_names: Optional[Sequence[str]] = None,
        unet_bilinear: bool = True,
        unet_encoder_name: Optional[str] = None,
        unet_encoder_weights: Optional[str] = "imagenet",
        unet_encoder_depth: int = 5,
        unet_decoder_channels: Optional[Sequence[int]] = None,
        unet_decoder_use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self._class_names = (
            list(class_names) if class_names is not None else None
        )
        if hasattr(self, "hparams"):
            extra_hparams = {
                "unet_bilinear": unet_bilinear,
                "unet_encoder_name": unet_encoder_name,
                "unet_encoder_weights": unet_encoder_weights,
                "unet_encoder_depth": unet_encoder_depth,
                "unet_decoder_channels": unet_decoder_channels,
                "unet_decoder_use_batchnorm": unet_decoder_use_batchnorm,
            }
            for key, value in extra_hparams.items():
                try:
                    self.hparams[key] = value
                except (AttributeError, TypeError):
                    setattr(self.hparams, key, value)

    def configure_callbacks(self):
        return self.hparams["callbacks"]  # self.callbacks

    def configure_models(self):
        if self.hparams["segmentation_model"] == "unet":
            base_channels = self.hparams.get("num_filters", 64)
            encoder_name = self.hparams.get("unet_encoder_name")
            if not encoder_name:
                encoder_name = None
            decoder_channels = self.hparams.get("unet_decoder_channels")
            if decoder_channels is not None:
                decoder_channels = [int(ch) for ch in decoder_channels]
            self.model = src.models_segmentation.UNet(
                in_channels=self.hparams["in_channels"],
                num_classes=self.hparams["num_classes"],
                base_channels=base_channels,
                bilinear=self.hparams.get("unet_bilinear", True),
                encoder_name=encoder_name,
                encoder_weights=self.hparams.get("unet_encoder_weights", "imagenet"),
                encoder_depth=self.hparams.get("unet_encoder_depth", 5),
                decoder_channels=decoder_channels,
                decoder_use_batchnorm=self.hparams.get(
                    "unet_decoder_use_batchnorm", True
                ),
            )
            if self.hparams.get("freeze_backbone", False):
                self.model.freeze_encoder()
            return

        backbone = src.models.get_model(**self.hparams)

        # add segmentation head
        if self.hparams["segmentation_model"] == "fcn":
            self.model = src.models_segmentation.ViTWithFCNHead(
                backbone,
                num_classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "upernet":
            self.model = src.models_segmentation.UPerNetWrapper(
                backbone,
                self.hparams["feature_map_indices"],
                num_classes=self.hparams["num_classes"],
            )
        else:
            raise NotImplementedError(
                "`segmentation_model` must be one of ['fcn', 'upernet', 'unet'], "
                f"got {self.hparams['segmentation_model']}"
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"]
        ignore_index: Optional[int] = self.hparams["ignore_index"]
        per_class_metrics = self._build_per_class_iou_metrics(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        metric_dict: dict[str, Metric] = {
            "MulticlassAccuracy": MulticlassAccuracy(
                num_classes=num_classes,
                ignore_index=ignore_index,
                multidim_average="global",
                average="micro",
            ),
            "MulticlassJaccardIndex": MulticlassJaccardIndex(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average="micro",
            ),
            **per_class_metrics,
        }
        metrics = MetricCollection(metric_dict)
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.train_aux_metrics = metrics.clone(prefix="train_aux_")
        self.val_aux_metrics = metrics.clone(prefix="val_aux_")
        self.test_aux_metrics = metrics.clone(prefix="test_aux_")

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch["image"]
        y = batch["mask"]
        if self.model.deepsup:
            y_hat, y_aux = self(x)
            y_aux_hard = y_aux.argmax(dim=1)
            loss_aux = self.criterion(y_aux, y)
            self.log("train_aux_loss", loss_aux)
            self.train_aux_metrics(y_aux_hard, y)
            self.log_dict(self.train_aux_metrics)
        else:
            y_hat = self(x)

        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.log("train_loss", loss)
        self.train_metrics(y_hat_hard.squeeze(), y.squeeze())
        self.log_dict(self.train_metrics)

        if batch_idx % 100 == 0:
            imgs = self.PIL_imgs_from_batch(x)
            target_imgs = self.PIL_masks_from_batch(y.squeeze())
            pred_imgs = self.PIL_masks_from_batch(y_hat_hard.squeeze())
            self.logger.log_image(
                key="train_imgs",
                images=imgs,
            )
            self.logger.log_image(key="train_preds", images=pred_imgs)
            self.logger.log_image(key="train_targets", images=target_imgs)

        if self.model.deepsup:
            loss = loss + self.hparams["aux_loss_factor"] * loss_aux
        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        if self.model.deepsup:
            y_hat, y_aux = self(x)
            y_aux_hard = y_aux.argmax(dim=1)
            loss_aux = self.criterion(y_aux, y)
            self.log("val_aux_loss", loss_aux)
            self.val_aux_metrics(y_aux_hard, y)
            self.log_dict(self.val_aux_metrics)
        else:
            y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.log("val_loss", loss)
        self.val_metrics(y_hat_hard.squeeze(), y.squeeze())
        self.log_dict(self.val_metrics)

        if batch_idx % 100 == 0:
            imgs = self.PIL_imgs_from_batch(x)
            target_imgs = self.PIL_masks_from_batch(y.squeeze())
            pred_imgs = self.PIL_masks_from_batch(y_hat_hard.squeeze())
            self.logger.log_image(
                key="val_imgs",
                images=imgs,
            )
            self.logger.log_image(key="val_preds", images=pred_imgs)
            self.logger.log_image(key="val_targets", images=target_imgs)

        # log some figures
        if False:
            #  (
            #  batch_idx < 10
            #  and hasattr(self.trainer, "datamodule")
            #  and self.logger
            #  and hasattr(self.logger, "experiment")
            #  and hasattr(self.logger.experiment, "add_figure")
            # ):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = torchgeo.datasets.utils.unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    summary_writer.add_figure(
                        f"image/{batch_idx}", fig, global_step=self.global_step
                    )
                    plt.close()
            except ValueError:
                pass

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        if self.model.deepsup:
            y_hat, y_aux = self(x)
            y_aux_hard = y_aux.argmax(dim=1)
            loss_aux = self.criterion(y_aux, y)
            self.log("test_aux_loss", loss_aux)
            self.test_aux_metrics(y_aux_hard, y)
            self.log_dict(self.test_aux_metrics)
        else:
            y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.log("test_loss", loss)
        self.test_metrics(y_hat_hard.squeeze(), y.squeeze())
        self.log_dict(self.test_metrics)

        if False:  # batch_idx % 100 == 0:
            imgs = self.PIL_imgs_from_batch(x)
            target_imgs = self.PIL_masks_from_batch(y.squeeze())
            pred_imgs = self.PIL_masks_from_batch(y_hat_hard.squeeze())
            self.logger.log_image(
                key="test_imgs",
                images=imgs,
            )
            self.logger.log_image(key="test_preds", images=pred_imgs)
            self.logger.log_image(key="test_targets", images=target_imgs)

        if self.model.deepsup:
            loss = loss + self.hparams["aux_loss_factor"] * loss_aux
        return loss

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        if self.model.deepsup:
            y_hat, _ = self(x)
            y_hat = y_hat.softmax(dim=q)
        else:
            y_hat: torch.Tensor = self(x).softmax(dim=1)
        return y_hat

    def PIL_imgs_from_batch(self, x, n=4):
        """return list of PIL images from tensor input images"""
        imgs = []
        for img in x[:n]:
            img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
            # assert img.shape[-1] == 3
            if img.shape[-1] not in [3, 1]:
                img = img[:, :, [3, 2, 1]]  # S2 RGB
            # img = img.detach().cpu().numpy()
            img /= img.max(axis=(0, 1))
            img *= 255
            img = np.clip(img, 0, 255).astype(np.uint8)
            imgs.append(PIL.Image.fromarray(img))

        return imgs

    def _generate_default_palette(self, num_colors: int) -> list[int]:
        num_colors = max(1, num_colors)
        if num_colors == 1:
            return [0, 0, 0]

        palette: list[int] = []
        for idx in range(num_colors):
            value = int(round(255 * idx / (num_colors - 1)))
            palette.extend([value, value, value])
        return palette

    def _get_mask_palette(self) -> list[int]:
        num_classes = int(self.hparams["num_classes"])

        palette_config = None
        try:
            palette_config = self.hparams["mask_palette"]
        except (KeyError, TypeError):
            palette_config = getattr(self.hparams, "mask_palette", None)

        user_palette: list[int] = []
        if palette_config is not None:
            if isinstance(palette_config, np.ndarray):
                palette_values = palette_config.reshape(-1).tolist()
            else:
                palette_values = []
                for value in palette_config:
                    if isinstance(value, (list, tuple, np.ndarray)):
                        palette_values.extend(np.asarray(value).reshape(-1).tolist())
                    else:
                        palette_values.append(value)
            user_palette = [
                int(max(0, min(255, round(float(v))))) for v in palette_values
            ]
            if len(user_palette) % 3 != 0:
                user_palette.extend([0] * (3 - len(user_palette) % 3))

        user_color_count = len(user_palette) // 3
        color_count = min(max(num_classes, user_color_count), 256)
        color_count = max(color_count, 1)
        palette = self._generate_default_palette(color_count)

        if user_palette:
            limit = min(user_color_count, color_count)
            palette[: limit * 3] = user_palette[: limit * 3]

        if len(palette) < 256 * 3:
            palette.extend([0] * (256 * 3 - len(palette)))
        else:
            palette = palette[: 256 * 3]

        return palette

    def PIL_masks_from_batch(self, x, n=4):
        """return list of PIL images from tensor input images"""
        imgs = []
        palette = self._get_mask_palette()
        for img in x[:n]:
            # img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
            img = img.detach().cpu().numpy()
            img = np.squeeze(img)
            assert img.ndim == 2, f"{img.shape=}"
            assert img.min() >= 0
            assert img.max() <= 255
            img = img.astype(np.uint8)
            pil_img = PIL.Image.fromarray(img, mode="P")
            pil_img.putpalette(palette)
            imgs.append(pil_img)

        return imgs

    def _build_per_class_iou_metrics(
        self,
        num_classes: int,
        ignore_index: Optional[int],
    ) -> dict[str, Metric]:
        class_names = self._resolve_class_names(num_classes)
        metrics: dict[str, Metric] = {}
        used_names: set[str] = set()
        for class_index, class_name in enumerate(class_names):
            if ignore_index is not None and class_index == ignore_index:
                continue
            metric_name = self._sanitize_class_metric_name(
                class_name, class_index, used_names
            )
            metrics[metric_name] = _ClasswiseJaccardIndex(
                num_classes=num_classes,
                ignore_index=ignore_index,
                class_index=class_index,
            )
        return metrics

    def _resolve_class_names(self, num_classes: int) -> list[str]:
        if self._class_names is not None:
            class_names: list[str] = [str(name) for name in self._class_names]
        else:
            class_names = []
            datamodule = getattr(self.trainer, "datamodule", None)
            dataset = getattr(datamodule, "train_dataset", None) if datamodule else None
            if dataset is not None and hasattr(dataset, "classes"):
                class_names = [str(name) for name in dataset.classes]
        if not class_names:
            class_names = [str(index) for index in range(num_classes)]
        if len(class_names) < num_classes:
            class_names.extend(
                str(index) for index in range(len(class_names), num_classes)
            )
        elif len(class_names) > num_classes:
            class_names = class_names[:num_classes]
        return class_names

    def _sanitize_class_metric_name(
        self, class_name: str, class_index: int, used_names: set[str]
    ) -> str:
        base = "".join(
            ch.lower() if ch.isalnum() else "_"
            for ch in class_name.strip()
        ).strip("_")
        if not base:
            base = f"class_{class_index}"
        candidate = f"{base}_iou"
        suffix = 1
        while candidate in used_names:
            candidate = f"{base}_{suffix}_iou"
            suffix += 1
        used_names.add(candidate)
        return candidate


class _ClasswiseJaccardIndex(MulticlassJaccardIndex):
    """Compute IoU for a single class index using multiclass Jaccard."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int],
        class_index: int,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average=None,
        )
        self.class_index = class_index

    def compute(self):  # type: ignore[override]
        values = super().compute()
        if values.ndim == 0:
            return values
        return values[self.class_index]
