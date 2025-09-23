"""Lightning DataModule for the Paris 224x224 segmentation dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ..datasets.paris_224 import ParisBuildingSegmentation


class ParisSegmentationDataModule(pl.LightningDataModule):
    """Provide dataloaders for Paris building segmentation tiles."""

    mean = torch.tensor([0.2563, 0.2543, 0.2532], dtype=torch.float32)
    std = torch.tensor([0.2563, 0.2543, 0.2532], dtype=torch.float32)

    def __init__(
        self,
        root: str = "data/paris_224",
        batch_size: int = 32,
        num_workers: int = 0,
        transforms: Optional[torch.nn.Module] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = root
        self.train_batch_size = batch_size
        self.eval_batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.train_augmentations = A.Compose(
            [
                A.RandomResizedCrop(
                    height=224,
                    width=224,
                    scale=(0.75, 1.0),
                    ratio=(0.8, 1.25),
                    p=0.5,
                ),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=30,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.75,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1.0),
                        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
                    ],
                    p=0.2,
                ),
            ]
        )

    def prepare_data(self) -> None:  # noqa: D401 - no-op hook
        """Placeholder to comply with LightningDataModule API."""

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D401
        """Create datasets for each split."""

        root_path = Path(self.root)
        splits_available = all(
            (root_path / split / "optical").exists()
            for split in ParisBuildingSegmentation.splits
        )

        if splits_available:
            self.train_dataset = ParisBuildingSegmentation(
                self.root,
                split="train",
                transforms=self.transforms,
                augmentations=self.train_augmentations,
            )
            self.val_dataset = ParisBuildingSegmentation(
                self.root, split="val", transforms=self.transforms
            )
            self.test_dataset = ParisBuildingSegmentation(
                self.root, split="test", transforms=self.transforms
            )
        else:
            full_dataset = ParisBuildingSegmentation(
                self.root, split=None, transforms=self.transforms
            )
            lengths = self._split_lengths(len(full_dataset))
            generator = torch.Generator().manual_seed(self.seed)
            permutation = torch.randperm(len(full_dataset), generator=generator)

            train_end = lengths[0]
            val_end = train_end + lengths[1]

            train_files = [full_dataset.files[i] for i in permutation[:train_end]]
            val_files = [full_dataset.files[i] for i in permutation[train_end:val_end]]
            test_files = [full_dataset.files[i] for i in permutation[val_end:]]

            self.train_dataset = ParisBuildingSegmentation(
                self.root,
                split=None,
                transforms=self.transforms,
                augmentations=self.train_augmentations,
                files=train_files,
            )
            self.val_dataset = ParisBuildingSegmentation(
                self.root,
                split=None,
                transforms=self.transforms,
                files=val_files,
            )
            self.test_dataset = ParisBuildingSegmentation(
                self.root,
                split=None,
                transforms=self.transforms,
                files=test_files,
            )

            for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
                dataset.classes = full_dataset.classes

        self.drop_last = stage == "fit"

    def _split_lengths(self, dataset_size: int) -> list[int]:
        test = int(dataset_size * self.test_split)
        val = int(dataset_size * self.val_split)
        train = dataset_size - val - test
        if train <= 0:
            raise ValueError(
                "ParisSegmentationDataModule requires more samples than reserved for"
                " validation and test splits."
            )
        return [train, val, dataset_size - train - val]

    def train_dataloader(self) -> DataLoader[Dataset]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader[Dataset]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader[Dataset]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
