"""Paris 224x224 building segmentation dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


_VALID_EXTENSIONS: Sequence[str] = (".tif", ".tiff", ".geotiff")


class ParisBuildingSegmentation(Dataset):
    """Dataset wrapper for Paris building segmentation tiles.

    The dataset expects GeoTIFF tiles stored under ``optical`` and ``labels``
    directories. Tiles stored in ``optical`` contain 224x224 RGB imagery and the
    corresponding mask is stored inside ``labels`` with ``RGB`` replaced by
    ``MASK`` in the filename. Masks encode two classes: 0 for background and 1
    for buildings.
    """

    classes = ["background", "building"]
    splits = ("train", "val", "test")

    def __init__(
        self,
        root: str = "data/paris_224",
        split: Optional[str] = None,
        transforms: Optional[
            Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
        ] = None,
        files: Optional[Iterable[Path]] = None,
    ) -> None:
        self.root = Path(root)
        self.transforms = transforms

        if files is not None:
            self.files: List[Path] = sorted(Path(p) for p in files)
        else:
            search_root = self._resolve_split_directory(split)
            self.files = self._discover_optical_tiles(search_root)

        if not self.files:
            raise FileNotFoundError(
                f"No GeoTIFF tiles found for split={split!r} in {self.root}."
            )

    def _resolve_split_directory(self, split: Optional[str]) -> Path:
        if split is None:
            return self.root

        split_dir = self.root / split
        return split_dir if split_dir.exists() else self.root

    def _discover_optical_tiles(self, directory: Path) -> List[Path]:
        optical_root = directory / "optical"
        if not optical_root.exists():
            raise FileNotFoundError(
                f"Could not find 'optical' directory within {directory}."
            )

        files: List[Path] = []
        for extension in _VALID_EXTENSIONS:
            files.extend(sorted(optical_root.rglob(f"*{extension}")))
        return files

    def _mask_path(self, image_path: Path) -> Path:
        parts = list(image_path.parts)
        try:
            idx = parts.index("optical")
        except ValueError as exc:
            raise ValueError(
                "Image path is expected to contain an 'optical' directory"
            ) from exc

        mask_parts = parts.copy()
        mask_parts[idx] = "labels"
        mask_name = image_path.name.replace("RGB", "MASK")
        mask_parts[-1] = mask_name
        mask_path = Path(*mask_parts)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for image {image_path!s}.")
        return mask_path

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path = self.files[index]
        mask_path = self._mask_path(image_path)

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0)

        sample = {"image": image_tensor, "mask": mask_tensor}

        if self.transforms is not None:
            sample = self.transforms(sample)
            sample["mask"] = sample["mask"].squeeze().long()
        else:
            sample["mask"] = sample["mask"].squeeze().long()

        return sample

    def __repr__(self) -> str:
        split_info = "all"
        for split in self.splits:
            split_dir = self.root / split / "optical"
            if any(split_dir in file.parents for file in self.files):
                split_info = split
                break
        return (
            f"ParisBuildingSegmentation(root={self.root!s}, num_tiles={len(self.files)}, "
            f"split={split_info})"
        )


# Alias exposed for compatibility with the datamodule registration logic
ParisSegmentation = ParisBuildingSegmentation
