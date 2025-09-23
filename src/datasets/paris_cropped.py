"""Paris Cropped dataset utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


_VALID_EXTENSIONS: Sequence[str] = (".tif", ".tiff", ".geotiff")


class ParisCropped(Dataset):
    """Dataset for the Paris cropped GeoTIFF tiles."""

    classes = ["tile"]
    splits = ("train", "val", "test")

    def __init__(
        self,
        root: str = "data/paris_cropped",
        split: Optional[str] = "train",
        transforms: Optional[
            Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
        ] = None,
        files: Optional[Iterable[Path]] = None,
    ) -> None:
        """Initialize the dataset."""

        self.root = Path(root)
        self.transforms = transforms

        if files is not None:
            self.files: List[Path] = sorted(Path(p) for p in files)
        else:
            search_root = self._resolve_split_directory(split)
            self.files = self._discover_files(search_root)

        if not self.files:
            raise FileNotFoundError(
                f"No GeoTIFF tiles found for split={split!r} in {self.root}."
            )

    def _resolve_split_directory(self, split: Optional[str]) -> Path:
        if split is None:
            return self.root

        split_dir = self.root / split
        return split_dir if split_dir.exists() else self.root

    def _discover_files(self, directory: Path) -> List[Path]:
        files: List[Path] = []
        for extension in _VALID_EXTENSIONS:
            files.extend(sorted(directory.rglob(f"*{extension}")))
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.files[index]
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)

        tensor = torch.from_numpy(image)
        sample = {"image": tensor, "label": torch.tensor(0, dtype=torch.long)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __repr__(self) -> str:
        split_info = "unknown"
        for split in self.splits:
            if any((self.root / split) in file.parents for file in self.files):
                split_info = split
                break
        return (
            f"ParisCropped(root={self.root!s}, num_tiles={len(self.files)}, "
            f"split={split_info})"
        )
