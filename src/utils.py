"""Collection of utility methods for model training and evaluation."""

import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

from collections import defaultdict


import kornia
import lightning.pytorch.callbacks
import numpy as np
import torch
import torchgeo.transforms
import yaml
import mlflow
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.tracking import MlflowClient
from PIL import Image

import src.datamodules


def update_configs(config: dict) -> dict:
    """Creates a new config dict without Dotdict entries.

    Args:
        config: a dict that might contain Dotdict type entries.

    Returns:
        a new dictionary where Dotdicts objects are resolved into dicts.

    """

    updated_configs = {}
    for k, v in config.__dict__.items():
        if isinstance(v, Dotdict):
            updated_configs[k] = v.__dict__
        else:
            updated_configs[k] = v

    return updated_configs


def set_resources(num_threads: int):
    """Sets environment variables to control resource usage.

    The environment variables control the number of used threads
    for different vector op libraries and GDAL.

    Args.
        num_threads: the max number of threads.

    """

    num_threads = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    os.environ["NUMEXPR_NUM_THREADS"] = num_threads
    os.environ["GDAL_NUM_THREADS"] = num_threads


def set_seed(seed: int):
    """Set the seed across multiple libraries.

    Sets seed for builtin random, numpy, and torch libraries.

    Args:
        seed: the seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Dotdict:
    """Wraps dictionaries to allow value access in dot notation.

    Instead of data[key], access value as data.key"""

    def __init__(self, data: dict):
        super().__init__()
        for k, v in data.items():
            if isinstance(v, dict):
                # take care of nested dicts
                v = Dotdict(v)
            self.__dict__[k] = v


class LocalMLFlowLogger(MLFlowLogger):
    """Lightning MLflow logger with helpers for local artifact management."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image_counters = defaultdict(int)


    @property
    def run_dir(self) -> str:
        """Return the local artifact directory for the active MLflow run."""

        run_info = self.experiment.get_run(self.run_id)
        return _artifact_uri_to_path(run_info.info.artifact_uri)

    def log_image(self, key: str, images, step: int = None, caption=None) -> None:
        """Log PIL images as MLflow artifacts."""

        if not isinstance(images, (list, tuple)):
            images = [images]

        artifact_subdir = os.path.join("images", key)
        if step is None:
            step = self._image_counters[key]
            self._image_counters[key] += 1

        for idx, image in enumerate(images):
            if not isinstance(image, Image.Image):
                raise TypeError("Expected PIL.Image.Image instances for logging")

            filename = f"{key}-{step}-{idx}.png"


            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, filename)
                image.save(path)
                self.experiment.log_artifact(
                    self.run_id,
                    path,
                    artifact_path=artifact_subdir,
                )


def _artifact_uri_to_path(artifact_uri: str) -> str:
    """Convert an MLflow artifact URI into a filesystem path."""

    parsed = urlparse(artifact_uri)
    if parsed.scheme not in ("", "file"):
        raise ValueError(f"Unsupported artifact URI scheme: {parsed.scheme}")

    if parsed.scheme == "file":
        path = parsed.path
        if not path:
            path = artifact_uri.split("file:", 1)[1]
    else:
        path = artifact_uri

    if parsed.netloc:
        path = os.path.join(os.sep + parsed.netloc, path.lstrip("/"))

    return os.path.abspath(path)


def _resolve_tracking_uri(raw_uri: str) -> Tuple[str, str]:
    """Resolve a tracking URI to MLflow's expected format and ensure the directory exists."""

    parsed = urlparse(raw_uri)
    if parsed.scheme not in ("", "file"):
        raise ValueError("Only local file-based MLflow tracking URIs are supported")

    if parsed.scheme == "":
        path = os.path.abspath(parsed.path)
        uri = Path(path).as_uri()
    else:
        path = parsed.path or raw_uri.split("file:", 1)[1]
        path = os.path.abspath(path)
        uri = Path(path).as_uri()

    os.makedirs(path, exist_ok=True)
    return uri, path


def setup_mlflow(
    config: Dotdict,
) -> Tuple[mlflow.ActiveRun, LocalMLFlowLogger, Dotdict]:
    """Configure MLflow tracking for a training or evaluation run."""

    tracking_uri, tracking_path = _resolve_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(config.mlflow.experiment)

    run_kwargs = {}
    if getattr(config.mlflow, "run_name", None):
        run_kwargs["run_name"] = config.mlflow.run_name

    run = mlflow.start_run(**run_kwargs)
    mlflow_logger = LocalMLFlowLogger(
        experiment_name=experiment.name,
        tracking_uri=tracking_uri,
        run_id=run.info.run_id,
        log_model=config.mlflow.log_model,
    )

    run_dir = mlflow_logger.run_dir
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config.mlflow.tracking_uri = tracking_uri
    config.mlflow.tracking_path = tracking_path
    config.mlflow.experiment_id = experiment.experiment_id
    config.mlflow.run_id = run.info.run_id
    config.mlflow.run_dir = run_dir
    config.mlflow.checkpoint_dir = checkpoint_dir

    resolved_config = update_configs(config)
    mlflow.log_dict(resolved_config, "updated_setup_configs.yml")

    if config.verbose:
        print(run_dir)
    with open(os.path.join(run_dir, "updated_setup_configs.yml"), "w") as outfile:
        yaml.dump(resolved_config, outfile, default_flow_style=False)

    return run, mlflow_logger, config


def get_datamodule(
    config: Dotdict,
) -> Tuple[lightning.pytorch.LightningDataModule, Dotdict]:
    """Creates the lightning datamodule for the dataset defined in the config.

    Args:
        config: the training run config.

    Returns:
        a tuple of the lightning datamodule for a dataset and the latest config.
    """

    # get the correct datamodule and dataset objects
    datamodule = src.datamodules.__dict__[config.data.datamodule]
    dataset = src.datasets.__dict__[config.data.datamodule.replace("DataModule", "")]

    # scale images to expected size and standardize
    if (
        "benge" in config.data.datamodule.lower()
        or "treesatai" in config.data.datamodule.lower()
    ):
        if config.data.modality == "s1":
            datamodule.mean = datamodule.s1_mean
            datamodule.std = datamodule.s1_std
        elif config.data.modality == "s2":
            datamodule.mean = datamodule.s2_mean
            datamodule.std = datamodule.s2_std
        elif config.data.modality == "aerial":
            datamodule.mean = datamodule.aerial_mean
            datamodule.std = datamodule.aerial_std
        else:
            raise AttributeError()

    # ensure bands are correctly selected for multi-spectral data
    band_idx = range(len(datamodule.mean))
    if "eurosat" in config.data.datamodule.lower():
        band_idx = []
        for b in config.data.bands:
            band_idx.append(dataset.all_band_names.index(b))
        if config.verbose:
            print(f"Band indices: {band_idx=}")

    data_keys = ["image"]
    if config.task == "segmentation":
        data_keys.append("mask")

    if config.verbose:
        print(f"Augmentation keys: {data_keys=}")

    additional_transforms = torchgeo.transforms.AugmentationSequential(
        kornia.augmentation.Normalize(
            mean=datamodule.mean[band_idx],
            std=datamodule.std[band_idx],
            keepdim=True,
        ),
        kornia.augmentation.Resize(
            (config.data.img_size, config.data.img_size), keepdim=True
        ),
        data_keys=data_keys,
    )

    root = config.data.root
    if config.verbose:
        print(f"Dataset root directory: {root=}")

    # handle directory structure of different datasets
    if hasattr(src.datamodules.__dict__[config.data.datamodule], "folder"):
        root = f"data/{src.datamodules.__dict__[config.data.datamodule].folder}"

    # initialze few-shot datamodule with limited number of samples
    if config.data.few_shot_k is not None:
        if "eurosat" in config.data.datamodule.lower():
            datamodule = datamodule(
                root=root,
                bands=config.data.bands,
                batch_size=config.optim.batch_size,
                num_workers=config.optim.num_workers,
                train_split_file_suffix=f"-k{config.data.few_shot_k}-seed{config.data.few_shot_seed}.txt",
                transforms=additional_transforms,
            )
        elif "treesatai" in config.data.datamodule.lower():
            # no few-shot split defined yet for TreeSatAI
            raise NotImplementedError()
        else:
            datamodule = datamodule(
                root=root,
                batch_size=config.optim.batch_size,
                num_workers=config.optim.num_workers,
                train_split_file_suffix=f"-k{config.data.few_shot_k}-seed{config.data.few_shot_seed}.txt",
                transforms=additional_transforms,
            )
    else:
        # initialze the full dataset
        if "eurosat" in config.data.datamodule.lower():
            datamodule = datamodule(
                root=root,
                bands=config.data.bands,
                batch_size=config.optim.batch_size,
                num_workers=config.optim.num_workers,
                transforms=additional_transforms,
            )
        elif "treesatai" in config.data.datamodule.lower():
            datamodule = datamodule(
                root=root,
                modality=config.data.modality,
                bands=config.data.bands,
                batch_size=config.optim.batch_size,
                num_workers=config.optim.num_workers,
                transforms=additional_transforms,
                size=config.data.size,
            )
        else:
            datamodule = datamodule(
                root=root,
                batch_size=config.optim.batch_size,
                num_workers=config.optim.num_workers,
                transforms=additional_transforms,
            )

    datamodule.setup("fit")
    config.data.num_classes = len(datamodule.train_dataset.classes)
    config.data.in_chans = datamodule.train_dataset[0]["image"].shape[0]

    return datamodule, config


def get_callbacks(
    dir: str,
) -> Tuple[
    lightning.pytorch.callbacks.ModelCheckpoint,
    lightning.pytorch.callbacks.EarlyStopping,
    lightning.pytorch.callbacks.LearningRateMonitor,
]:
    """Initialze lightning callbacks for checkpointing, early stopping and LR monitoring.

    Args:
        dir: a directory where model checkpoints will be stored.

    Returns:
        a tuple of the three callback objects.
    """

    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        # dirpath=args.experiment_dir,
        dirpath=dir,
        save_top_k=1,  # save best
        save_last=True,
    )
    early_stopping_callback = lightning.pytorch.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
    )

    lr_monitor = lightning.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step"
    )
    return checkpoint_callback, early_stopping_callback, lr_monitor


def _get_mlflow_client(tracking_uri: str = None) -> MlflowClient:
    """Return an MLflow client for the configured tracking URI."""

    if tracking_uri:
        return MlflowClient(tracking_uri=tracking_uri)
    return MlflowClient(tracking_uri=mlflow.get_tracking_uri())


def _get_artifact_cache_dir(run_id: str) -> str:
    """Return (and create) a local cache directory for downloaded MLflow artifacts."""

    cache_dir = os.path.join("logs", "mlflow_artifacts", run_id)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def log_checkpoints_to_mlflow(
    checkpoint_callback: lightning.pytorch.callbacks.ModelCheckpoint,
    artifact_path: str = "checkpoints",
) -> None:
    """Upload best and last checkpoints saved by a callback to MLflow."""

    if checkpoint_callback is None:
        return

    checkpoints = {}
    best_path = getattr(checkpoint_callback, "best_model_path", None)
    if best_path and os.path.isfile(best_path):
        checkpoints["best.ckpt"] = best_path

    last_path = getattr(checkpoint_callback, "last_model_path", None)
    if last_path and os.path.isfile(last_path):
        checkpoints["last.ckpt"] = last_path

    for filename, path in checkpoints.items():
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = os.path.join(tmpdir, filename)
            shutil.copy2(path, target_path)
            mlflow.log_artifact(target_path, artifact_path=artifact_path)


def get_ckpt_path_from_mlflow_run(
    config: Dotdict, run_id_key: str = "continual_pretrain_run", state: str = "best"
) -> str:
    """Return the local path to a checkpoint artifact stored in MLflow."""

    run_id = getattr(config, run_id_key)
    if run_id is None:
        raise ValueError(f"Config does not define {run_id_key}")

    artifact_file = "best.ckpt" if state == "best" else "last.ckpt"
    if state not in {"best", "last"}:
        raise ValueError(f"{state} not in [best, last]")

    client = _get_mlflow_client(
        getattr(getattr(config, "mlflow", Dotdict({})), "tracking_uri", None)
    )
    cache_dir = _get_artifact_cache_dir(run_id)
    local_path = client.download_artifacts(
        run_id,
        os.path.join("checkpoints", artifact_file),
        cache_dir,
    )
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Checkpoint {artifact_file} not found for run {run_id}")
    return local_path


def assert_model_compatibility(
    pretrain_config: Dotdict, downstream_config: Dotdict, ignore: list = []
):
    """Performs some checks to ensure pre-training run and downstream tasks are compatible.

    Args:
        pretrain_config: config of the pretraining run.
        downstream_config: config of the downstream task.
        ignore: list of checks to be skipped.

    Returns:
        True if all checks passed.

    Raises:
        AssertionError if any check fails.
    """

    if not "model" in ignore:
        assert (
            pretrain_config["model"] == downstream_config.model.name
        ), f"{pretrain_config['model']=}, {downstream_config.model.name=}"
    assert pretrain_config["in_channels"] == downstream_config.data.in_chans
    if not "embed_dim" in ignore:
        assert pretrain_config["embed_dim"] == downstream_config.model.embed_dim
    assert pretrain_config["input_size"] == downstream_config.data.img_size
    assert pretrain_config["patch_size"] == downstream_config.model.patch_size
    assert pretrain_config["adapter"] == downstream_config.model.adapter
    assert (
        pretrain_config["adapter_type"] == downstream_config.model.adapter_type
    ), f"{pretrain_config['adapter_type']=}, {downstream_config.model.adapter_type=}"
    assert pretrain_config["adapter_shared"] == downstream_config.model.adapter_shared
    assert pretrain_config["adapter_scale"] == downstream_config.model.adapter_scale
    assert (
        pretrain_config["adapter_hidden_dim"]
        == downstream_config.model.adapter_hidden_dim
    ), f"{pretrain_config['adapter_hidden_dim']=}, {downstream_config.model.adapter_hidden_dim=}"
    assert (
        pretrain_config["patch_embed_adapter"]
        == downstream_config.model.patch_embed_adapter
    ), f"{pretrain_config['patch_embed_adapter']=}, {downstream_config.model.patch_embed_adapter=}"
    assert (
        pretrain_config["patch_embed_adapter_scale"]
        == downstream_config.model.patch_embed_adapter_scale
    )

    return True


def get_config_from_mlflow_run(
    config: Dotdict,
    run_id_key: str = "continual_pretrain_run",
    return_ckpt_path: bool = False,
) -> Dotdict:
    """Get the configuration dictionary stored with a finished MLflow run."""

    run_id = getattr(config, run_id_key)
    if run_id is None:
        raise ValueError(f"Config does not define {run_id_key}")

    client = _get_mlflow_client(
        getattr(getattr(config, "mlflow", Dotdict({})), "tracking_uri", None)
    )
    cache_dir = _get_artifact_cache_dir(run_id)
    config_path = client.download_artifacts(
        run_id,
        "updated_setup_configs.yml",
        cache_dir,
    )

    with open(config_path, "r") as fh:
        args = yaml.safe_load(fh)

    if return_ckpt_path:
        ckpt_path = get_ckpt_path_from_mlflow_run(
            config, run_id_key=run_id_key
        )
        return args, ckpt_path
    return args


def load_weights_from_mlflow_run(
    model: torch.nn.Module,
    config: Dotdict,
    prefix: str = None,
    run_id_key: str = "continual_pretrain_run",
    return_ckpt: bool = False,
    which_state: str = "best",
):
    """Load weights from a finished MLflow run into a model object.

    Args:
        model: the torch model.
        config: the config of the finished run.
        prefix: prefix in model layer names that wasn't present in the pre-training run.
        run_id_key: type of the pre-training run.
        return_ckpt: if the checkpoint is returned as well.
        which_state: best or latest checkpoint will be used.

    Returns:
        the model initialzed with weights from ´config´, or a tuple with the checkpoint.
    """

    best_ckpt = get_ckpt_path_from_mlflow_run(
        config,
        run_id_key=run_id_key,
        state=which_state,
    )
    print(f"Loading checkpoint {best_ckpt=}...")

    ckpt = torch.load(best_ckpt)
    state = ckpt["state_dict"]

    # remove prefix from state dict keys
    for k in list(state.keys()):
        state[k.replace("model.", "")] = state[k]
        del state[k]

    if "cls_token" in model.state_dict() and not "cls_token" in state.keys():
        state["cls_token"] = model.state_dict()["cls_token"]

    if "pos_embed" in model.state_dict() and not "pos_embed" in state.keys():
        state["pos_embed"] = model.state_dict()["pos_embed"]

    if prefix is not None:
        for k in list(state.keys()):
            state[prefix + k] = state[k]
            del state[k]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Missing weights in pre-training: {missing=}")
    print(
        f"Unexpected weights (except decoder): {[k for k in unexpected if not 'decoder' in k]}"
    )

    if return_ckpt:
        return model, ckpt

    return model


def assert_run_validity(row: dict, config: Dotdict, idx: int) -> bool:
    """Checks if a config defines a valid run (no invalid combination of args).

    Args:
        row: the row in the run_id file that is being checked.
        config: the training run config.
        idx: the index of the current run in the run_id file.

    Returns:
        True if the run is valid.

    Raises:
        ValueError if there is an invalid combination of configurations.
    """

    if row["mode"] == "lin_eval":
        assert (
            config.model.adapter == False
        ), f"{idx=}, {row.run_id=}, {config.model.adapter=}"
        # assert (
        #     config.model.patch_embed_adapter == False
        # ), f"{idx=}, {row.run_id=}, {config.model.patch_embed_adapter=}"
        if hasattr(config.model, "norm_trainable"):
            assert (
                config.model.norm_trainable == False
            ), f"{idx=}, {row.run_id=}, {config.norm_trainable=}"
    elif row["mode"] == "lin_eval_slr":
        assert config.model.adapter, f"{idx=}, {row.run_id=}, {config.model.adapter=}"
        assert (
            config.model.adapter_trainable == False
        ), f"{idx=}, {row.run_id=}, {config.model.adapter_trainable=}"
        if hasattr(config.model, "norm_trainable"):
            assert (
                config.model.norm_trainable == False
            ), f"{idx=}, {row.run_id=}, {config.model.norm_trainable=}"
    elif row["mode"] == "slr_ft":
        assert config.continual_pretrain_run is not None
        assert config.model.adapter, f"{idx=}, {row.run_id=}, {config.model.adapter=}"
        assert (
            config.model.adapter_trainable
        ), f"{idx=}, {row.run_id=}, {config.model.adapter_trainable=}"
        assert (
            config.model.norm_trainable
        ), f"{idx=}, {row.run_id=}, {config.model.norm_trainable=}"
    elif row["mode"] == "ft":
        assert config.continual_pretrain_run is None
        assert not config.model.adapter
        assert config.model.train_all_params
    elif row["mode"] == "slr_scale":
        assert config.continual_pretrain_run is not None
        assert config.model.adapter
        assert not config.model.adapter_trainable
        assert config.model.only_scaler_trainable
    elif row["mode"] == "slr_full_ft":
        assert config.model.train_all_params
        assert config.continual_pretrain_run is not None
        assert config.model.adapter
    else:
        raise ValueError(f"{row['mode']=}, {row.run_id=}")

    if row.dataset == "eurosat":
        assert (
            config.data.datamodule == "EuroSATDataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset in ["benge_s1_c", "benge_s1_seg"]:
        assert (
            config.data.datamodule == "BENGEDataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset == "resisc45":
        assert (
            config.data.datamodule == "RESISC45DataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset == "firerisk":
        assert (
            config.data.datamodule == "FireRiskDataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset == "treesatai":
        assert (
            config.data.datamodule == "TreeSatAIDataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset == "ucmerced":
        assert (
            config.data.datamodule == "UCMercedDataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset == "eurosat_sar":
        assert (
            config.data.datamodule == "EuroSATSARDataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    elif row.dataset == "caltech256":
        assert (
            config.data.datamodule == "Caltech256DataModule"
        ), f"{idx=}, {row.run_id=}, {config.data.datamodule=}"
    else:
        raise ValueError(f"{row.dataset=}, {row.run_id=}")

    if "seg" in row.dataset:
        assert (
            row.model == config.model.backbone
        ), f"{row.model=}, {config.model.backbone=}"
    else:
        assert row.model == config.model.name, f"{row.model=}, {config.model.name=}"

    return True


# else:
#    raise ValueError(f"{row.dataset=}, {row.run_id=}")
#
#    if "seg" in row.dataset:
#        assert (
#            row.model == config.model.backbone
#        ), f"{row.model=}, {config.model.backbone=}"
#    else:
#        assert row.model == config.model.name, f"{row.model=}, {config.model.name=}"

#    return True
