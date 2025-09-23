"""Main script to evaluate the performance of trained models."""

import os
from pathlib import Path

import lightning.pytorch
import pandas as pd
import torch
import tqdm

import src.utils
import src.trainers.linear_eval
import src.trainers.segmentation
import src.trainers.knn_eval


DEFAULT_MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", Path("logs/mlruns").resolve().as_uri()
)


def test_run(row, idx, chosen_ckpt):
    run_id = row.run_id

    lookup_config = src.utils.Dotdict(
        {
            "training_run_id": run_id,
            "mlflow": src.utils.Dotdict({"tracking_uri": DEFAULT_MLFLOW_TRACKING_URI}),
        }
    )
    raw_config = src.utils.get_config_from_mlflow_run(
        lookup_config, run_id_key="training_run_id"
    )
    config = src.utils.Dotdict(raw_config)
    if not hasattr(config, "mlflow"):
        config.mlflow = src.utils.Dotdict({"tracking_uri": DEFAULT_MLFLOW_TRACKING_URI})
    elif not hasattr(config.mlflow, "tracking_uri"):
        config.mlflow.tracking_uri = DEFAULT_MLFLOW_TRACKING_URI
    config.training_run_id = run_id
    config.num_workers = 6

    try:
        src.utils.assert_run_validity(row, config, idx)
    except AttributeError as e:
        print(e)
        print(f"Invalid configuration for run {run_id}")
        print(config)

    src.utils.set_seed(config.seed)
    datamodule, config = src.utils.get_datamodule(config)

    if not hasattr(config.model, "norm_trainable"):
        config.model.norm_trainable = False
    if not hasattr(config.optim, "head_lr"):
        config.optim.head_lr = config.optim.lr

    if "seg" in row.dataset:
        task = src.trainers.segmentation.SegmentationTrainer(
            segmentation_model=config.model.name,
            model=config.model.backbone,
            model_type=config.model.backbone_type,
            feature_map_indices=config.model.feature_map_indices,
            aux_loss_factor=config.optim.aux_loss_factor,
            num_classes=config.data.num_classes,
            in_channels=config.data.in_chans,
            loss="ce",
            lr=config.optim.lr,
            input_size=config.data.img_size,
            patch_size=config.model.patch_size,
            patience=config.optim.lr_schedule_patience,
            freeze_backbone=config.model.freeze_backbone,
            pretrained=config.model.pretrained,
            callbacks=[],
            input_res=config.model.input_res,
            adapter=config.model.adapter,
            adapter_hidden_dim=config.model.adapter_hidden_dim,
            norm_trainable=config.model.norm_trainable,
            adapter_scale=config.model.adapter_scale,
            adapter_shared=config.model.adapter_shared,
            fixed_output_size=config.model.fixed_output_size,
            adapter_type=config.model.adapter_type,
            patch_embed_adapter=config.model.patch_embed_adapter,
            use_mask_token=config.model.use_mask_token,
            train_patch_embed=config.model.train_patch_embed,
            patch_embed_adapter_scale=config.model.patch_embed_adapter_scale,
            train_all_params=config.model.train_all_params,
            train_cls_mask_tokens=config.model.train_cls_mask_tokens,
            adapter_trainable=config.model.adapter_trainable,
            only_bias_trainable=False,
            only_scaler_trainable=False,
            class_names=config.data.class_names,
        )
    else:
        task = src.trainers.linear_eval.LinearEvaluationTask(
            model=config.model.name,
            model_type=config.model.type,
            num_classes=config.data.num_classes,
            in_channels=config.data.in_chans,
            input_size=config.data.img_size,
            patch_size=config.model.patch_size,
            loss="ce",
            lr=config.optim.lr,
            head_lr=config.optim.head_lr,
            patience=config.optim.lr_schedule_patience,
            freeze_backbone=config.model.freeze_backbone,
            pretrained=config.model.pretrained,
            callbacks=[],
            input_res=config.model.input_res,
            adapter=config.model.adapter,
            adapter_scale=config.model.adapter_scale,
            adapter_shared=config.model.adapter_shared,
            adapter_type=config.model.adapter_type,
            adapter_hidden_dim=config.model.adapter_hidden_dim,
            patch_embed_adapter=config.model.patch_embed_adapter,
            train_patch_embed=config.model.train_patch_embed,
            patch_embed_adapter_scale=config.model.patch_embed_adapter_scale,
            train_all_params=config.model.train_all_params,
            train_cls_mask_tokens=config.model.train_cls_mask_tokens,
            fixed_output_size=config.model.fixed_output_size,
            adapter_trainable=config.model.adapter_trainable,
            norm_trainable=config.model.norm_trainable,
        )

    task.model = src.utils.load_weights_from_mlflow_run(
        task.model,
        config,
        run_id_key="training_run_id",
        which_state=chosen_ckpt,
    )
    task.model.eval()

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = lightning.pytorch.Trainer(
        fast_dev_run=False,
        logger=None,
        accelerator=accelerator,
    )

    datamodule.setup("test")
    test_stats = trainer.test(
        model=task,
        dataloaders=datamodule.test_dataloader(),
    )

    return test_stats


if __name__ == "__main__":
    # run_id_file = "configs/run_ids.csv"
    # run_id_file = "configs/few_shot_run_ids.csv"
    # run_id_file = "configs/adapter_ablation_run_ids.csv"
    # run_id_file = "configs/cont_pretrain_run_ids.csv"
    # run_id_file = "configs/run_ids_with_seeds.csv"
    run_id_file = "configs/caltech256_run_ids.csv"
    chosen_ckpt = "best"  # 'last', or 'best'

    run_data = pd.read_csv(run_id_file)

    for idx, row in tqdm.tqdm(run_data.iterrows(), total=run_data.shape[0]):
        filename = "_".join(map(str, row)) + ".txt"
        file = f"logs/tests/{filename}"
        if not os.path.isfile(file):
            print(row.run_id)
            # try:
            with open(file, "a+") as f:
                test_stats = test_run(row, idx, chosen_ckpt)
                print(f"{test_stats=}")

                f.write(str(test_stats) + "\n")
