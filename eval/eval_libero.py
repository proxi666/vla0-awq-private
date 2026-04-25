# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

VLA0_ROOT = Path(__file__).resolve().parents[1]
LIBERO_SRC_CANDIDATES = [
    VLA0_ROOT / "libs" / "LIBERO",
    VLA0_ROOT / "libs" / "LIBERO" / "libero",
    VLA0_ROOT.parent / "LIBERO",
    VLA0_ROOT.parent / "LIBERO" / "libero",
]
LEROBOT_SRC_CANDIDATES = [
    VLA0_ROOT / "libs" / "RoboVerse" / "libs" / "lerobot" / "src",
    VLA0_ROOT.parent / "lerobot" / "src",
]


def _prepend_first_existing_path(candidates):
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            return candidate
    return None


def build_eval_log_dir(
    model_path,
    task_suite_name,
    task_name,
    load_mode="bf16",
    action_horizon=0,
    amp=False,
    generate_temperature=0.0,
    ensemble_prediction=1,
    ensemble_version=1,
    ensemble_2_weight=0.5,
    num_steps=0,
):
    log_dir = f"{model_path}"
    if load_mode != "bf16":
        log_dir = f"{log_dir}_{load_mode}"
    if action_horizon != 0:
        log_dir = f"{log_dir}_ah_{action_horizon}"
    if amp:
        log_dir = f"{log_dir}_amp"
    if generate_temperature > 0:
        log_dir = f"{log_dir}_gen_temp_{generate_temperature}"
    if ensemble_prediction > 1:
        log_dir = f"{log_dir}_ens_pred_{ensemble_prediction}"
    if ensemble_version > 1:
        log_dir = f"{log_dir}_ens_ver_{ensemble_version}"
        if ensemble_version == 2 and ensemble_2_weight != 0.5:
            log_dir = f"{log_dir}_ens_2_weight_{ensemble_2_weight}"
    if num_steps > 0:
        log_dir = f"{log_dir}_num_steps_{num_steps}"
    log_dir = f"{log_dir}_eval_libero"
    log_dir = os.path.join(log_dir, task_suite_name)
    log_dir = os.path.join(log_dir, task_name)
    return log_dir


for libero_src in LIBERO_SRC_CANDIDATES:
    if libero_src.exists():
        sys.path.insert(0, str(libero_src))
        break

_prepend_first_existing_path(LEROBOT_SRC_CANDIDATES)

from roboverse.evals.libero.eval import (eval,  # noqa: E402
                                         get_evaluation_tasks)

from rv_train.train import get_pretrained_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on LIBERO environment")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device to use for loading and evaluation.",
    )
    parser.add_argument(
        "--load-mode",
        type=str,
        default="bf16",
        choices=["bf16", "int8", "nf4", "awq"],
        help="Precision / quantized load mode for the checkpoint.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="Name of the task inside the task suite to evaluate on. Must be provided.",
        required=True,
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        help="Name of the task suite to evaluate on. Must be provided.",
        required=True,
    )
    parser.add_argument(
        "--start_seed", type=int, default=7, help="Start seed for evaluation"
    )  # default value same as as in openpi, https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py#L45
    parser.add_argument(
        "--action_horizon", type=int, default=0, help="Action horizon for evaluation"
    )
    parser.add_argument(
        "--save_all_data",
        action="store_true",
        help="Save all data to the log directory",
    )
    parser.add_argument(
        "--ensemble_prediction",
        type=int,
        default=1,
        help="Ensemble prediction for evaluation",
    )
    parser.add_argument(
        "--not_skip_evaluated",
        action="store_true",
        help="Do not skip evaluated tasks",
    )
    parser.add_argument("--amp", action="store_true", help="Use AMP for evaluation")
    parser.add_argument(
        "--generate_temperature",
        type=float,
        default=0.0,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--ensemble_version",
        type=int,
        default=1,
        help="Ensemble version for evaluation",
    )
    parser.add_argument(
        "--task_id_index",
        type=int,
        default=0,
        help="Index of the task subset to evaluate. Defaults to 0.",
    )
    parser.add_argument(
        "--task_id_count",
        type=int,
        default=1,
        help="Total number of task subsets to evaluate. Defaults to 1.",
    )
    parser.add_argument(
        "--no-torch-compile",
        action="store_true",
        help="Disable torch.compile for evaluation",
        default=False,
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=0,
        help="Number of steps for evaluation. If 0, use the default number of steps for each task suite, otherwise use the specified number of steps.",
    )
    parser.add_argument(
        "--ensemble_2_weight",
        type=float,
        default=0.5,
        help="Weight for the second ensemble prediction",
    )
    args = parser.parse_args()

    all_tasks = get_evaluation_tasks()
    assert (
        args.task_suite_name in all_tasks
    ), f"Task suite {args.task_suite_name} not found in {all_tasks.keys()}"
    assert (
        args.task_name in all_tasks[args.task_suite_name]
    ), f"Task {args.task_name} not found in {all_tasks[args.task_suite_name]}"

    model, cfg = get_pretrained_model(
        args.model_path,
        args.device,
        torch_compile=not args.no_torch_compile,
        load_mode=args.load_mode,
    )
    model.eval()

    assert (
        cfg.EXP.DATASET == "roboverse"
    ), f"Dataset is {cfg.EXP.DATASET}, not roboverse"

    assert cfg.EXP.MODEL in [
        "qwen",
    ], f"Model is {cfg.EXP.MODEL}, not qwen, if expanding must take care of action_type and action_horizon"

    action_type = {
        "qwen": cfg.MODEL.QWEN.action_type,
    }[cfg.EXP.MODEL]

    if args.action_horizon == 0:
        action_horizon = {
            "qwen": cfg.MODEL.QWEN.horizon,
        }[cfg.EXP.MODEL]
    else:
        action_horizon = args.action_horizon

    enable_amp = args.amp
    other_args = {}
    if cfg.EXP.MODEL == "qwen":
        other_args["generate_temperature"] = args.generate_temperature

    device_type = "cuda" if str(args.device).startswith("cuda") else "cpu"

    def model_act(*args, **kwargs):
        with torch.no_grad():
            with torch.autocast(
                device_type=device_type, dtype=torch.bfloat16, enabled=enable_amp
            ):
                out = model(  # noqa
                    *args,
                    **kwargs,
                    **other_args,
                    get_loss=False,
                    get_action=True,
                )
                return out

    log_dir = build_eval_log_dir(
        model_path=args.model_path,
        task_suite_name=args.task_suite_name,
        task_name=args.task_name,
        load_mode=args.load_mode,
        action_horizon=args.action_horizon,
        amp=args.amp,
        generate_temperature=args.generate_temperature,
        ensemble_prediction=args.ensemble_prediction,
        ensemble_version=args.ensemble_version,
        ensemble_2_weight=args.ensemble_2_weight,
        num_steps=args.num_steps,
    )
    os.makedirs(log_dir, exist_ok=True)

    eval(
        model=model_act,
        action_type=action_type,
        cfg_path=cfg.DATALOADER.ROBOVERSE.cfg_path,
        cfg_opts=cfg.DATALOADER.ROBOVERSE.cfg_opts,
        task_name=args.task_name,
        task_suite_name=args.task_suite_name,
        log_dir=log_dir,
        save_video=True,
        seed=args.start_seed,
        action_horizon=action_horizon,
        skip_evaluated=not args.not_skip_evaluated,
        save_all_data=args.save_all_data,
        ensemble_prediction=args.ensemble_prediction,
        ensemble_2_weight=args.ensemble_2_weight,
        ensemble_version=args.ensemble_version,
        task_id_index=args.task_id_index,
        task_id_count=args.task_id_count,
        num_steps=args.num_steps,
    )

    del model
    del model_act
    gc.collect()


if __name__ == "__main__":
    main()
