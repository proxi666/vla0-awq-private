import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

VLA0_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = VLA0_ROOT.parent
LEROBOT_SRC_CANDIDATES = [
    WORKSPACE_ROOT / "lerobot" / "src",
    VLA0_ROOT / "libs" / "RoboVerse" / "libs" / "lerobot" / "src",
]
for lerobot_src in LEROBOT_SRC_CANDIDATES:
    if lerobot_src.exists():
        sys.path.insert(0, str(lerobot_src))
        break

from compare_vla0_generation import override_lerobot_settings  # noqa: E402

from rv_train.train import (get_cfg, get_dataloader,  # noqa: E402
                            get_pretrained_model)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-sample VLA-0 generation comparisons across load modes."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Emit one progress update every N samples, plus the first and last sample.",
    )
    parser.add_argument(
        "--load-modes",
        nargs="+",
        default=["bf16", "int8", "nf4"],
        choices=["bf16", "int8", "nf4", "awq"],
    )
    parser.add_argument("--generate-temperature", type=float, default=0.0)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument(
        "--lerobot-repo-id",
        type=str,
        default="lerobot/libero_10",
    )
    return parser.parse_args()


def collect_batches(cfg, split: str, start_index: int, num_samples: int, device: str):
    dataset = get_dataloader(split=split, cfg=cfg, get_dataset=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    wanted = set(range(start_index, start_index + num_samples))
    batches = []
    for idx, batch in enumerate(loader):
        if idx in wanted:
            batches.append((idx, loader.dataset.batch_proc(batch, device)))
        if len(batches) == num_samples:
            break
    if len(batches) != num_samples:
        raise IndexError(
            f"Requested {num_samples} samples from index {start_index}, got {len(batches)}"
        )
    return batches


def token_count(text: str):
    text = text.strip()
    return 0 if not text else len(text.split())


def compare_actions(reference: torch.Tensor, candidate: torch.Tensor):
    diff = candidate - reference
    return {
        "mae": diff.abs().mean().item(),
        "mse": diff.pow(2).mean().item(),
        "max_abs": diff.abs().max().item(),
    }


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def summarize_mode(
    mode_name: str,
    mode_results: list[dict[str, Any]],
    *,
    has_bf16_reference: bool,
):
    summary: dict[str, Any] = {
        "num_samples": len(mode_results),
        "mean_load_seconds": statistics.mean(
            item["load_seconds"] for item in mode_results
        ),
        "mean_cuda_memory_allocated_mb": statistics.mean(
            item["cuda_memory_allocated_mb"] for item in mode_results
        ),
        "mean_cuda_memory_reserved_mb": statistics.mean(
            item["cuda_memory_reserved_mb"] for item in mode_results
        ),
        "mean_cuda_max_memory_allocated_mb": statistics.mean(
            item["cuda_max_memory_allocated_mb"] for item in mode_results
        ),
        "mean_generation_seconds": statistics.mean(
            item["generation_seconds"] for item in mode_results
        ),
        "mean_token_count": statistics.mean(
            item["pred_token_count"] for item in mode_results
        ),
    }
    if mode_name == "bf16":
        return summary

    summary["vs_bf16_available"] = has_bf16_reference
    if not has_bf16_reference:
        return summary

    drift_items = [
        item["vs_bf16"] for item in mode_results if item.get("vs_bf16") is not None
    ]
    if not drift_items:
        return summary

    maes = [item["mae"] for item in drift_items]
    mses = [item["mse"] for item in drift_items]
    max_abs_values = [item["max_abs"] for item in drift_items]
    exact_matches = sum(1 for item in mode_results if item.get("exact_text_match_bf16"))

    summary.update(
        {
            "mean_mae_vs_bf16": statistics.mean(maes),
            "mean_mse_vs_bf16": statistics.mean(mses),
            "max_abs_vs_bf16": max(max_abs_values),
            "exact_text_match_rate_vs_bf16": exact_matches / len(mode_results),
        }
    )
    return summary


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested, but torch.cuda.is_available() is False"
        )

    checkpoint_path = args.checkpoint.rstrip("/")
    model_folder = "/".join(checkpoint_path.split("/")[:-1])
    cfg = get_cfg(f"{model_folder}/config.yaml", "")
    cfg = override_lerobot_settings(cfg, args.lerobot_repo_id)
    batches = collect_batches(
        cfg, args.split, args.start_index, args.num_samples, args.device
    )

    results: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "split": args.split,
        "start_index": args.start_index,
        "num_samples": args.num_samples,
        "load_modes": args.load_modes,
        "samples": [],
        "modes": {},
    }

    baseline_actions_by_index: dict[int, torch.Tensor] = {}
    baseline_text_by_index: dict[int, str] = {}

    if args.device.startswith("cuda"):
        torch.cuda.set_device(device)

    total_modes = len(args.load_modes)
    for mode_index, load_mode in enumerate(args.load_modes, start=1):
        print(
            f"[compare-batch] loading mode {mode_index}/{total_modes}: {load_mode}",
            flush=True,
        )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        mode_started_at = time.perf_counter()
        load_start = time.perf_counter()
        model, _ = get_pretrained_model(
            args.checkpoint,
            args.device,
            torch_compile=False,
            load_mode=load_mode,
        )
        model.eval()
        load_seconds = time.perf_counter() - load_start
        print(
            f"[compare-batch] {load_mode}: model ready in {load_seconds:.2f}s",
            flush=True,
        )

        mode_results = []
        has_bf16_reference = load_mode == "bf16" or bool(baseline_actions_by_index)
        total_samples = len(batches)
        for sample_position, (sample_index, batch) in enumerate(batches, start=1):
            with torch.no_grad():
                generation_start = time.perf_counter()
                output = model(
                    **batch,
                    get_loss=False,
                    get_action=True,
                    generate_temperature=args.generate_temperature,
                )
                generation_seconds = time.perf_counter() - generation_start

            pred_action = output["out_ori_act"].detach().cpu()
            pred_text = output["pred_action_txt"][0]
            item = {
                "sample_index": sample_index,
                "instruction": batch["instr"][0],
                "load_seconds": load_seconds,
                "generation_seconds": generation_seconds,
                "pred_action_txt": pred_text,
                "gt_action_text": output["gt_action_text"][0],
                "pred_token_count": token_count(pred_text),
                "gt_token_count": token_count(output["gt_action_text"][0]),
                "first_action_step": pred_action[0, 0].tolist(),
                "pred_action": pred_action.squeeze(0).tolist(),
                "cuda_memory_allocated_mb": (
                    torch.cuda.memory_allocated(device) / 1024**2
                    if args.device.startswith("cuda")
                    else 0.0
                ),
                "cuda_memory_reserved_mb": (
                    torch.cuda.memory_reserved(device) / 1024**2
                    if args.device.startswith("cuda")
                    else 0.0
                ),
                "cuda_max_memory_allocated_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if args.device.startswith("cuda")
                    else 0.0
                ),
            }
            if load_mode == "bf16":
                baseline_actions_by_index[sample_index] = pred_action
                baseline_text_by_index[sample_index] = pred_text
                item["vs_bf16"] = None
                item["exact_text_match_bf16"] = None
            else:
                reference_action = baseline_actions_by_index.get(sample_index)
                reference_text = baseline_text_by_index.get(sample_index)
                if reference_action is not None and reference_text is not None:
                    item["vs_bf16"] = compare_actions(reference_action, pred_action)
                    item["exact_text_match_bf16"] = pred_text == reference_text
                else:
                    item["vs_bf16"] = None
                    item["exact_text_match_bf16"] = None
            mode_results.append(item)

            elapsed_mode = time.perf_counter() - mode_started_at
            mean_per_sample = elapsed_mode / sample_position
            remaining_samples = total_samples - sample_position
            eta_seconds = remaining_samples * mean_per_sample
            should_report = (
                sample_position == 1
                or sample_position == total_samples
                or sample_position % max(1, args.progress_every) == 0
            )
            if should_report:
                print(
                    "[compare-batch] "
                    f"{load_mode}: sample {sample_position}/{total_samples} "
                    f"(dataset idx {sample_index}) | gen={generation_seconds:.2f}s | "
                    f"peak_vram={item['cuda_max_memory_allocated_mb']:.1f}MB | "
                    f"elapsed={format_duration(elapsed_mode)} | eta={format_duration(eta_seconds)}",
                    flush=True,
                )

        results["modes"][load_mode] = {
            "summary": summarize_mode(
                load_mode,
                mode_results,
                has_bf16_reference=has_bf16_reference,
            ),
            "samples": mode_results,
        }
        print(
            f"[compare-batch] completed {load_mode}: {json.dumps(results['modes'][load_mode]['summary'], indent=2)}",
            flush=True,
        )

        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    for sample_index, batch in batches:
        results["samples"].append(
            {
                "sample_index": sample_index,
                "instruction": batch["instr"][0],
            }
        )

    output_text = json.dumps(results, indent=2)
    print(output_text)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
