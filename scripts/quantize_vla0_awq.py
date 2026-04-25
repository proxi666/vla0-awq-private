#!/usr/bin/env python3
"""Create an AWQ checkpoint for the trained VLA-0 Qwen2.5-VL model.

The default path is intentionally a tiny local smoke run. It verifies that
VLA-formatted calibration samples can be built and that the AWQ backend can
start without touching the existing BF16 checkpoint. Use the larger Kaggle
settings in the accompanying notebook for the full 128-sample run.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import \
    Qwen2_5_VLForConditionalGeneration

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

from rv_train import constants as C  # noqa: E402
from rv_train.models.qwen.model import format_data  # noqa: E402
from rv_train.train import get_cfg, get_dataloader  # noqa: E402

DEFAULT_CHECKPOINT = VLA0_ROOT / "checkpoints" / "vla0-libero" / "model_last"
DEFAULT_OUTPUT = VLA0_ROOT / "checkpoints" / "vla0-libero" / "model_last_awq"
DEFAULT_OFFLOAD = VLA0_ROOT / "research" / "results" / "active" / "awq_offload"
DECODER_LINEAR_TARGETS = [
    "re:.*(language_model|model)\\.layers\\..*\\.self_attn\\.(q_proj|k_proj|v_proj|o_proj)$",
    "re:.*(language_model|model)\\.layers\\..*\\.mlp\\.(gate_proj|up_proj|down_proj)$",
]
TOKEN_BATCH_KEYS = {
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "token_type_ids",
    "position_ids",
}
PROCESSOR_SIDECAR_FILES = [
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]


@dataclass(frozen=True)
class AwqImports:
    oneshot: Callable
    awq_modifier: type
    awq_mapping: type | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--backend", choices=["llm-compressor", "autoawq"], default="llm-compressor"
    )
    parser.add_argument("--lerobot-repo-id", default="lerobot/libero_10")
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-calibration-samples", type=int, default=4)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--dataset-device", default="cpu")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--max-memory",
        default="",
        help="Comma-separated accelerate max_memory map, for example '0:7GiB,cpu:32GiB'.",
    )
    parser.add_argument("--offload-folder", type=Path, default=DEFAULT_OFFLOAD)
    parser.add_argument(
        "--pipeline",
        default="sequential",
        choices=["independent", "sequential", "basic"],
    )
    parser.add_argument("--sequential-offload-device", default="cpu")
    parser.add_argument(
        "--sequential-targets",
        default="Qwen2_5_VLDecoderLayer",
        help="Comma-separated llm-compressor sequential targets.",
    )
    parser.add_argument("--scheme", default="W4A16_ASYM")
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=128)
    parser.add_argument("--layer-limit", type=int, default=1)
    parser.add_argument(
        "--clip-actions", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run-calibration",
        action="store_true",
        help="Only build calibration samples and print their tensor shapes.",
    )
    return parser.parse_args()


def parse_max_memory(value: str) -> dict[Any, str] | None:
    if not value:
        return None
    parsed: dict[Any, str] = {}
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid max-memory part: {part}")
        key, memory = part.split(":", 1)
        key = key.strip()
        parsed[int(key) if key.isdigit() else key] = memory.strip()
    return parsed


def checkpoint_parent(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.suffix == ".pth":
        return checkpoint.parent
    return checkpoint.parent


def load_dataset_stats(checkpoint: Path) -> dict[str, Any]:
    stats_path = checkpoint_parent(checkpoint) / "dataset_stats.pkl"
    with stats_path.open("rb") as handle:
        return pickle.load(handle)


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(max(0.0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def action_text_from_tensor(
    actions: torch.Tensor,
    action_stats: dict[str, Any],
    num_bins_actions: int,
    *,
    clip_actions: bool,
) -> str:
    min_act = torch.tensor(
        action_stats["min"], dtype=actions.dtype, device=actions.device
    )
    max_act = torch.tensor(
        action_stats["max"], dtype=actions.dtype, device=actions.device
    )
    if clip_actions:
        actions = torch.maximum(torch.minimum(actions, max_act), min_act)
    if torch.any(actions < min_act) or torch.any(actions > max_act):
        raise ValueError("Calibration action is outside dataset min/max bounds")
    actions = (actions - min_act) / (max_act - min_act)
    actions = torch.round(actions * num_bins_actions).long().reshape(-1)
    return " ".join(map(str, actions.cpu().tolist()))


def tile_pil_images(images: list[Image.Image]) -> Image.Image:
    widths = [image.width for image in images]
    heights = [image.height for image in images]
    canvas = Image.new("RGB", (sum(widths), max(heights)))
    x_offset = 0
    for image in images:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width
    return canvas


def pil_images_from_batch(
    batch: dict[str, Any], *, tiled_rgb_imgs: bool
) -> list[Image.Image]:
    rgb = batch["rgb"]
    if not isinstance(rgb, torch.Tensor):
        rgb = torch.as_tensor(rgb)
    rgb = rgb[0]
    images: list[Image.Image] = []
    for history_index in range(rgb.shape[0]):
        for cam_index in range(rgb.shape[1]):
            array = (
                rgb[history_index, cam_index].detach().cpu().numpy().astype(np.uint8)
            )
            images.append(Image.fromarray(array))
    if tiled_rgb_imgs:
        return [tile_pil_images(images)]
    return images


def build_system_message(cfg) -> str:
    if cfg.MODEL.QWEN.action_type == C.ORIGINAL:
        act_dim = cfg.MODEL.QWEN.original_action_dim
    else:
        act_dim = 7
    horizon = cfg.MODEL.QWEN.horizon
    num_bins_actions = cfg.MODEL.QWEN.num_bins_actions
    return (
        f"Analyze the input image and predict robot actions for the next {horizon} timesteps. "
        f"Each action has {act_dim} dimensions. Output a single sequence of {horizon * act_dim} "
        f"integers (0-{num_bins_actions} each), representing the {horizon} timesteps sequentially. "
        "Provide only space separated numbers. Nothing else."
    )


class VLAQwenCalibrationDataset(Dataset):
    def __init__(
        self,
        *,
        cfg,
        checkpoint: Path,
        split: str,
        start_index: int,
        num_samples: int,
        dataset_device: str,
        lerobot_repo_id: str,
        clip_actions: bool,
    ):
        self.cfg = override_lerobot_settings(cfg.clone(), lerobot_repo_id)
        self.checkpoint = checkpoint
        self.dataset_stats = load_dataset_stats(checkpoint)
        self.system_message = build_system_message(self.cfg)
        self.num_bins_actions = self.cfg.MODEL.QWEN.num_bins_actions
        self.clip_actions = clip_actions

        rgb_size = tuple(self.cfg.MODEL.QWEN.rgb_img_size)
        min_pixel = max_pixel = int(rgb_size[0]) * int(rgb_size[1])
        if self.cfg.MODEL.QWEN.rgb_input and self.cfg.MODEL.QWEN.tiled_rgb_imgs:
            min_pixel *= self.cfg.MODEL.QWEN.history * self.cfg.MODEL.QWEN.num_cam
            max_pixel *= self.cfg.MODEL.QWEN.history * self.cfg.MODEL.QWEN.num_cam
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            str(checkpoint),
            min_pixels=min_pixel,
            max_pixels=max_pixel,
            use_fast=False,
        )

        old_cwd = Path.cwd()
        self.samples: list[dict[str, torch.Tensor]] = []
        self.metadata: list[dict[str, Any]] = []
        try:
            # RoboVerse config paths are relative to the VLA-0 repository root.
            os.chdir(VLA0_ROOT)
            dataset = get_dataloader(split=split, cfg=self.cfg, get_dataset=True)
            loader = DataLoader(
                dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False
            )
            wanted = set(range(start_index, start_index + num_samples))
            for idx, raw_batch in enumerate(loader):
                if idx not in wanted:
                    continue
                batch = loader.dataset.batch_proc(raw_batch, dataset_device)
                model_inputs, action_txt = self._build_model_inputs(batch)
                self.samples.append(model_inputs)
                self.metadata.append(
                    {
                        "dataset_index": idx,
                        "instruction": batch["instr"][0],
                        "action_text": action_txt,
                        "action_token_count": len(action_txt.split()),
                    }
                )
                if len(self.samples) == num_samples:
                    break
        finally:
            os.chdir(old_cwd)
        if len(self.samples) != num_samples:
            raise IndexError(
                f"Requested {num_samples} samples from index {start_index}, got {len(self.samples)}"
            )

    def _build_model_inputs(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], str]:
        images = pil_images_from_batch(
            batch, tiled_rgb_imgs=self.cfg.MODEL.QWEN.tiled_rgb_imgs
        )
        action_txt = action_text_from_tensor(
            batch["out_ori_act"][0],
            self.dataset_stats["out_ori_act"],
            self.num_bins_actions,
            clip_actions=self.clip_actions,
        )
        example = format_data(
            system_message=self.system_message,
            image=images,
            instr=batch["instr"][0],
            action_txt=action_txt,
        )
        text = self.processor.apply_chat_template(
            example,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=self.cfg.MODEL.QWEN.add_vision_id,
        )
        image_inputs = process_vision_info(example)[0]
        model_inputs = self.processor(
            text=[text],
            images=[image_inputs],
            return_tensors="pt",
            padding=True,
        )
        return dict(model_inputs), action_txt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def single_item_collator(
    items: list[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    if len(items) != 1:
        raise ValueError("VLA AWQ calibration currently supports batch size 1 only")
    return items[0]


def tensor_shapes(item: dict[str, Any]) -> dict[str, Any]:
    shapes = {}
    for key, value in item.items():
        shapes[key] = (
            list(value.shape) if hasattr(value, "shape") else type(value).__name__
        )
    return shapes


def serialize_calibration_tensor(key: str, value: torch.Tensor) -> Any:
    tensor = value.detach().cpu()
    if key in TOKEN_BATCH_KEYS and tensor.ndim >= 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.tolist()


def build_hf_calibration_dataset(calibration_dataset: VLAQwenCalibrationDataset):
    try:
        from datasets import Dataset as HFDataset
    except ImportError as exc:
        raise ImportError(
            "llm-compressor calibration requires the `datasets` package. "
            "Install the AWQ requirements in the isolated env first."
        ) from exc

    records = []
    for sample in calibration_dataset:
        records.append(
            {
                key: serialize_calibration_tensor(key, value)
                for key, value in sample.items()
            }
        )
    return HFDataset.from_list(records)


def vla_qwen_data_collator(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if len(features) != 1:
        raise ValueError("VLA AWQ calibration currently supports batch size 1 only")

    sample = features[0]
    batch: dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        tensor = torch.as_tensor(value)
        if key in TOKEN_BATCH_KEYS and tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif key == "image_grid_thw" and tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        batch[key] = tensor
    return batch


def import_llm_compressor_awq() -> AwqImports:
    try:
        from llmcompressor import oneshot
    except ImportError as exc:
        raise ImportError(
            "llm-compressor is not installed. Install it in an isolated env, for example: "
            "`pip install llmcompressor`."
        ) from exc

    try:
        from llmcompressor.modifiers.awq import AWQModifier
    except ImportError:
        from llmcompressor.modifiers.transform.awq import AWQModifier

    awq_mapping = None
    for module_name in [
        "llmcompressor.modifiers.awq.mappings",
        "llmcompressor.modifiers.transform.awq.mappings",
    ]:
        try:
            module = importlib.import_module(module_name)
            awq_mapping = getattr(module, "AWQMapping")
            break
        except (ImportError, AttributeError):
            continue

    return AwqImports(
        oneshot=oneshot, awq_modifier=AWQModifier, awq_mapping=awq_mapping
    )


def build_awq_mappings(
    awq_mapping: type | None,
    *,
    layer_limit: int | None,
    num_hidden_layers: int,
) -> list[Any] | None:
    if awq_mapping is None:
        return None
    selected_layers = (
        num_hidden_layers if layer_limit is None or layer_limit <= 0 else layer_limit
    )
    mappings = []
    for layer_index in range(selected_layers):
        layer_prefix = f"re:.*(language_model|model)\\.layers\\.{layer_index}\\."
        mappings.extend(
            [
                awq_mapping(
                    layer_prefix + "input_layernorm$",
                    [
                        layer_prefix + "self_attn\\.q_proj$",
                        layer_prefix + "self_attn\\.k_proj$",
                        layer_prefix + "self_attn\\.v_proj$",
                    ],
                ),
                awq_mapping(
                    layer_prefix + "self_attn\\.v_proj$",
                    [layer_prefix + "self_attn\\.o_proj$"],
                ),
                awq_mapping(
                    layer_prefix + "post_attention_layernorm$",
                    [
                        layer_prefix + "mlp\\.gate_proj$",
                        layer_prefix + "mlp\\.up_proj$",
                    ],
                ),
                awq_mapping(
                    layer_prefix + "mlp\\.up_proj$",
                    [layer_prefix + "mlp\\.down_proj$"],
                ),
            ]
        )
    return mappings


def build_ignore_patterns(layer_limit: int | None, num_hidden_layers: int) -> list[str]:
    ignore = [
        "re:.*visual.*",
        "re:.*embed_tokens.*",
        "re:.*lm_head.*",
    ]
    if layer_limit is not None and layer_limit > 0:
        for layer_index in range(layer_limit, num_hidden_layers):
            ignore.append(f"re:.*(language_model|model)\\.layers\\.{layer_index}\\..*")
    return ignore


def build_awq_recipe(args: argparse.Namespace, num_hidden_layers: int):
    imports = import_llm_compressor_awq()
    ignore = build_ignore_patterns(args.layer_limit, num_hidden_layers)
    mappings = build_awq_mappings(
        imports.awq_mapping,
        layer_limit=args.layer_limit,
        num_hidden_layers=num_hidden_layers,
    )
    kwargs = {
        "ignore": ignore,
        "scheme": args.scheme,
        "targets": DECODER_LINEAR_TARGETS,
    }
    if mappings is not None:
        kwargs["mappings"] = mappings
    try:
        modifier = imports.awq_modifier(**kwargs)
    except TypeError:
        kwargs.pop("mappings", None)
        modifier = imports.awq_modifier(**kwargs)
    return imports.oneshot, [modifier], ignore


def get_num_hidden_layers(checkpoint: Path) -> int:
    config = AutoConfig.from_pretrained(str(checkpoint))
    return int(getattr(config, "num_hidden_layers", 36))


def ensure_output_path(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Pass --overwrite to replace it."
            )
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)


def load_qwen_model_for_quantization(args: argparse.Namespace):
    max_memory = parse_max_memory(args.max_memory)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
    }
    if max_memory:
        kwargs["max_memory"] = max_memory
    if args.offload_folder:
        args.offload_folder.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(args.offload_folder)
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.checkpoint), **kwargs
    )


def save_llm_compressor_checkpoint(
    *,
    model,
    processor: Qwen2_5_VLProcessor,
    source_checkpoint: Path,
    output: Path,
) -> None:
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        get_model_compressor, update_and_save_recipe)

    compressor = get_model_compressor(
        model=model,
        save_compressed=True,
        skip_sparsity_compression_stats=True,
    )
    if compressor is not None:
        compressor.compress_model(model)

    # Transformers' offloaded save path currently mis-maps some Qwen2.5-VL visual
    # state-dict keys after checkpoint key conversion. Supplying a CPU state dict
    # bypasses the module map while retaining the compressed tensor config.
    if hasattr(model, "hf_device_map"):
        delattr(model, "hf_device_map")
    state_dict = {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in model.state_dict().items()
    }

    original_save_pretrained = getattr(model.save_pretrained, "__wrapped__", None)
    if original_save_pretrained is not None:
        bound_save_pretrained = original_save_pretrained.__get__(model, model.__class__)
        bound_save_pretrained(
            str(output), safe_serialization=True, state_dict=state_dict
        )
    else:
        model.save_pretrained(
            str(output), safe_serialization=True, state_dict=state_dict
        )

    if compressor is not None:
        compressor.update_config(str(output))
    try:
        update_and_save_recipe(model.name_or_path, str(output))
    except Exception as exc:  # noqa: BLE001
        print(f"[awq] recipe save skipped: {exc}", flush=True)
    processor.save_pretrained(str(output))
    copy_missing_sidecar_files(source_checkpoint, output)


def copy_missing_sidecar_files(source_checkpoint: Path, output: Path) -> None:
    for filename in PROCESSOR_SIDECAR_FILES:
        source = source_checkpoint / filename
        destination = output / filename
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)


def run_llm_compressor(
    args: argparse.Namespace, calibration_dataset: VLAQwenCalibrationDataset
) -> dict[str, Any]:
    oneshot, recipe, ignore = build_awq_recipe(
        args,
        num_hidden_layers=get_num_hidden_layers(args.checkpoint),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_qwen_model_for_quantization(args)
    hf_dataset = build_hf_calibration_dataset(calibration_dataset)
    sequential_targets = [
        item.strip() for item in args.sequential_targets.split(",") if item.strip()
    ]

    started_at = time.perf_counter()
    compressed_model = oneshot(
        model=model,
        processor=calibration_dataset.processor,
        dataset=hf_dataset,
        data_collator=vla_qwen_data_collator,
        recipe=recipe,
        num_calibration_samples=len(calibration_dataset),
        shuffle_calibration_samples=False,
        batch_size=args.calibration_batch_size,
        pipeline=args.pipeline,
        sequential_targets=sequential_targets or None,
        sequential_offload_device=args.sequential_offload_device,
        output_dir=None,
        save_compressed=True,
    )
    elapsed_seconds = time.perf_counter() - started_at

    save_llm_compressor_checkpoint(
        model=compressed_model,
        processor=calibration_dataset.processor,
        source_checkpoint=args.checkpoint,
        output=args.output,
    )
    cuda_summary: dict[str, float] = {}
    if torch.cuda.is_available():
        cuda_summary = {
            "cuda_memory_allocated_mib": torch.cuda.memory_allocated() / 1024**2,
            "cuda_memory_reserved_mib": torch.cuda.memory_reserved() / 1024**2,
            "cuda_max_memory_allocated_mib": torch.cuda.max_memory_allocated()
            / 1024**2,
            "cuda_max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        }
    del model
    del compressed_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "backend": "llm-compressor",
        "elapsed_seconds": elapsed_seconds,
        "ignore": ignore,
        "pipeline": args.pipeline,
        "sequential_targets": sequential_targets,
        "sequential_offload_device": args.sequential_offload_device,
        **cuda_summary,
    }


def run_autoawq(
    args: argparse.Namespace, calibration_dataset: VLAQwenCalibrationDataset
) -> dict[str, Any]:
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "AutoAWQ is not installed. Install it in an isolated cloud runtime."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.checkpoint), trust_remote_code=True
    )
    model = AutoAWQForCausalLM.from_pretrained(
        str(args.checkpoint), device_map=args.device_map, safetensors=True
    )
    calib_data = []
    for meta in calibration_dataset.metadata:
        calib_data.append(f"{meta['instruction']}\n{meta['action_text']}")
    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM",
    }
    started_at = time.perf_counter()
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
    model.save_quantized(str(args.output), safetensors=True, shard_size="4GB")
    tokenizer.save_pretrained(str(args.output))
    calibration_dataset.processor.save_pretrained(str(args.output))
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "backend": "autoawq",
        "elapsed_seconds": elapsed_seconds,
        "note": "AutoAWQ fallback uses text-only calibration strings and may not support Qwen2.5-VL.",
    }


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.offload_folder = args.offload_folder.expanduser().resolve()

    cfg = get_cfg(str(checkpoint_parent(args.checkpoint) / "config.yaml"), "")
    calibration_dataset = VLAQwenCalibrationDataset(
        cfg=cfg,
        checkpoint=args.checkpoint,
        split=args.split,
        start_index=args.start_index,
        num_samples=args.num_calibration_samples,
        dataset_device=args.dataset_device,
        lerobot_repo_id=args.lerobot_repo_id,
        clip_actions=args.clip_actions,
    )

    print("[awq] calibration samples:", len(calibration_dataset), flush=True)
    print(
        "[awq] first metadata:",
        json.dumps(calibration_dataset.metadata[0], indent=2),
        flush=True,
    )
    print(
        "[awq] first tensor shapes:",
        json.dumps(tensor_shapes(calibration_dataset[0]), indent=2),
        flush=True,
    )
    if args.dry_run_calibration:
        try:
            hf_dataset = build_hf_calibration_dataset(calibration_dataset)
            collated = vla_qwen_data_collator([hf_dataset[0]])
            print(
                "[awq] llm-compressor collator shapes:",
                json.dumps(tensor_shapes(collated), indent=2),
                flush=True,
            )
        except ImportError as exc:
            print(f"[awq] llm-compressor dataset preview skipped: {exc}", flush=True)
        return

    ensure_output_path(args.output, args.overwrite)
    if args.backend == "llm-compressor":
        backend_summary = run_llm_compressor(args, calibration_dataset)
    else:
        backend_summary = run_autoawq(args, calibration_dataset)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "num_calibration_samples": len(calibration_dataset),
        "start_index": args.start_index,
        "lerobot_repo_id": args.lerobot_repo_id,
        "layer_limit": args.layer_limit,
        "backend": backend_summary,
        "calibration_metadata": calibration_dataset.metadata,
    }
    summary_path = args.output / "awq_quantization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[awq] wrote:", args.output, flush=True)
    print("[awq] summary:", summary_path, flush=True)
    print(
        "[awq] elapsed:",
        format_duration(backend_summary["elapsed_seconds"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
