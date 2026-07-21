#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import save_file
from safetensors import safe_open

try:
    import torch
    from safetensors.torch import load_file as load_torch_file
except ImportError:  # pragma: no cover - optional dependency for bf16 source weights
    torch = None
    load_torch_file = None


def parse_positive_ints(value: str, *, expected_len: int, field_name: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise ValueError(f"{field_name} must contain {expected_len} comma-separated integers")
    parsed = [int(part) for part in parts]
    if any(part <= 0 for part in parsed):
        raise ValueError(f"{field_name} values must be positive")
    return parsed


def allocate_even_counts(total: int, total_workers: int) -> list[int]:
    if total_workers <= 0:
        raise ValueError("workers must be positive")
    if total < total_workers:
        raise ValueError(f"cannot allocate {total} units across {total_workers} non-empty workers")
    base = total // total_workers
    remainder = total % total_workers
    return [base + (1 if worker < remainder else 0) for worker in range(total_workers)]


def allocate_weighted_counts(total: int, weights: list[int]) -> list[int]:
    if total < len(weights):
        raise ValueError(f"cannot allocate {total} units across {len(weights)} non-empty workers")
    weight_sum = sum(weights)
    raw = [(total * weight) / weight_sum for weight in weights]
    counts = [max(1, int(value)) for value in raw]
    while sum(counts) > total:
        candidates = [
            (raw[idx] - counts[idx], idx)
            for idx in range(len(counts))
            if counts[idx] > 1
        ]
        if not candidates:
            raise ValueError("unable to reduce weighted allocation without emptying a worker")
        _, idx = min(candidates)
        counts[idx] -= 1
    while sum(counts) < total:
        candidates = [(raw[idx] - counts[idx], idx) for idx in range(len(counts))]
        _, idx = max(candidates)
        counts[idx] += 1
    return counts


def ranges_from_counts(counts: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for width in counts:
        end = start + width
        ranges.append((start, end))
        start = end
    return ranges


def attention_shard_geometry(
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    group_ranges: list[tuple[int, int]],
    worker_position: int,
) -> tuple[int, int, int, int]:
    if hidden_dim % num_heads != 0:
        raise ValueError(f"hidden_dim {hidden_dim} is not divisible by num_heads {num_heads}")
    if num_kv_heads == 0 or num_heads % num_kv_heads != 0:
        raise ValueError(
            f"unsupported grouped-query attention geometry: num_heads={num_heads} num_kv_heads={num_kv_heads}"
        )
    head_dim = hidden_dim // num_heads
    q_heads_per_kv_head = num_heads // num_kv_heads
    q_group_width = q_heads_per_kv_head * head_dim
    group_start, group_end = group_ranges[worker_position]
    group_count = group_end - group_start
    q_start = group_start * q_group_width
    q_cols = group_count * q_group_width
    kv_start = group_start * head_dim
    kv_cols = group_count * head_dim
    return q_start, q_cols, kv_start, kv_cols


def copy_tokenizer(repo_id: str, out_dir: Path) -> None:
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    shutil.copy2(tokenizer_path, out_dir / "tokenizer.json")
    tokenizer_config_path = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json")
    shutil.copy2(tokenizer_config_path, out_dir / "tokenizer_config.json")


def load_config(repo_id: str) -> dict:
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_name_map(layer_idx: int) -> dict[str, str]:
    prefix = f"model.layers.{layer_idx}"
    return {
        "w_q": f"{prefix}.self_attn.q_proj.weight",
        "w_k": f"{prefix}.self_attn.k_proj.weight",
        "w_v": f"{prefix}.self_attn.v_proj.weight",
        "w_o": f"{prefix}.self_attn.o_proj.weight",
        "w_up": f"{prefix}.mlp.up_proj.weight",
        "w_gate": f"{prefix}.mlp.gate_proj.weight",
        "w_down": f"{prefix}.mlp.down_proj.weight",
        "attn_norm": f"{prefix}.input_layernorm.weight",
        "mlp_norm": f"{prefix}.post_attention_layernorm.weight",
    }


def tensor_to_f32_numpy(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=np.float32))


def tensor_from_file(source, name: str) -> np.ndarray:
    return tensor_to_f32_numpy(source.get_tensor(name))


def transpose_rows_slice(source, name: str, start: int, length: int) -> np.ndarray:
    tensor = source.get_tensor(name)
    return np.ascontiguousarray(np.asarray(tensor[start : start + length, :], dtype=np.float32).T)


def transpose_cols_slice(source, name: str, start: int, length: int) -> np.ndarray:
    tensor = source.get_tensor(name)
    return np.ascontiguousarray(np.asarray(tensor[:, start : start + length], dtype=np.float32).T)


def tensor_from_torch(source, name: str) -> np.ndarray:
    tensor = source[name]
    return tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()


def transpose_rows_slice_torch(source, name: str, start: int, length: int) -> np.ndarray:
    tensor = source[name]
    return (
        tensor[start : start + length, :]
        .transpose(0, 1)
        .detach()
        .to(dtype=torch.float32, device="cpu")
        .contiguous()
        .numpy()
    )


def transpose_cols_slice_torch(source, name: str, start: int, length: int) -> np.ndarray:
    tensor = source[name]
    return (
        tensor[:, start : start + length]
        .transpose(0, 1)
        .detach()
        .to(dtype=torch.float32, device="cpu")
        .contiguous()
        .numpy()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a real HF causal LM model and convert it to MeshNet shards.")
    parser.add_argument("--repo-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--model-id", default="smollm2-135m-instruct")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--kv-groups-per-worker",
        help="Comma-separated explicit GQA KV-group counts per worker; must sum to num_key_value_heads.",
    )
    parser.add_argument(
        "--worker-weights",
        help="Comma-separated positive worker weights used for KV groups and MLP columns when explicit counts are not provided.",
    )
    parser.add_argument(
        "--mlp-weights",
        help="Comma-separated positive worker weights for MLP intermediate columns; defaults to --worker-weights or even split.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve() / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.repo_id)
    model_path = hf_hub_download(repo_id=args.repo_id, filename="model.safetensors")
    copy_tokenizer(args.repo_id, out_dir)

    hidden_dim = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    num_layers = int(config["num_hidden_layers"])
    vocab_size = int(config["vocab_size"])
    intermediate_size = int(config["intermediate_size"])
    rms_norm_eps = float(config["rms_norm_eps"])
    rope_base = float(config.get("rope_theta", 10000.0))
    head_dim = hidden_dim // num_heads
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.kv_groups_per_worker:
        kv_group_counts = parse_positive_ints(
            args.kv_groups_per_worker,
            expected_len=args.workers,
            field_name="--kv-groups-per-worker",
        )
        if sum(kv_group_counts) != num_kv_heads:
            raise ValueError(
                f"--kv-groups-per-worker must sum to num_key_value_heads={num_kv_heads}, got {sum(kv_group_counts)}"
            )
    elif args.worker_weights:
        kv_group_counts = allocate_weighted_counts(
            num_kv_heads,
            parse_positive_ints(args.worker_weights, expected_len=args.workers, field_name="--worker-weights"),
        )
    else:
        kv_group_counts = allocate_even_counts(num_kv_heads, args.workers)
    group_ranges = ranges_from_counts(kv_group_counts)

    if args.mlp_weights:
        mlp_weights = parse_positive_ints(args.mlp_weights, expected_len=args.workers, field_name="--mlp-weights")
    elif args.worker_weights:
        mlp_weights = parse_positive_ints(args.worker_weights, expected_len=args.workers, field_name="--worker-weights")
    else:
        mlp_weights = [1] * args.workers
    mlp_ranges = ranges_from_counts(allocate_weighted_counts(intermediate_size, mlp_weights))

    model_size_bytes = os.path.getsize(model_path)
    save_json(
        out_dir / "model.json",
        {
            "model_id": args.model_id,
            "tensor_parallelism_dim": hidden_dim,
            "attention_head_count": num_heads,
            "kv_head_count": num_kv_heads,
            "transformer_layer_count": num_layers,
            "total_model_bytes": model_size_bytes,
            "tokenizer_file": "tokenizer.json",
            "tokenizer_config_file": "tokenizer_config.json",
        },
    )

    metadata = {
        "mesh.model_id": args.model_id,
        "mesh.hidden_dim": str(hidden_dim),
        "mesh.num_heads": str(num_heads),
        "mesh.num_kv_heads": str(num_kv_heads),
        "mesh.num_layers": str(num_layers),
        "mesh.vocab_size": str(vocab_size),
        "mesh.intermediate_size": str(intermediate_size),
        "mesh.rms_norm_eps": str(rms_norm_eps),
        "mesh.rope_base": str(rope_base),
    }

    try:
        with safe_open(model_path, framework="np") as source:
            keys = set(source.keys())
            embedding = tensor_from_file(source, "model.embed_tokens.weight")
            final_norm = tensor_from_file(source, "model.norm.weight")
            lm_head_name = "lm_head.weight" if "lm_head.weight" in keys else "model.embed_tokens.weight"
            lm_head = np.ascontiguousarray(np.asarray(source.get_tensor(lm_head_name), dtype=np.float32).T)

            for worker_position in range(args.workers):
                q_start, q_cols, kv_start, kv_cols = attention_shard_geometry(
                    hidden_dim, num_heads, num_kv_heads, group_ranges, worker_position
                )
                mlp_start, mlp_end = mlp_ranges[worker_position]
                mlp_cols = mlp_end - mlp_start

                tensors: dict[str, np.ndarray] = {
                    "embedding": embedding,
                    "final_norm": final_norm,
                    "lm_head": lm_head,
                }

                for layer_idx in range(num_layers):
                    names = tensor_name_map(layer_idx)
                    tensors[f"layers.{layer_idx}.w_q"] = transpose_rows_slice(
                        source, names["w_q"], q_start, q_cols
                    )
                    tensors[f"layers.{layer_idx}.w_k"] = transpose_rows_slice(
                        source, names["w_k"], kv_start, kv_cols
                    )
                    tensors[f"layers.{layer_idx}.w_v"] = transpose_rows_slice(
                        source, names["w_v"], kv_start, kv_cols
                    )
                    tensors[f"layers.{layer_idx}.w_o"] = transpose_cols_slice(
                        source, names["w_o"], q_start, q_cols
                    )
                    tensors[f"layers.{layer_idx}.w_up"] = transpose_rows_slice(
                        source, names["w_up"], mlp_start, mlp_cols
                    )
                    tensors[f"layers.{layer_idx}.w_gate"] = transpose_rows_slice(
                        source, names["w_gate"], mlp_start, mlp_cols
                    )
                    tensors[f"layers.{layer_idx}.w_down"] = transpose_cols_slice(
                        source, names["w_down"], mlp_start, mlp_cols
                    )
                    tensors[f"layers.{layer_idx}.attn_norm"] = tensor_from_file(source, names["attn_norm"])
                    tensors[f"layers.{layer_idx}.mlp_norm"] = tensor_from_file(source, names["mlp_norm"])

                shard_path = out_dir / f"shard-{worker_position}-of-{args.workers}.safetensors"
                tmp_shard_path = out_dir / f".shard-{worker_position}-of-{args.workers}.safetensors.tmp"
                tmp_shard_path.unlink(missing_ok=True)
                shard_metadata = {
                    **metadata,
                    "mesh.worker_position": str(worker_position),
                    "mesh.total_workers": str(args.workers),
                    "mesh.column_start": str(q_start),
                    "mesh.column_end": str(q_start + q_cols),
                    "mesh.mlp_start": str(mlp_start),
                    "mesh.mlp_end": str(mlp_end),
                }
                save_file(tensors, str(tmp_shard_path), metadata=shard_metadata)
                tmp_shard_path.replace(shard_path)

                digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
                save_json(
                    out_dir / f"shard-{worker_position}-of-{args.workers}.manifest.json",
                    {
                        "model_id": args.model_id,
                        "worker_position": worker_position,
                        "total_workers": args.workers,
                        "column_start": q_start,
                        "column_end": q_start + q_cols,
                        "expected_sha256": digest,
                    },
                )

                print(f"wrote {shard_path}")
    except TypeError as exc:
        if "bfloat16" not in str(exc) or load_torch_file is None or torch is None:
            raise

        source = load_torch_file(model_path, device="cpu")
        embedding = tensor_from_torch(source, "model.embed_tokens.weight")
        final_norm = tensor_from_torch(source, "model.norm.weight")
        lm_head_name = "lm_head.weight" if "lm_head.weight" in source else "model.embed_tokens.weight"
        lm_head = (
            source[lm_head_name]
            .transpose(0, 1)
            .detach()
            .to(dtype=torch.float32, device="cpu")
            .contiguous()
            .numpy()
        )

        for worker_position in range(args.workers):
            q_start, q_cols, kv_start, kv_cols = attention_shard_geometry(
                hidden_dim, num_heads, num_kv_heads, group_ranges, worker_position
            )
            mlp_start, mlp_end = mlp_ranges[worker_position]
            mlp_cols = mlp_end - mlp_start

            tensors: dict[str, np.ndarray] = {
                "embedding": embedding,
                "final_norm": final_norm,
                "lm_head": lm_head,
            }

            for layer_idx in range(num_layers):
                names = tensor_name_map(layer_idx)
                tensors[f"layers.{layer_idx}.w_q"] = transpose_rows_slice_torch(
                    source, names["w_q"], q_start, q_cols
                )
                tensors[f"layers.{layer_idx}.w_k"] = transpose_rows_slice_torch(
                    source, names["w_k"], kv_start, kv_cols
                )
                tensors[f"layers.{layer_idx}.w_v"] = transpose_rows_slice_torch(
                    source, names["w_v"], kv_start, kv_cols
                )
                tensors[f"layers.{layer_idx}.w_o"] = transpose_cols_slice_torch(
                    source, names["w_o"], q_start, q_cols
                )
                tensors[f"layers.{layer_idx}.w_up"] = transpose_rows_slice_torch(
                    source, names["w_up"], mlp_start, mlp_cols
                )
                tensors[f"layers.{layer_idx}.w_gate"] = transpose_rows_slice_torch(
                    source, names["w_gate"], mlp_start, mlp_cols
                )
                tensors[f"layers.{layer_idx}.w_down"] = transpose_cols_slice_torch(
                    source, names["w_down"], mlp_start, mlp_cols
                )
                tensors[f"layers.{layer_idx}.attn_norm"] = tensor_from_torch(source, names["attn_norm"])
                tensors[f"layers.{layer_idx}.mlp_norm"] = tensor_from_torch(source, names["mlp_norm"])

            shard_path = out_dir / f"shard-{worker_position}-of-{args.workers}.safetensors"
            tmp_shard_path = out_dir / f".shard-{worker_position}-of-{args.workers}.safetensors.tmp"
            tmp_shard_path.unlink(missing_ok=True)
            shard_metadata = {
                **metadata,
                "mesh.worker_position": str(worker_position),
                "mesh.total_workers": str(args.workers),
                "mesh.column_start": str(q_start),
                "mesh.column_end": str(q_start + q_cols),
                "mesh.mlp_start": str(mlp_start),
                "mesh.mlp_end": str(mlp_end),
            }
            save_file(tensors, str(tmp_shard_path), metadata=shard_metadata)
            tmp_shard_path.replace(shard_path)

            digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
            save_json(
                out_dir / f"shard-{worker_position}-of-{args.workers}.manifest.json",
                {
                    "model_id": args.model_id,
                    "worker_position": worker_position,
                    "total_workers": args.workers,
                    "column_start": q_start,
                    "column_end": q_start + q_cols,
                    "expected_sha256": digest,
                },
            )

            print(f"wrote {shard_path}")


if __name__ == "__main__":
    main()
