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
from safetensors.torch import load_file
import torch


def partition_start(total_columns: int, worker_position: int, total_workers: int) -> int:
    if total_workers == 0:
        return 0
    columns_per_worker = total_columns // total_workers
    remainder = total_columns % total_workers
    if worker_position < remainder:
        return worker_position * (columns_per_worker + 1)
    return remainder * (columns_per_worker + 1) + (worker_position - remainder) * columns_per_worker


def partition_columns(total_columns: int, worker_position: int, total_workers: int) -> int:
    if total_workers == 0:
        return total_columns
    columns_per_worker = total_columns // total_workers
    remainder = total_columns % total_workers
    if worker_position < remainder:
        return columns_per_worker + 1
    return columns_per_worker


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


def tensor_to_f32_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a real HF Llama-family model and convert it to MeshNet shards.")
    parser.add_argument("--repo-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--model-id", default="tinyllama-1.1b-chat-v1.0")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=2)
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
    kv_hidden_dim = num_kv_heads * head_dim

    model_size_bytes = os.path.getsize(model_path)
    save_json(
        out_dir / "model.json",
        {
            "model_id": args.model_id,
            "tensor_parallelism_dim": hidden_dim,
            "total_model_bytes": model_size_bytes,
            "tokenizer_file": "tokenizer.json",
            "tokenizer_config_file": "tokenizer_config.json",
        },
    )

    source = load_file(model_path, device="cpu")
    embedding = tensor_to_f32_numpy(source["model.embed_tokens.weight"])
    final_norm = tensor_to_f32_numpy(source["model.norm.weight"])
    lm_head_name = "lm_head.weight" if "lm_head.weight" in source else "model.embed_tokens.weight"
    lm_head = tensor_to_f32_numpy(source[lm_head_name].transpose(0, 1).contiguous())

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

    for worker_position in range(args.workers):
        q_start = partition_start(hidden_dim, worker_position, args.workers)
        q_cols = partition_columns(hidden_dim, worker_position, args.workers)
        kv_start = partition_start(kv_hidden_dim, worker_position, args.workers)
        kv_cols = partition_columns(kv_hidden_dim, worker_position, args.workers)
        mlp_start = partition_start(intermediate_size, worker_position, args.workers)
        mlp_cols = partition_columns(intermediate_size, worker_position, args.workers)

        tensors: dict[str, np.ndarray] = {
            "embedding": embedding,
            "final_norm": final_norm,
            "lm_head": lm_head,
        }

        for layer_idx in range(num_layers):
            names = tensor_name_map(layer_idx)
            q_proj = source[names["w_q"]]
            k_proj = source[names["w_k"]]
            v_proj = source[names["w_v"]]
            o_proj = source[names["w_o"]]
            up_proj = source[names["w_up"]]
            gate_proj = source[names["w_gate"]]
            down_proj = source[names["w_down"]]

            tensors[f"layers.{layer_idx}.w_q"] = tensor_to_f32_numpy(
                q_proj[q_start : q_start + q_cols, :].transpose(0, 1).contiguous()
            )
            tensors[f"layers.{layer_idx}.w_k"] = tensor_to_f32_numpy(
                k_proj[kv_start : kv_start + kv_cols, :].transpose(0, 1).contiguous()
            )
            tensors[f"layers.{layer_idx}.w_v"] = tensor_to_f32_numpy(
                v_proj[kv_start : kv_start + kv_cols, :].transpose(0, 1).contiguous()
            )
            tensors[f"layers.{layer_idx}.w_o"] = tensor_to_f32_numpy(
                o_proj[:, q_start : q_start + q_cols].transpose(0, 1).contiguous()
            )
            tensors[f"layers.{layer_idx}.w_up"] = tensor_to_f32_numpy(
                up_proj[mlp_start : mlp_start + mlp_cols, :].transpose(0, 1).contiguous()
            )
            tensors[f"layers.{layer_idx}.w_gate"] = tensor_to_f32_numpy(
                gate_proj[mlp_start : mlp_start + mlp_cols, :].transpose(0, 1).contiguous()
            )
            tensors[f"layers.{layer_idx}.w_down"] = tensor_to_f32_numpy(
                down_proj[:, mlp_start : mlp_start + mlp_cols].transpose(0, 1).contiguous()
            )
            tensors[f"layers.{layer_idx}.attn_norm"] = tensor_to_f32_numpy(source[names["attn_norm"]])
            tensors[f"layers.{layer_idx}.mlp_norm"] = tensor_to_f32_numpy(source[names["mlp_norm"]])

        shard_path = out_dir / f"shard-{worker_position}-of-{args.workers}.safetensors"
        shard_metadata = {
            **metadata,
            "mesh.worker_position": str(worker_position),
            "mesh.total_workers": str(args.workers),
        }
        save_file(tensors, str(shard_path), metadata=shard_metadata)

        digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        save_json(
            out_dir / f"shard-{worker_position}-of-{args.workers}.manifest.json",
            {
                "model_id": args.model_id,
                "worker_position": worker_position,
                "total_workers": args.workers,
                "expected_sha256": digest,
            },
        )

        print(f"wrote {shard_path}")


if __name__ == "__main__":
    main()
