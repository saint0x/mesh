#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.numpy import save_file


def load_model_metadata(model_dir: Path) -> tuple[dict, int, int, int]:
    model = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    hidden_dim = int(model["tensor_parallelism_dim"])
    num_heads = int(model["attention_head_count"])
    num_kv_heads = int(model["kv_head_count"])
    return model, hidden_dim, num_heads, num_kv_heads


def rewrite_manifests(model_dir: Path, model_id: str, shard_ranges: list[tuple[int, int]]) -> None:
    total_workers = len(shard_ranges)
    for worker_position, (column_start, column_end) in enumerate(shard_ranges):
        shard_path = model_dir / f"shard-{worker_position}-of-{total_workers}.safetensors"
        digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        manifest_path = model_dir / f"shard-{worker_position}-of-{total_workers}.manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "worker_position": worker_position,
                    "total_workers": total_workers,
                    "column_start": column_start,
                    "column_end": column_end,
                    "expected_sha256": digest,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def load_repo_config(repo_id: str | None) -> dict | None:
    if not repo_id:
        return None
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def build_metadata(
    existing: dict[str, str],
    model_id: str,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    total_workers: int,
    worker_position: int,
    column_start: int,
    column_end: int,
    layer_count: int,
    embedding_rows: int,
    local_mlp_width: int,
    repo_config: dict | None,
) -> dict[str, str]:
    metadata = dict(existing)
    metadata["mesh.model_id"] = model_id
    metadata["mesh.hidden_dim"] = str(hidden_dim)
    metadata["mesh.num_heads"] = str(num_heads)
    metadata["mesh.num_kv_heads"] = str(num_kv_heads)
    metadata["mesh.num_layers"] = metadata.get("mesh.num_layers") or str(
        (repo_config or {}).get("num_hidden_layers", layer_count)
    )
    metadata["mesh.vocab_size"] = metadata.get("mesh.vocab_size") or str((repo_config or {}).get("vocab_size", embedding_rows))
    metadata["mesh.intermediate_size"] = metadata.get("mesh.intermediate_size") or str(
        (repo_config or {}).get("intermediate_size", local_mlp_width * total_workers)
    )
    metadata["mesh.rms_norm_eps"] = metadata.get("mesh.rms_norm_eps") or str(
        (repo_config or {}).get("rms_norm_eps", 1e-5)
    )
    metadata["mesh.rope_base"] = metadata.get("mesh.rope_base") or str(
        (repo_config or {}).get("rope_theta", 10000.0)
    )
    metadata["mesh.worker_position"] = str(worker_position)
    metadata["mesh.total_workers"] = str(total_workers)
    metadata["mesh.column_start"] = str(column_start)
    metadata["mesh.column_end"] = str(column_end)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite an existing 2-way MeshNet model into GQA-aligned shards.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--repo-id")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    model, hidden_dim, num_heads, num_kv_heads = load_model_metadata(model_dir)
    repo_config = load_repo_config(args.repo_id)
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError(
            f"unsupported grouped-query attention geometry: num_heads={num_heads} num_kv_heads={num_kv_heads}"
        )

    shard0_path = model_dir / "shard-0-of-2.safetensors"
    shard1_path = model_dir / "shard-1-of-2.safetensors"
    if not shard0_path.exists() or not shard1_path.exists():
        raise FileNotFoundError("expected existing shard-0-of-2.safetensors and shard-1-of-2.safetensors")

    head_dim = hidden_dim // num_heads
    q_heads_per_kv_head = num_heads // num_kv_heads
    old_q_width = hidden_dim // 2
    old_kv_width = (num_kv_heads * head_dim) // 2
    new_q0_width = q_heads_per_kv_head * head_dim * (num_kv_heads - 1)
    new_kv0_width = head_dim * (num_kv_heads - 1)
    q_prefix_from_shard1 = new_q0_width - old_q_width
    kv_prefix_from_shard1 = new_kv0_width - old_kv_width
    shard_ranges = [(0, new_q0_width), (new_q0_width, hidden_dim)]
    if q_prefix_from_shard1 <= 0 or kv_prefix_from_shard1 <= 0:
        raise ValueError("model does not require an in-place shard rewrite")

    with safe_open(str(shard0_path), framework="np") as shard0, safe_open(str(shard1_path), framework="np") as shard1:
        layer_ids = sorted(
            {
                int(key.split(".")[1])
                for key in shard0.keys()
                if key.startswith("layers.") and key.endswith(".w_q")
            }
        )

        shard0_metadata = dict(shard0.metadata())
        shard1_metadata = dict(shard1.metadata())
        embedding_rows = int(shard0.get_tensor("embedding").shape[0])
        local_mlp_width = int(shard0.get_tensor("layers.0.w_up").shape[1])
        q_widths = [
            int(shard0.get_tensor("layers.0.w_q").shape[1]),
            int(shard1.get_tensor("layers.0.w_q").shape[1]),
        ]
        kv_widths = [
            int(shard0.get_tensor("layers.0.w_k").shape[1]),
            int(shard1.get_tensor("layers.0.w_k").shape[1]),
        ]
        target_q_widths = [new_q0_width, hidden_dim - new_q0_width]
        target_kv_widths = [new_kv0_width, (num_kv_heads * head_dim) - new_kv0_width]

        if q_widths == target_q_widths and kv_widths == target_kv_widths:
            rewrite_manifests(model_dir, model["model_id"], shard_ranges)
            return

        shard1_prefix_cache: dict[str, np.ndarray] = {}
        common0: dict[str, np.ndarray] = {
            "embedding": np.ascontiguousarray(shard0.get_tensor("embedding")),
            "final_norm": np.ascontiguousarray(shard0.get_tensor("final_norm")),
            "lm_head": np.ascontiguousarray(shard0.get_tensor("lm_head")),
        }
        common1: dict[str, np.ndarray] = {
            "embedding": np.ascontiguousarray(shard1.get_tensor("embedding")),
            "final_norm": np.ascontiguousarray(shard1.get_tensor("final_norm")),
            "lm_head": np.ascontiguousarray(shard1.get_tensor("lm_head")),
        }

        source_mode = None
        if q_widths == [old_q_width, old_q_width] and kv_widths == [old_kv_width, old_kv_width]:
            source_mode = "equal_split"
        elif q_widths == [hidden_dim, 0] and kv_widths == [num_kv_heads * head_dim, 0]:
            source_mode = "full_shard0"
        if source_mode is None:
            raise ValueError(
                f"unsupported source shard widths q={q_widths} kv={kv_widths}; refusing to rewrite artifacts"
            )

        for layer_idx in layer_ids:
            prefix = f"layers.{layer_idx}"
            common0[f"{prefix}.attn_norm"] = np.ascontiguousarray(shard0.get_tensor(f"{prefix}.attn_norm"))
            common0[f"{prefix}.mlp_norm"] = np.ascontiguousarray(shard0.get_tensor(f"{prefix}.mlp_norm"))
            common1[f"{prefix}.attn_norm"] = np.ascontiguousarray(shard1.get_tensor(f"{prefix}.attn_norm"))
            common1[f"{prefix}.mlp_norm"] = np.ascontiguousarray(shard1.get_tensor(f"{prefix}.mlp_norm"))
            if source_mode == "equal_split":
                shard1_prefix_cache[f"{prefix}.w_q"] = np.ascontiguousarray(
                    shard1.get_tensor(f"{prefix}.w_q")[:, :q_prefix_from_shard1]
                )
                shard1_prefix_cache[f"{prefix}.w_k"] = np.ascontiguousarray(
                    shard1.get_tensor(f"{prefix}.w_k")[:, :kv_prefix_from_shard1]
                )
                shard1_prefix_cache[f"{prefix}.w_v"] = np.ascontiguousarray(
                    shard1.get_tensor(f"{prefix}.w_v")[:, :kv_prefix_from_shard1]
                )
                shard1_prefix_cache[f"{prefix}.w_o"] = np.ascontiguousarray(
                    shard1.get_tensor(f"{prefix}.w_o")[:q_prefix_from_shard1, :]
                )

        rewritten_shard1 = dict(common1)
        for layer_idx in layer_ids:
            prefix = f"layers.{layer_idx}"
            if source_mode == "equal_split":
                rewritten_shard1[f"{prefix}.w_q"] = np.ascontiguousarray(
                    shard1.get_tensor(f"{prefix}.w_q")[:, q_prefix_from_shard1:]
                )
                rewritten_shard1[f"{prefix}.w_k"] = np.ascontiguousarray(
                    shard1.get_tensor(f"{prefix}.w_k")[:, kv_prefix_from_shard1:]
                )
                rewritten_shard1[f"{prefix}.w_v"] = np.ascontiguousarray(
                    shard1.get_tensor(f"{prefix}.w_v")[:, kv_prefix_from_shard1:]
                )
                rewritten_shard1[f"{prefix}.w_o"] = np.ascontiguousarray(
                    shard1.get_tensor(f"{prefix}.w_o")[q_prefix_from_shard1:, :]
                )
            else:
                rewritten_shard1[f"{prefix}.w_q"] = np.ascontiguousarray(
                    shard0.get_tensor(f"{prefix}.w_q")[:, new_q0_width:]
                )
                rewritten_shard1[f"{prefix}.w_k"] = np.ascontiguousarray(
                    shard0.get_tensor(f"{prefix}.w_k")[:, new_kv0_width:]
                )
                rewritten_shard1[f"{prefix}.w_v"] = np.ascontiguousarray(
                    shard0.get_tensor(f"{prefix}.w_v")[:, new_kv0_width:]
                )
                rewritten_shard1[f"{prefix}.w_o"] = np.ascontiguousarray(
                    shard0.get_tensor(f"{prefix}.w_o")[new_q0_width:, :]
                )
            rewritten_shard1[f"{prefix}.w_up"] = np.ascontiguousarray(shard1.get_tensor(f"{prefix}.w_up"))
            rewritten_shard1[f"{prefix}.w_gate"] = np.ascontiguousarray(shard1.get_tensor(f"{prefix}.w_gate"))
            rewritten_shard1[f"{prefix}.w_down"] = np.ascontiguousarray(shard1.get_tensor(f"{prefix}.w_down"))
        save_file(
            rewritten_shard1,
            str(shard1_path),
            metadata=build_metadata(
                shard1_metadata,
                model["model_id"],
                hidden_dim,
                num_heads,
                num_kv_heads,
                total_workers=2,
                worker_position=1,
                column_start=shard_ranges[1][0],
                column_end=shard_ranges[1][1],
                layer_count=len(layer_ids),
                embedding_rows=embedding_rows,
                local_mlp_width=local_mlp_width,
                repo_config=repo_config,
            ),
        )

        rewritten_shard0 = dict(common0)
        for layer_idx in layer_ids:
            prefix = f"layers.{layer_idx}"
            if source_mode == "equal_split":
                rewritten_shard0[f"{prefix}.w_q"] = np.ascontiguousarray(
                    np.concatenate(
                        [shard0.get_tensor(f"{prefix}.w_q"), shard1_prefix_cache[f"{prefix}.w_q"]],
                        axis=1,
                    )
                )
                rewritten_shard0[f"{prefix}.w_k"] = np.ascontiguousarray(
                    np.concatenate(
                        [shard0.get_tensor(f"{prefix}.w_k"), shard1_prefix_cache[f"{prefix}.w_k"]],
                        axis=1,
                    )
                )
                rewritten_shard0[f"{prefix}.w_v"] = np.ascontiguousarray(
                    np.concatenate(
                        [shard0.get_tensor(f"{prefix}.w_v"), shard1_prefix_cache[f"{prefix}.w_v"]],
                        axis=1,
                    )
                )
                rewritten_shard0[f"{prefix}.w_o"] = np.ascontiguousarray(
                    np.concatenate(
                        [shard0.get_tensor(f"{prefix}.w_o"), shard1_prefix_cache[f"{prefix}.w_o"]],
                        axis=0,
                    )
                )
            else:
                rewritten_shard0[f"{prefix}.w_q"] = np.ascontiguousarray(
                    shard0.get_tensor(f"{prefix}.w_q")[:, :new_q0_width]
                )
                rewritten_shard0[f"{prefix}.w_k"] = np.ascontiguousarray(
                    shard0.get_tensor(f"{prefix}.w_k")[:, :new_kv0_width]
                )
                rewritten_shard0[f"{prefix}.w_v"] = np.ascontiguousarray(
                    shard0.get_tensor(f"{prefix}.w_v")[:, :new_kv0_width]
                )
                rewritten_shard0[f"{prefix}.w_o"] = np.ascontiguousarray(
                    shard0.get_tensor(f"{prefix}.w_o")[:new_q0_width, :]
                )
            rewritten_shard0[f"{prefix}.w_up"] = np.ascontiguousarray(shard0.get_tensor(f"{prefix}.w_up"))
            rewritten_shard0[f"{prefix}.w_gate"] = np.ascontiguousarray(shard0.get_tensor(f"{prefix}.w_gate"))
            rewritten_shard0[f"{prefix}.w_down"] = np.ascontiguousarray(shard0.get_tensor(f"{prefix}.w_down"))
        save_file(
            rewritten_shard0,
            str(shard0_path),
            metadata=build_metadata(
                shard0_metadata,
                model["model_id"],
                hidden_dim,
                num_heads,
                num_kv_heads,
                total_workers=2,
                worker_position=0,
                column_start=shard_ranges[0][0],
                column_end=shard_ranges[0][1],
                layer_count=len(layer_ids),
                embedding_rows=embedding_rows,
                local_mlp_width=local_mlp_width,
                repo_config=repo_config,
            ),
        )

    rewrite_manifests(model_dir, model["model_id"], shard_ranges)


if __name__ == "__main__":
    main()
