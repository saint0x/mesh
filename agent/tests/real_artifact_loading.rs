use agent::inference::forward_pass::SharedModelResidency;
use agent::inference::{ArtifactShardLoader, ShardLoader};
use agent::model::ShardAssignment;
use agent::model::ShardRegistry;
use agent::model_assets::{load_model_manifest, model_store_dir};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_MODEL_ID: &str = "smollm2-135m-instruct";

fn enabled() -> bool {
    std::env::var("MESHNET_ENABLE_REAL_ARTIFACT_TEST")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn discover_model_dir(root: &Path) -> Option<PathBuf> {
    if let Ok(model_id) = std::env::var("MESHNET_REAL_ARTIFACT_MODEL_ID") {
        let candidate = root.join(model_id);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }

    let preferred = root.join(DEFAULT_MODEL_ID);
    if preferred.is_dir() {
        return Some(preferred);
    }

    let entries = fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let has_manifest = fs::read_dir(&path)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .any(|item| {
                item.file_name().to_string_lossy().starts_with("shard-")
                    && item
                        .file_name()
                        .to_string_lossy()
                        .ends_with(".manifest.json")
            });
        if has_manifest {
            return Some(path);
        }
    }
    None
}

#[derive(Deserialize)]
struct RealShardArtifactManifest {
    worker_position: u32,
    total_workers: u32,
    column_start: u32,
    column_end: u32,
}

fn discover_assignment(model_id: &str, model_dir: &Path) -> Option<ShardAssignment> {
    let entries = fs::read_dir(model_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let filename = path.file_name()?.to_string_lossy().to_string();
        if !(filename.starts_with("shard-") && filename.ends_with(".manifest.json")) {
            continue;
        }
        let manifest: RealShardArtifactManifest =
            serde_json::from_slice(&fs::read(path).ok()?).ok()?;
        return Some(ShardAssignment::from_column_range(
            model_id.to_string(),
            manifest.worker_position,
            manifest.total_workers,
            manifest.column_start,
            manifest.column_end,
        ));
    }
    None
}

#[tokio::test]
async fn loads_real_artifact_shard_into_residency() {
    if !enabled() {
        eprintln!("skipping real artifact loading test; set MESHNET_ENABLE_REAL_ARTIFACT_TEST=1");
        return;
    }

    let store = model_store_dir();
    let model_dir =
        discover_model_dir(&store).expect("expected at least one real model artifact set");
    let model_id = model_dir
        .file_name()
        .expect("model dir name")
        .to_string_lossy()
        .to_string();
    let _manifest = load_model_manifest(&model_id).expect("load real model manifest");
    let assignment = discover_assignment(&model_id, &model_dir)
        .expect("discover shard assignment from real artifact manifest");

    let registry_root = tempfile::tempdir().expect("registry tempdir");
    let registry = ShardRegistry::new(registry_root.path().join("registry")).expect("registry");
    registry
        .assign_shard(assignment.clone())
        .await
        .expect("assign shard");

    let loader = ArtifactShardLoader::new(store);
    let weights = loader
        .load_shard(&model_id, &assignment, &registry)
        .await
        .expect("load real safetensors shard");
    let residency =
        SharedModelResidency::from_host(weights).expect("materialize real model residency");

    assert!(residency.resident_bytes() > 0);
}
