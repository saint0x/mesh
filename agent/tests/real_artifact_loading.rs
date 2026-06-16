use agent::model_assets::probe_local_production_artifact_materialization;
use agent::provider::{
    default_execution_provider, detect_execution_providers, resolve_requested_provider,
    set_selected_execution_provider, BackendContractDescriptor, ExecutionProviderKind,
};

const DEFAULT_MODEL_ID: &str = "smollm2-135m-instruct";

fn enabled() -> bool {
    std::env::var("MESHNET_ENABLE_REAL_ARTIFACT_TEST")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn configure_real_artifact_provider() {
    let requested = std::env::var("MESHNET_REAL_ARTIFACT_PROVIDER")
        .ok()
        .and_then(|value| ExecutionProviderKind::from_str(&value));
    let providers = detect_execution_providers();
    let provider = resolve_requested_provider(
        requested.or_else(|| Some(default_execution_provider(&providers))),
        &providers,
    )
    .expect("resolve real artifact execution provider");
    let contract = BackendContractDescriptor::for_provider(provider);
    assert!(
        contract.supports_production_serving(),
        "real artifact test requires a production-serving provider, got {}",
        provider.as_str()
    );
    set_selected_execution_provider(provider).expect("initialize real artifact execution provider");
}

#[tokio::test]
async fn loads_real_artifact_shard_into_residency() {
    if !enabled() {
        eprintln!("skipping real artifact loading test; set MESHNET_ENABLE_REAL_ARTIFACT_TEST=1");
        return;
    }

    configure_real_artifact_provider();
    let preferred_model_id = std::env::var("MESHNET_REAL_ARTIFACT_MODEL_ID").ok();
    let probe = probe_local_production_artifact_materialization(
        preferred_model_id.as_deref().or(Some(DEFAULT_MODEL_ID)),
    )
    .await
    .expect("probe local production artifact materialization")
    .expect("expected at least one real model artifact set");

    assert!(probe.resident_bytes > 0);
}
