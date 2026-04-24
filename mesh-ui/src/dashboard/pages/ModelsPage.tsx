import { EmptyState, MutationButton, Stat, StatRow, StatusBadge } from '../primitives'
import { ProgressBar, chartColors } from '../charts'
import { formatBytes } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

export function ModelsPage({ controller }: DashboardPageProps) {
  const selectedModel = controller.selectedModel
  const models = controller.networkModels
  const ready = models.filter((model) => model.artifactReady).length
  const totalParticipants = models.reduce((sum, model) => sum + model.participantCount, 0)
  const totalBytes = models.reduce((sum, model) => sum + (model.totalModelBytes ?? 0), 0)
  const joinState = controller.mutationState['ring:join']?.state ?? 'idle'

  return (
    <div className="dashboard-stack">
      <StatRow>
        <Stat label="Models" value={models.length} accent="cool" caption={`${ready} ready · ${models.length - ready} partial`} />
        <Stat label="Participants" value={totalParticipants} accent="accent" caption="Sum of devices serving any model." />
        <Stat
          label="Cataloged size"
          value={totalBytes > 0 ? formatBytes(totalBytes) : '—'}
          accent="violet"
          caption="Aggregated model artifact bytes."
        />
        <Stat
          label="Local shard"
          value={selectedModel?.loadedLocalShard ? 'loaded' : '—'}
          accent={selectedModel?.loadedLocalShard ? 'accent' : 'neutral'}
          caption={
            selectedModel?.shardStatus ? selectedModel.shardStatus : 'No local shard for selected model.'
          }
        />
      </StatRow>

      <div className="ms-credit-grid">
        <section className="panel dashboard-panel">
          <div className="ms-section-head">
            <div className="ms-section-head-copy">
              <div className="eyebrow">Inventory</div>
              <h3>Network models</h3>
              <p>Model artifacts mapped to the selected network and their readiness state.</p>
            </div>
          </div>
          {models.length === 0 ? (
            <EmptyState title="No models registered for this network" />
          ) : (
            <div className="ms-assignments">
              <div className="ms-assignments-head">
                <span>Model</span>
                <span>Ready</span>
                <span className="num">Participants</span>
                <span>Providers</span>
                <span>Size</span>
              </div>
              {models.map((model) => (
                <button
                  key={model.id}
                  type="button"
                  className="ms-assignments-row"
                  style={{ background: 'transparent', border: 'none', textAlign: 'left', width: '100%', cursor: 'pointer' }}
                  onClick={() => controller.setSelectedModelId(model.id)}
                >
                  <span>
                    <strong style={{ color: chartColors.textStrong, display: 'block', fontSize: 12 }}>
                      {model.id}
                    </strong>
                    <small style={{ color: chartColors.textMuted, fontSize: 10 }}>
                      {model.manifestCount} manifests · {model.weightsCount} weights
                    </small>
                  </span>
                  <span>
                    <StatusBadge status={model.artifactReady ? 'ok' : 'warn'}>
                      {model.artifactReady ? 'ready' : 'partial'}
                    </StatusBadge>
                  </span>
                  <span className="num">{model.participantCount}</span>
                  <span style={{ fontSize: 11 }}>
                    {model.providerCompatibility.length > 0 ? model.providerCompatibility.join(', ') : '—'}
                  </span>
                  <span>{model.totalModelBytes ? formatBytes(model.totalModelBytes) : '—'}</span>
                </button>
              ))}
            </div>
          )}
        </section>

        <section className="panel dashboard-panel">
          {selectedModel ? (
            <>
              <div className="ms-section-head">
                <div className="ms-section-head-copy">
                  <div className="eyebrow">Selected model</div>
                  <h3>{selectedModel.id}</h3>
                </div>
                <div className="ms-section-head-actions">
                  <MutationButton
                    variant="primary"
                    size="sm"
                    state={joinState}
                    onClick={() => void controller.joinRing(selectedModel.id)}
                    pendingLabel="Joining…"
                  >
                    Join ring
                  </MutationButton>
                </div>
              </div>
              <div className="ms-drawer-lifecycle">
                <div className="cell">
                  <span>Artifact</span>
                  <strong>
                    <StatusBadge status={selectedModel.artifactReady ? 'ok' : 'warn'}>
                      {selectedModel.artifactReady ? 'ready' : 'partial'}
                    </StatusBadge>
                  </strong>
                </div>
                <div className="cell">
                  <span>Tokenizer</span>
                  <strong>
                    <StatusBadge status={selectedModel.tokenizerReady ? 'ok' : 'warn'}>
                      {selectedModel.tokenizerReady ? 'ready' : 'missing'}
                    </StatusBadge>
                  </strong>
                </div>
                <div className="cell">
                  <span>Tensor dim</span>
                  <strong>{selectedModel.tensorParallelismDim ?? '—'}</strong>
                </div>
                <div className="cell">
                  <span>Total size</span>
                  <strong>{selectedModel.totalModelBytes ? formatBytes(selectedModel.totalModelBytes) : '—'}</strong>
                </div>
              </div>
              <ProgressBar
                current={selectedModel.weightsCount}
                total={Math.max(selectedModel.manifestCount, selectedModel.weightsCount, 1)}
                color={chartColors.cool}
                label={
                  <>
                    <span>shard coverage</span>
                    <span>
                      {selectedModel.weightsCount} / {selectedModel.manifestCount}
                    </span>
                  </>
                }
                caption={`${selectedModel.providerCompatibility.length} compatible providers`}
              />
              {selectedModel.localShardRange ? (
                <div className="ms-drawer-lifecycle" style={{ marginTop: 12 }}>
                  <div className="cell">
                    <span>Local shard range</span>
                    <strong>
                      {selectedModel.localShardRange.start}–{selectedModel.localShardRange.end}
                    </strong>
                  </div>
                  <div className="cell">
                    <span>Local memory</span>
                    <strong>{selectedModel.localMemoryBytes ? formatBytes(selectedModel.localMemoryBytes) : '—'}</strong>
                  </div>
                </div>
              ) : null}
            </>
          ) : (
            <EmptyState title="No model selected" hint="Pick a model from the inventory." />
          )}
        </section>
      </div>
    </div>
  )
}
