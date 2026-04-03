import type { ModelRecord } from '../../domain/dashboard'
import { formatBytes } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

export function ModelsPage({ controller }: DashboardPageProps) {
  const selectedModel = controller.selectedModel

  return (
    <div className="dashboard-grid">
      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Model inventory</div>
            <h3>Artifact readiness and network participation</h3>
          </div>
        </div>
        <div className="dashboard-list">
          {controller.networkModels.map((model: ModelRecord) => (
            <button
              key={model.id}
              className={model.id === selectedModel?.id ? 'dashboard-list-item active' : 'dashboard-list-item'}
              onClick={() => controller.setSelectedModelId(model.id)}
            >
              <strong>{model.id}</strong>
              <span>{model.artifactReady ? 'ready' : 'partial'} / {model.participantCount} participants / {model.providerCompatibility.join(', ') || 'no compatible providers reported'}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="panel dashboard-panel">
        {selectedModel ? (
          <>
            <div className="dashboard-panel-head">
              <div>
                <div className="eyebrow">Selected model</div>
                <h3>{selectedModel.id}</h3>
              </div>
            </div>
            <div className="dashboard-detail-grid">
              <article><span>Artifact ready</span><strong>{selectedModel.artifactReady ? 'yes' : 'no'}</strong></article>
              <article><span>Tokenizer ready</span><strong>{selectedModel.tokenizerReady ? 'yes' : 'no'}</strong></article>
              <article><span>Tensor dim</span><strong>{selectedModel.tensorParallelismDim ?? 'n/a'}</strong></article>
              <article><span>Total bytes</span><strong>{selectedModel.totalModelBytes ? formatBytes(selectedModel.totalModelBytes) : 'n/a'}</strong></article>
            </div>

            <div className="snapshot-grid">
              <article>
                <span>Manifest coverage</span>
                <strong>{selectedModel.manifestCount}</strong>
                <small>{selectedModel.weightsCount} weights files found under ~/.meshnet/models.</small>
              </article>
              <article>
                <span>Provider compatibility</span>
                <strong>{selectedModel.providerCompatibility.join(', ') || 'n/a'}</strong>
                <small>Union of available providers reported by participating devices.</small>
              </article>
              <article>
                <span>Local shard status</span>
                <strong>{selectedModel.shardStatus ?? 'not loaded locally'}</strong>
                <small>{selectedModel.loadedLocalShard ? 'Shard is loaded on this machine.' : 'No loaded local shard entry.'}</small>
              </article>
              <article>
                <span>Local shard range</span>
                <strong>{selectedModel.localShardRange ? `${selectedModel.localShardRange.start} - ${selectedModel.localShardRange.end}` : 'n/a'}</strong>
                <small>{selectedModel.localMemoryBytes ? formatBytes(selectedModel.localMemoryBytes) : 'No local shard memory recorded'}</small>
              </article>
            </div>
          </>
        ) : (
          <div className="dashboard-empty">No model inventory is attached to this network yet.</div>
        )}
      </section>
    </div>
  )
}
