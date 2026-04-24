import { useMemo, useState } from 'react'
import {
  BulletList,
  ChartTooltipFoot,
  ChartTooltipHeader,
  ChartTooltipRow,
  ProgressBar,
  Sparkline,
  chartColors,
  formatChartNumber,
} from '../charts'
import { Drawer, EmptyState, MutationButton, Stat, StatRow, StatusBadge } from '../primitives'
import { formatDecimal, formatInteger, formatLatency } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'
import type { JobRecord } from '../../domain/dashboard'

const ACTIVE_STATUSES = new Set(['running', 'acknowledged', 'dispatched'])
const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled'])

function shortId(id: string): string {
  return id.length > 12 ? id.slice(0, 12) : id
}

function tokenProgress(current: number, total: number): number {
  if (total <= 0) return 0
  return Math.max(0, Math.min(100, Math.round((current / total) * 100)))
}

export function JobsPage({ controller }: DashboardPageProps) {
  const [drawerOpen, setDrawerOpen] = useState(true)
  const selectedJob = controller.selectedJob
  const jobs = controller.networkJobs

  const counts = useMemo(() => {
    let running = 0
    let completed = 0
    let failed = 0
    let totalReserved = 0
    for (const job of jobs) {
      if (ACTIVE_STATUSES.has(job.status)) running += 1
      if (job.status === 'completed') completed += 1
      if (job.status === 'failed' || job.status === 'cancelled') failed += 1
      totalReserved += job.reservedCredits
    }
    return { running, completed, failed, totalReserved }
  }, [jobs])

  const runState = controller.mutationState['job:run']?.state ?? 'idle'
  const cancelState = selectedJob
    ? controller.mutationState[`job:cancel:${selectedJob.id}`]?.state ?? 'idle'
    : 'idle'
  const refreshState = controller.mutationState['refresh']?.state ?? 'idle'

  const quote = controller.quote
  const draftValid = controller.jobDraft.prompt.trim().length > 0 && controller.jobDraft.modelId.length > 0
  const submitDisabled = !draftValid || quote?.feasible === false

  const openDrawerForJob = (jobId: string) => {
    controller.setSelectedJobId(jobId)
    setDrawerOpen(true)
  }

  return (
    <div className="dashboard-stack">
      <StatRow>
        <Stat
          label="Running"
          value={formatInteger(counts.running)}
          accent="accent"
          caption={counts.running > 0 ? 'Live workers reserving credits.' : 'No active reservations.'}
        />
        <Stat
          label="Completed"
          value={formatInteger(counts.completed)}
          accent="cool"
          caption="Settled assignment ledger."
        />
        <Stat
          label="Failed / cancelled"
          value={formatInteger(counts.failed)}
          accent={counts.failed > 0 ? 'danger' : 'neutral'}
          caption="Reservations released without success."
        />
        <Stat
          label="Reserved (network)"
          value={formatChartNumber(counts.totalReserved)}
          accent="warm"
          caption={`Across ${jobs.length} jobs in this view.`}
        />
      </StatRow>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">Submit job</div>
            <h3>Dispatch new inference work</h3>
            <p>Live cost preview is sourced from the model size factor and your device&apos;s available credits.</p>
          </div>
        </div>
        <div className="ms-job-submit-grid">
          <div className="ms-job-submit">
            <textarea
              placeholder="Prompt"
              value={controller.jobDraft.prompt}
              onChange={(event) => controller.setJobDraft({ prompt: event.target.value })}
            />
            <div className="ms-field-grid">
              <label className="ms-field">
                <span>Model</span>
                <select
                  value={controller.jobDraft.modelId}
                  onChange={(event) => controller.setJobDraft({ modelId: event.target.value })}
                >
                  <option value="">Select a model</option>
                  {controller.networkModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.id}
                    </option>
                  ))}
                </select>
              </label>
              <label className="ms-field">
                <span>Max tokens</span>
                <input
                  type="number"
                  min={1}
                  value={controller.jobDraft.maxTokens}
                  onChange={(event) => controller.setJobDraft({ maxTokens: Number(event.target.value) })}
                />
              </label>
              <label className="ms-field">
                <span>Temperature</span>
                <input
                  type="number"
                  step="0.1"
                  min={0}
                  max={2}
                  value={controller.jobDraft.temperature}
                  onChange={(event) => controller.setJobDraft({ temperature: Number(event.target.value) })}
                />
              </label>
              <label className="ms-field">
                <span>Top-p</span>
                <input
                  type="number"
                  step="0.05"
                  min={0}
                  max={1}
                  value={controller.jobDraft.topP}
                  onChange={(event) => controller.setJobDraft({ topP: Number(event.target.value) })}
                />
              </label>
            </div>
          </div>
          <div className={quote && !quote.feasible ? 'ms-job-quote infeasible' : 'ms-job-quote'}>
            <div className="ms-job-quote-head">
              <div>
                <small>Estimated reservation</small>
                <div>
                  <strong>{quote ? formatDecimal(quote.totalCreditsCap, 2) : '—'}</strong>
                </div>
              </div>
              {quote ? (
                <StatusBadge status={quote.feasible ? 'ok' : 'fail'}>
                  {quote.feasible ? 'feasible' : 'insufficient'}
                </StatusBadge>
              ) : null}
            </div>
            <div className="ms-job-quote-grid">
              <span>
                Prompt tokens
                <strong>{quote ? formatInteger(quote.promptTokens) : '—'}</strong>
              </span>
              <span>
                Completion cap
                <strong>{quote ? formatInteger(quote.availableCompletionTokens) : '—'}</strong>
              </span>
              <span>
                Prompt credits
                <strong>{quote ? formatDecimal(quote.promptCredits, 2) : '—'}</strong>
              </span>
              <span>
                Completion credits
                <strong>{quote ? formatDecimal(quote.completionCreditsCap, 2) : '—'}</strong>
              </span>
              <span>
                Model size factor
                <strong>{quote ? formatDecimal(quote.modelSizeFactor, 2) : '—'}</strong>
              </span>
              <span>
                Available credits
                <strong>{quote ? formatDecimal(quote.deviceAvailableCredits, 2) : '—'}</strong>
              </span>
            </div>
            <ProgressBar
              current={quote?.totalCreditsCap ?? 0}
              total={quote?.deviceAvailableCredits ?? 1}
              color={quote && !quote.feasible ? chartColors.danger : chartColors.settled}
              caption={
                quote
                  ? quote.feasible
                    ? `Will reserve ${formatDecimal(quote.totalCreditsCap, 2)} of ${formatDecimal(quote.deviceAvailableCredits, 2)} available.`
                    : (quote.reason ?? 'Insufficient credits to dispatch this job.')
                  : 'Type a prompt to fetch a live quote.'
              }
            />
            <MutationButton
              variant="primary"
              state={runState}
              disabled={submitDisabled}
              onClick={() => void controller.runJob()}
              pendingLabel="Submitting…"
            >
              Submit job
            </MutationButton>
          </div>
        </div>
      </section>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">Jobs</div>
            <h3>Reservation and runtime ledger</h3>
            <p>Click a row to open its assignment-level detail in the side drawer.</p>
          </div>
          <div className="ms-section-head-actions">
            <input
              className="dashboard-search"
              placeholder="Search id, model, submitter"
              value={controller.jobSearch}
              onChange={(event) => controller.setJobSearch(event.target.value)}
            />
            <MutationButton
              variant="ghost"
              size="sm"
              state={refreshState}
              onClick={() => void controller.refresh()}
              pendingLabel="Refreshing…"
            >
              Refresh
            </MutationButton>
          </div>
        </div>
        {jobs.length === 0 ? (
          <EmptyState
            title="No jobs in this network yet"
            hint="Submit a prompt above and the dispatched job will appear here."
          />
        ) : (
          <div className="ms-jobs-table">
            <div className="ms-jobs-table-head">
              <span>ID</span>
              <span>Status</span>
              <span>Model</span>
              <span className="num">Reserved</span>
              <span className="num">Settled</span>
              <span className="num">Released</span>
              <span>Tokens</span>
            </div>
            {jobs.map((job) => {
              const isSelected = job.id === selectedJob?.id
              const isActive = ACTIVE_STATUSES.has(job.status)
              return (
                <button
                  key={job.id}
                  type="button"
                  className={isSelected ? 'ms-jobs-table-row active' : 'ms-jobs-table-row'}
                  onClick={() => openDrawerForJob(job.id)}
                >
                  <span className="mono">{shortId(job.id)}</span>
                  <span>
                    <StatusBadge status={job.status} />
                  </span>
                  <span>{job.modelId}</span>
                  <span className="num">{formatDecimal(job.reservedCredits, 2)}</span>
                  <span className="num">{formatDecimal(job.settledCredits, 2)}</span>
                  <span className="num">{formatDecimal(job.releasedCredits, 2)}</span>
                  <span className="progress-cell">
                    {isActive && job.availableCompletionTokens > 0 ? (
                      <ProgressBar
                        current={job.accountedCompletionTokens}
                        total={job.availableCompletionTokens}
                        height={6}
                      />
                    ) : (
                      <ProgressBar
                        current={job.completionTokens}
                        total={Math.max(job.availableCompletionTokens, job.completionTokens, 1)}
                        height={6}
                        color={
                          TERMINAL_STATUSES.has(job.status) && job.status !== 'completed'
                            ? chartColors.danger
                            : chartColors.cool
                        }
                      />
                    )}
                    <small>
                      {formatInteger(
                        isActive ? job.accountedCompletionTokens : job.completionTokens,
                      )}
                      {' / '}
                      {formatInteger(Math.max(job.availableCompletionTokens, job.completionTokens))}
                    </small>
                  </span>
                </button>
              )
            })}
          </div>
        )}
      </section>

      <Drawer
        open={drawerOpen && selectedJob !== undefined}
        onClose={() => setDrawerOpen(false)}
        eyebrow={selectedJob ? selectedJob.modelId : 'Job detail'}
        title={selectedJob ? <span className="mono">{selectedJob.id}</span> : null}
        actions={
          selectedJob ? (
            <>
              {ACTIVE_STATUSES.has(selectedJob.status) ? (
                <MutationButton
                  variant="danger"
                  size="sm"
                  state={cancelState}
                  onClick={() => void controller.cancelJob(selectedJob.id)}
                  pendingLabel="Cancelling…"
                >
                  Cancel
                </MutationButton>
              ) : null}
            </>
          ) : null
        }
      >
        {selectedJob ? <JobDrawerContent job={selectedJob} /> : null}
      </Drawer>
    </div>
  )
}

function JobDrawerContent({ job }: { job: JobRecord }) {
  const outstanding = Math.max(
    0,
    job.reservedCredits - job.settledCredits - job.releasedCredits,
  )
  const tokenPct = tokenProgress(job.accountedCompletionTokens, job.availableCompletionTokens)
  const isActive = ACTIVE_STATUSES.has(job.status)

  const bulletRows = [
    {
      id: job.id,
      label: shortId(job.id),
      reservedCap: job.reservedCredits || 1,
      settled: job.settledCredits,
      released: job.releasedCredits,
      outstanding,
      ...(job.completionTokens > 0 && job.modelSizeFactor > 0
        ? { liveMarker: job.completionTokens * job.modelSizeFactor }
        : {}),
    },
  ]

  return (
    <>
      <section className="ms-drawer-section">
        <h4>Status</h4>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <StatusBadge status={job.status} size="md" />
          <span style={{ color: chartColors.text, fontSize: 12 }}>
            Submitted by <strong style={{ color: chartColors.textStrong }}>{job.submittedByName}</strong>
          </span>
          <span style={{ color: chartColors.text, fontSize: 12 }}>
            {job.ringWorkerCount} workers
          </span>
          <span style={{ color: chartColors.text, fontSize: 12 }}>
            {job.executionTimeMs > 0 ? formatLatency(job.executionTimeMs) : 'n/a'}
          </span>
        </div>
      </section>

      <section className="ms-drawer-section">
        <h4>Reservation lifecycle</h4>
        <div className="ms-drawer-lifecycle">
          <div className="cell">
            <span>Reserved</span>
            <strong>{formatDecimal(job.reservedCredits, 2)}</strong>
          </div>
          <div className="cell">
            <span>Settled</span>
            <strong style={{ color: chartColors.settled }}>{formatDecimal(job.settledCredits, 2)}</strong>
          </div>
          <div className="cell">
            <span>Released</span>
            <strong style={{ color: chartColors.released }}>{formatDecimal(job.releasedCredits, 2)}</strong>
          </div>
          <div className="cell">
            <span>Outstanding</span>
            <strong style={{ color: chartColors.outstanding }}>{formatDecimal(outstanding, 2)}</strong>
          </div>
        </div>
        <div className="ms-chart-card">
          <BulletList
            rows={bulletRows}
            labelWidth={92}
            renderTooltip={(row) => (
              <>
                <ChartTooltipHeader>{row.label}</ChartTooltipHeader>
                <ChartTooltipRow color={chartColors.settled} label="settled" value={formatDecimal(row.settled, 2)} />
                <ChartTooltipRow color={chartColors.released} label="released" value={formatDecimal(row.released, 2)} />
                <ChartTooltipRow color={chartColors.outstanding} label="outstanding" value={formatDecimal(row.outstanding, 2)} />
                <ChartTooltipRow color={chartColors.reserved} label="reserved cap" value={formatDecimal(row.reservedCap, 2)} outlined />
                <ChartTooltipFoot>
                  Net consumed {formatDecimal(job.settledCredits - job.releasedCredits, 2)}
                </ChartTooltipFoot>
              </>
            )}
          />
        </div>
      </section>

      <section className="ms-drawer-section">
        <h4>Token meter</h4>
        <ProgressBar
          current={isActive ? job.accountedCompletionTokens : job.completionTokens}
          total={job.availableCompletionTokens || job.completionTokens || 1}
          color={isActive ? chartColors.settled : chartColors.cool}
          label={
            <>
              <span>tokens accounted</span>
              <span>
                {formatInteger(isActive ? job.accountedCompletionTokens : job.completionTokens)}
                {' / '}
                {formatInteger(job.availableCompletionTokens)}
              </span>
            </>
          }
          caption={`${tokenPct}% of available cap · model size factor ${formatDecimal(job.modelSizeFactor, 2)}`}
        />
      </section>

      <section className="ms-drawer-section">
        <h4>Assignments · {job.assignments.length}</h4>
        {job.assignments.length === 0 ? (
          <EmptyState title="No assignments yet" hint="Workers will appear once the control-plane dispatches the job." />
        ) : (
          <div className="ms-assignments">
            <div className="ms-assignments-head">
              <span>Worker</span>
              <span>Status</span>
              <span>Shard</span>
              <span className="num">Tokens</span>
              <span className="num">Credits</span>
            </div>
            {job.assignments.map((assignment) => {
              const sparkData =
                assignment.reportedCompletionTokens > 0
                  ? Array.from({ length: 8 }, (_, i) =>
                      Math.max(0, assignment.reportedCompletionTokens * ((i + 1) / 8)),
                    )
                  : [0, 0, 0, 0, 0]
              return (
                <div key={assignment.assignmentId} className="ms-assignments-row">
                  <span>
                    <strong style={{ color: chartColors.textStrong, display: 'block', fontSize: 12 }}>
                      {assignment.deviceName}
                    </strong>
                    <small style={{ color: chartColors.textMuted, fontSize: 10.5 }}>
                      {assignment.executionProvider ?? 'unknown provider'}
                    </small>
                  </span>
                  <span>
                    <StatusBadge status={assignment.status} />
                  </span>
                  <span>
                    {assignment.shardColumnStart != null
                      ? `${assignment.shardColumnStart}–${assignment.shardColumnEnd ?? '?'}`
                      : '—'}
                  </span>
                  <span className="num">
                    <Sparkline
                      data={sparkData}
                      width={64}
                      height={18}
                      renderTooltip={(_, value) => (
                        <>
                          <ChartTooltipHeader>{assignment.deviceName}</ChartTooltipHeader>
                          <ChartTooltipRow
                            color={chartColors.settled}
                            label="reported"
                            value={formatInteger(Math.round(value))}
                          />
                        </>
                      )}
                    />
                    <div style={{ marginTop: 2 }}>{formatInteger(assignment.reportedCompletionTokens)}</div>
                  </span>
                  <span className="num">
                    {assignment.creditsEarned != null ? formatDecimal(assignment.creditsEarned, 2) : '—'}
                  </span>
                </div>
              )
            })}
          </div>
        )}
      </section>

      {job.creditPolicy ? (
        <section className="ms-drawer-section">
          <h4>Credit policy breakdown</h4>
          <div className="ms-chart-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
              <span style={{ fontSize: 11, color: chartColors.text, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Total job credit budget
              </span>
              <strong style={{ color: chartColors.textStrong, fontVariantNumeric: 'tabular-nums' }}>
                {formatDecimal(job.creditPolicy.jobCreditBudget, 2)}
              </strong>
            </div>
            <div className="ms-assignments">
              <div className="ms-assignments-head">
                <span>Device</span>
                <span className="num">Credits</span>
                <span className="num">Throughput</span>
                <span className="num">Pressure</span>
                <span className="num">Share</span>
              </div>
              {job.creditPolicy.assignments.map((entry) => (
                <div key={entry.deviceId} className="ms-assignments-row">
                  <span className="mono" style={{ fontSize: 11 }}>{entry.deviceId.slice(0, 12)}</span>
                  <span className="num">{formatDecimal(entry.credits, 2)}</span>
                  <span className="num">×{formatDecimal(entry.throughputMultiplier, 2)}</span>
                  <span className="num">×{formatDecimal(entry.resourcePressureMultiplier, 2)}</span>
                  <span className="num">{formatDecimal(entry.normalizedContributionShare * 100, 1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      ) : null}

      {job.error ? (
        <section className="ms-drawer-section">
          <h4>Error</h4>
          <div className="ms-empty" style={{ borderColor: 'rgba(255,107,107,0.3)' }}>
            <strong style={{ color: '#ff8d86' }}>{job.error}</strong>
          </div>
        </section>
      ) : null}
    </>
  )
}
