import { useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { JobRecord } from '../../domain/dashboard'
import { formatChartNumber, formatLatency } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

function truncateId(id: string): string {
  return id.split('-')[0] ?? id.slice(0, 8)
}

export function JobsPage({ controller }: DashboardPageProps) {
  const selectedJob = controller.selectedJob

  const chartData = useMemo(() => {
    return controller.networkJobs.slice(0, 12).map((job) => ({
      name: truncateId(job.id),
      runtime: job.executionTimeMs,
      status: job.status,
      workers: job.ringWorkerCount,
      tokens: job.completionTokens,
    })).reverse()
  }, [controller.networkJobs])

  return (
    <div className="dashboard-stack">
      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Dispatch timeline</div>
            <h3>Job execution across the mesh</h3>
          </div>
          <input
            className="dashboard-search"
            placeholder="Search job ID, model, or submitter"
            value={controller.jobSearch}
            onChange={(event) => controller.setJobSearch(event.target.value)}
          />
        </div>
        {chartData.length > 0 ? (
          <div className="chart-card">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -12 }} barSize={chartData.length > 8 ? 18 : 28}>
                <CartesianGrid
                  strokeDasharray="5 6"
                  stroke="var(--line)"
                  vertical={false}
                />
                <XAxis
                  dataKey="name"
                  tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tickFormatter={formatChartNumber}
                  tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  unit=" ms"
                />
                <Tooltip
                  contentStyle={{
                    background: 'var(--panel-elevated)',
                    border: '1px solid var(--line)',
                    borderRadius: 14,
                    color: 'var(--text-strong)',
                    fontSize: '0.82rem',
                  }}
                  formatter={(value) => [formatLatency(Number(value)), 'Runtime']}
                  labelFormatter={(label) => `Job ${String(label)}`}
                  cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                />
                <Bar dataKey="runtime" radius={[6, 6, 2, 2]} fill="rgba(102, 240, 192, 0.72)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : null}
      </section>

      <div className="dashboard-grid">
        <section className="panel dashboard-panel">
          <div className="dashboard-panel-head">
            <div>
              <div className="eyebrow">Inference jobs</div>
              <h3>Assignment-level execution detail</h3>
            </div>
          </div>
          <div className="dashboard-list">
            {controller.networkJobs.map((job: JobRecord) => (
              <button
                key={job.id}
                className={job.id === selectedJob?.id ? 'dashboard-list-item active' : 'dashboard-list-item'}
                onClick={() => controller.setSelectedJobId(job.id)}
              >
                <strong>{job.id}</strong>
                <span>{job.status} / {job.modelId} / {job.assignments.length} assignments</span>
              </button>
            ))}
          </div>
        </section>

        <section className="panel dashboard-panel">
          {selectedJob ? (
            <>
              <div className="dashboard-panel-head">
                <div>
                  <div className="eyebrow">Selected job</div>
                  <h3>{selectedJob.id}</h3>
                </div>
              </div>
              <div className="dashboard-detail-grid">
                <article><span>Status</span><strong>{selectedJob.status}</strong></article>
                <article><span>Tokens</span><strong>{selectedJob.completionTokens}</strong></article>
                <article><span>Workers</span><strong>{selectedJob.ringWorkerCount}</strong></article>
                <article><span>Submitted by</span><strong>{selectedJob.submittedByName}</strong></article>
                <article><span>Created</span><strong>{selectedJob.createdAt}</strong></article>
                <article><span>Runtime</span><strong>{selectedJob.executionTimeMs > 0 ? formatLatency(selectedJob.executionTimeMs) : 'n/a'}</strong></article>
              </div>

              <div className="dashboard-data-table">
                <div className="dashboard-data-head">
                  <span>Worker</span>
                  <span>Status</span>
                  <span>Shard range</span>
                  <span>Capacity units</span>
                  <span>Provider</span>
                  <span>Runtime</span>
                </div>
                {selectedJob.assignments.map((assignment) => (
                  <div key={assignment.assignmentId} className="dashboard-data-row">
                    <span className="row-primary">{assignment.deviceName}</span>
                    <span><span className={`status-badge ${assignment.status}`}>{assignment.status}</span></span>
                    <span>
                      {assignment.shardColumnStart !== null && assignment.shardColumnStart !== undefined
                        ? `${assignment.shardColumnStart} - ${assignment.shardColumnEnd ?? 'n/a'}`
                        : 'n/a'}
                    </span>
                    <span>{assignment.assignedCapacityUnits}</span>
                    <span>{assignment.executionProvider ?? 'n/a'}</span>
                    <span>{assignment.executionTimeMs > 0 ? formatLatency(assignment.executionTimeMs) : 'n/a'}</span>
                  </div>
                ))}
              </div>

              {selectedJob.error ? <div className="dashboard-empty">Job error: {selectedJob.error}</div> : null}
              {selectedJob.assignments.some((assignment) => assignment.failureReason) ? (
                <div className="dashboard-empty">
                  Participant failures:
                  {' '}
                  {selectedJob.assignments
                    .filter((assignment) => assignment.failureReason)
                    .map((assignment) => `${assignment.deviceName}: ${assignment.failureReason}`)
                    .join(' | ')}
                </div>
              ) : null}
            </>
          ) : (
            <div className="dashboard-empty">No jobs are available for this network yet.</div>
          )}
        </section>
      </div>
    </div>
  )
}
