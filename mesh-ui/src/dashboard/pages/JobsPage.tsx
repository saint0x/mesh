import type { JobRecord } from '../../domain/dashboard'
import { formatLatency } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

export function JobsPage({ controller }: DashboardPageProps) {
  const selectedJob = controller.selectedJob

  return (
    <div className="dashboard-grid">
      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Inference jobs</div>
            <h3>Assignment-level execution detail</h3>
          </div>
          <input
            className="dashboard-search"
            placeholder="Search job ID, model, or submitter"
            value={controller.jobSearch}
            onChange={(event) => controller.setJobSearch(event.target.value)}
          />
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
  )
}
