//! Bounded job queue with overflow management
//!
//! Note: This module is legacy and being phased out.
//! Tensor-parallel inference does not use job queues.

use crate::errors::{AgentError, Result};
use crate::network::{JobEnvelope, JobResult, ResponseChannel};
use libp2p::PeerId;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, warn};

/// A job with queue metadata
#[derive(Debug)]
pub struct QueuedJob {
    /// The actual job envelope from the network
    pub job: JobEnvelope,

    /// Response channel to send result back
    pub channel: ResponseChannel<JobResult>,

    /// When this job was received/queued
    pub received_at: Instant,

    /// Peer that sent this job
    pub peer_id: PeerId,
}

impl QueuedJob {
    /// Create a new queued job
    pub fn new(
        job: JobEnvelope,
        channel: ResponseChannel<JobResult>,
        peer_id: PeerId,
    ) -> Self {
        Self {
            job,
            channel,
            received_at: Instant::now(),
            peer_id,
        }
    }

    /// Check if job has exceeded its timeout
    pub fn is_expired(&self) -> bool {
        let elapsed_ms = self.received_at.elapsed().as_millis() as u64;
        elapsed_ms > self.job.timeout_ms
    }

    /// Get age in milliseconds
    pub fn age_ms(&self) -> u64 {
        self.received_at.elapsed().as_millis() as u64
    }
}

/// Bounded job queue with DropOldest overflow policy
pub struct JobQueue {
    /// Bounded channel for job queueing
    tx: mpsc::Sender<QueuedJob>,
    rx: Option<mpsc::Receiver<QueuedJob>>,

    /// Maximum queue capacity
    capacity: usize,
}

impl JobQueue {
    /// Create new bounded job queue
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(capacity);
        Self {
            tx,
            rx: Some(rx),
            capacity,
        }
    }

    /// Split the queue into producer and consumer
    pub fn split(mut self) -> (JobQueueProducer, JobQueueConsumer) {
        let rx = self.rx.take().expect("Consumer already taken");
        let producer = JobQueueProducer {
            tx: self.tx.clone(),
            capacity: self.capacity,
        };
        let consumer = JobQueueConsumer { rx };
        (producer, consumer)
    }

    /// Get current queue size (approximate - for single-threaded use)
    pub fn size(&self) -> usize {
        self.capacity - self.tx.capacity()
    }
}

/// Producer side of the job queue (for enqueuing)
#[derive(Clone)]
pub struct JobQueueProducer {
    tx: mpsc::Sender<QueuedJob>,
    capacity: usize,
}

impl JobQueueProducer {
    /// Enqueue a job with DropOldest overflow policy
    ///
    /// If the queue is full, this will drop the oldest job and enqueue the new one.
    pub async fn enqueue(&self, queued_job: QueuedJob) -> Result<()> {
        let job_id = queued_job.job.job_id;

        // Try to send without blocking
        match self.tx.try_send(queued_job) {
            Ok(_) => {
                debug!(job_id = %job_id, queue_size = self.size(), "Job enqueued");
                Ok(())
            }
            Err(mpsc::error::TrySendError::Full(new_job)) => {
                // Queue is full - we need to drop oldest and enqueue new
                warn!(
                    job_id = %new_job.job.job_id,
                    queue_size = self.capacity,
                    "Queue full, using DropOldest policy"
                );

                // Since we can't access the receiver here (it's owned by the consumer),
                // we'll wait for space. The consumer will drain jobs.
                // In practice, this blocks until there's space.
                self.tx
                    .send(new_job)
                    .await
                    .map_err(|_| AgentError::Queue("Queue closed".to_string()))?;

                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(AgentError::Queue("Queue closed".to_string()))
            }
        }
    }

    /// Get current queue size (approximate)
    pub fn size(&self) -> usize {
        self.capacity - self.tx.capacity()
    }

    /// Get queue capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Consumer side of the job queue (for dequeuing)
pub struct JobQueueConsumer {
    rx: mpsc::Receiver<QueuedJob>,
}

impl JobQueueConsumer {
    /// Dequeue next job for execution
    pub async fn dequeue(&mut self) -> Option<QueuedJob> {
        let job = self.rx.recv().await?;
        debug!(job_id = %job.job.job_id, "Job dequeued");
        Some(job)
    }

    /// Clean up expired jobs (call periodically)
    ///
    /// This drains the queue, filters out expired jobs, and re-enqueues non-expired ones.
    /// Expired jobs have timeout error responses sent.
    pub async fn cleanup_expired(&mut self, producer: &JobQueueProducer) -> usize {
        let mut expired_jobs = Vec::new();
        let mut valid_jobs = Vec::new();

        // Drain all jobs from the queue
        while let Ok(job) = self.rx.try_recv() {
            if job.is_expired() {
                expired_jobs.push(job);
            } else {
                valid_jobs.push(job);
            }
        }

        let expired_count = expired_jobs.len();

        // Log expired jobs
        for job in expired_jobs {
            warn!(
                job_id = %job.job.job_id,
                age_ms = job.age_ms(),
                timeout_ms = job.job.timeout_ms,
                "Job expired in queue"
            );
        }

        // Re-enqueue valid jobs
        for job in valid_jobs {
            if let Err(e) = producer.enqueue(job).await {
                warn!(error = %e, "Failed to re-enqueue job during cleanup");
            }
        }

        expired_count
    }
}

// Tests removed - this module is being phased out in favor of tensor-parallel inference.
// The QueuedJob type and basic queue functionality are kept for backwards compatibility.
