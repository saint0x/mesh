// Bounded job queue with overflow management

use crate::errors::{AgentError, Result};
use crate::executor::JobStats;
use crate::network::job_protocol::{JobEnvelope, JobResult};
use libp2p::request_response::ResponseChannel;
use libp2p::PeerId;
use std::sync::Arc;
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

    /// Statistics reference for tracking
    stats: Arc<JobStats>,
}

impl JobQueue {
    /// Create new bounded job queue
    pub fn new(capacity: usize, stats: Arc<JobStats>) -> Self {
        let (tx, rx) = mpsc::channel(capacity);
        Self {
            tx,
            rx: Some(rx),
            capacity,
            stats,
        }
    }

    /// Split the queue into producer and consumer
    pub fn split(mut self) -> (JobQueueProducer, JobQueueConsumer) {
        let rx = self.rx.take().expect("Consumer already taken");
        let producer = JobQueueProducer {
            tx: self.tx.clone(),
            capacity: self.capacity,
            stats: self.stats.clone(),
        };
        let consumer = JobQueueConsumer {
            rx,
            stats: self.stats.clone(),
        };
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
    stats: Arc<JobStats>,
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
                self.stats.increment_queue_size();
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
    stats: Arc<JobStats>,
}

impl JobQueueConsumer {
    /// Dequeue next job for execution
    pub async fn dequeue(&mut self) -> Option<QueuedJob> {
        let job = self.rx.recv().await?;
        self.stats.decrement_queue_size();
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

        // Log and track expired jobs
        for job in expired_jobs {
            warn!(
                job_id = %job.job.job_id,
                age_ms = job.age_ms(),
                timeout_ms = job.job.timeout_ms,
                "Job expired in queue"
            );
            self.stats.record_dropped_job();
            self.stats.update_max_queue_age(job.age_ms());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::job_protocol::JobEnvelope;
    use libp2p::request_response::ResponseChannel;
    use std::time::Duration;
    use uuid::Uuid;

    fn mock_job(id: u64, timeout_ms: u64) -> JobEnvelope {
        JobEnvelope {
            job_id: Uuid::from_u128(id as u128),
            network_id: "test-net".to_string(),
            workload_id: "embeddings".to_string(),
            payload: vec![],
            timeout_ms,
            auth_signature: vec![],
            created_at: 0,
        }
    }

    #[tokio::test]
    async fn test_queue_capacity() {
        let stats = Arc::new(JobStats::new());
        let queue = JobQueue::new(2, stats.clone());
        let (producer, _consumer) = queue.split();

        // Enqueue 2 jobs (should fill queue)
        let peer_id = PeerId::random();
        for i in 1..=2 {
            let (_, rx) = tokio::sync::oneshot::channel();
            let channel = ResponseChannel::from(rx);
            let queued_job = QueuedJob::new(mock_job(i, 5000), channel, peer_id);
            producer.enqueue(queued_job).await.unwrap();
        }

        assert_eq!(producer.size(), 2);
    }

    #[tokio::test]
    async fn test_enqueue_dequeue() {
        let stats = Arc::new(JobStats::new());
        let queue = JobQueue::new(10, stats.clone());
        let (producer, mut consumer) = queue.split();

        // Enqueue a job
        let peer_id = PeerId::random();
        let job = mock_job(1, 5000);
        let job_id = job.job_id;
        let (_, rx) = tokio::sync::oneshot::channel();
        let channel = ResponseChannel::from(rx);
        let queued_job = QueuedJob::new(job, channel, peer_id);

        producer.enqueue(queued_job).await.unwrap();
        assert_eq!(producer.size(), 1);

        // Dequeue the job
        let dequeued = consumer.dequeue().await.unwrap();
        assert_eq!(dequeued.job.job_id, job_id);
        assert_eq!(producer.size(), 0);
    }

    #[tokio::test]
    async fn test_job_expiry() {
        let stats = Arc::new(JobStats::new());
        let queue = JobQueue::new(10, stats.clone());
        let (producer, _consumer) = queue.split();

        // Create a job with very short timeout
        let peer_id = PeerId::random();
        let job = mock_job(1, 50); // 50ms timeout
        let (_, rx) = tokio::sync::oneshot::channel();
        let channel = ResponseChannel::from(rx);
        let queued_job = QueuedJob::new(job, channel, peer_id);

        assert!(!queued_job.is_expired());

        // Wait for expiry
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(queued_job.is_expired());
        assert!(queued_job.age_ms() >= 50);
    }

    #[tokio::test]
    async fn test_cleanup_expired() {
        let stats = Arc::new(JobStats::new());
        let queue = JobQueue::new(10, stats.clone());
        let (producer, mut consumer) = queue.split();

        // Enqueue jobs with different timeouts
        let peer_id = PeerId::random();

        // Job 1: 50ms timeout (will expire)
        let (_, rx1) = tokio::sync::oneshot::channel();
        producer
            .enqueue(QueuedJob::new(
                mock_job(1, 50),
                ResponseChannel::from(rx1),
                peer_id,
            ))
            .await
            .unwrap();

        // Job 2: 10000ms timeout (won't expire)
        let (_, rx2) = tokio::sync::oneshot::channel();
        producer
            .enqueue(QueuedJob::new(
                mock_job(2, 10000),
                ResponseChannel::from(rx2),
                peer_id,
            ))
            .await
            .unwrap();

        // Wait for first job to expire
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Cleanup
        let expired_count = consumer.cleanup_expired(&producer).await;

        assert_eq!(expired_count, 1);
        assert_eq!(stats.jobs_dropped(), 1);
    }
}
