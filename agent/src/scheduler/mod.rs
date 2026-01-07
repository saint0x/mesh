//! Job scheduling module (legacy - being phased out)
//!
//! Note: For tensor-parallel inference, job queues are not used.
//! Workers participate in ALL inference jobs simultaneously via ring all-reduce.
//! This module is kept for backwards compatibility but should not be extended.

mod job_queue;

pub use job_queue::{QueuedJob, JobQueue, JobQueueProducer, JobQueueConsumer};
