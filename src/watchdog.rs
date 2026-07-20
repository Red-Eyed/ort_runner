//! Bounding how long a single inference may take.
//!
//! An inference that never returns takes the whole run with it: no statistics, no report, no
//! indication of which stage was at fault. This turns that into a bounded failure -- the run stops
//! and says what happened -- which is the position this repo already takes for containerised
//! commands, where every one is given a deadline so a hang cannot masquerade as slow progress.
//!
//! The timeout is not enforced by killing the process. ONNX Runtime can be asked to abandon an
//! in-flight `Run` from another thread, so the inference returns an ordinary error and the usual
//! reporting path handles it.
//!
//! What it does not cover is a hang before inference begins -- loading a model, or an execution
//! provider initialising its device. Those happen while the session is being built, where there is
//! no run to terminate.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

/// What the timer thread is told about the inference it is guarding.
enum Signal {
    Started,
    Finished,
}

/// Calls a termination action when one inference overruns its budget.
///
/// Generic over the action rather than taking ONNX Runtime's `RunOptions`, so the timing logic
/// carries no dependency on the runtime and can be tested without one.
#[derive(Debug)]
pub struct Watchdog {
    /// `None` when disabled, which is how a zero budget is represented -- there is then no thread
    /// and no channel, so an unbounded run pays nothing at all.
    sender: Option<Sender<Signal>>,
    timer: Option<JoinHandle<()>>,
    fired: Arc<AtomicBool>,
}

impl Watchdog {
    /// A watchdog that never fires.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            sender: None,
            timer: None,
            fired: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Starts a timer thread that runs `on_timeout` if any one inference exceeds `budget`.
    pub fn arm<F>(budget: Duration, on_timeout: F) -> Self
    where
        F: Fn() + Send + 'static,
    {
        let (sender, receiver) = mpsc::channel();
        let fired = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&fired);

        let timer = std::thread::spawn(move || {
            // Blocks until signalled rather than polling on a timer. A thread waking periodically
            // during a benchmark is contention, and contention is what the tail latency this tool
            // exists to measure is made of.
            while matches!(receiver.recv(), Ok(Signal::Started)) {
                match receiver.recv_timeout(budget) {
                    // Finished within budget; wait for the next one.
                    Ok(_) => {}
                    Err(RecvTimeoutError::Timeout) => {
                        flag.store(true, Ordering::SeqCst);
                        on_timeout();
                        return;
                    }
                    // The benchmark ended and dropped its sender.
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
        });

        Self {
            sender: Some(sender),
            timer: Some(timer),
            fired,
        }
    }

    /// Marks one inference as in flight. The returned guard ends it when dropped.
    ///
    /// A guard rather than a pair of calls, so the "finished" signal cannot be lost on an early
    /// return: were it missed, the watchdog would attribute the next iteration's wait to this one
    /// and fire against a healthy run.
    #[must_use]
    pub fn iteration(&self) -> InFlight<'_> {
        if let Some(sender) = &self.sender {
            let _ = sender.send(Signal::Started);
        }
        InFlight {
            sender: self.sender.as_ref(),
        }
    }

    /// Whether the budget was exceeded, which is what distinguishes a timeout from an inference
    /// that failed on its own.
    #[must_use]
    pub fn fired(&self) -> bool {
        self.fired.load(Ordering::SeqCst)
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        // Dropping the sender disconnects the channel, which is what tells the timer to finish.
        self.sender = None;
        if let Some(timer) = self.timer.take() {
            let _ = timer.join();
        }
    }
}

/// An inference the watchdog is currently timing.
#[derive(Debug)]
pub struct InFlight<'a> {
    sender: Option<&'a Sender<Signal>>,
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        if let Some(sender) = self.sender {
            let _ = sender.send(Signal::Finished);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Long enough that a loaded CI machine cannot miss it, short enough not to pad the suite.
    const GENEROUS: Duration = Duration::from_millis(400);

    #[test]
    fn an_inference_within_budget_does_not_fire() {
        let watchdog = Watchdog::arm(GENEROUS, || unreachable!("fired within budget"));

        {
            let _guard = watchdog.iteration();
        }

        assert!(!watchdog.fired());
    }

    #[test]
    fn an_overrunning_inference_fires_once_the_budget_passes() {
        let fired = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&fired);
        let watchdog = Watchdog::arm(Duration::from_millis(20), move || {
            flag.store(true, Ordering::SeqCst);
        });

        let _guard = watchdog.iteration();
        std::thread::sleep(GENEROUS);

        assert!(fired.load(Ordering::SeqCst), "the action should have run");
        assert!(watchdog.fired());
    }

    /// The budget applies per inference, not to the run as a whole: many iterations that are each
    /// comfortably inside it must not accumulate into a timeout.
    #[test]
    fn successive_iterations_each_get_the_full_budget() {
        let watchdog = Watchdog::arm(Duration::from_millis(200), || unreachable!("fired"));

        for _ in 0..8 {
            let _guard = watchdog.iteration();
            std::thread::sleep(Duration::from_millis(30));
        }

        assert!(!watchdog.fired());
    }

    #[test]
    fn a_disabled_watchdog_never_fires() {
        let watchdog = Watchdog::disabled();

        let _guard = watchdog.iteration();
        std::thread::sleep(Duration::from_millis(50));

        assert!(!watchdog.fired());
    }
}
