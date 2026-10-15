//! Paces the windowed frame on macOS without ever making the main thread wait
//! for a drawable.
//!
//! The main thread is the one AppKit serves the rest of the system on. Some of
//! what it serves is on someone else's clock: a window manager with a filtering
//! event tap (Magnet, BetterTouchTool's snap areas) holds every mouse-down until
//! it has asked the clicked app about its window over Accessibility, and that
//! request is a message the main run loop answers when it gets to it. A loop that
//! spends each frame blocked in `nextDrawable` until the next vsync answers one
//! such message per frame, so a click and the start of a title-bar drag wait
//! tens of frames. See winit#1737 and the design record in the README.
//!
//! Two pieces fix it. [`FramePacer`] decides when a frame is due: one refresh
//! period after the last, run through winit's `request_redraw`, with the event
//! loop sleeping in `WaitUntil` between frames. [`Acquirer`] takes the waiting
//! out of the main thread: a worker thread calls `get_current_texture` as soon
//! as the previous frame is presented, and a due frame is put off, re-checked
//! every millisecond, until the drawable is in hand. A frame put off for longer
//! than [`MAX_DEFERRAL`] runs anyway, waiting for the drawable as the old loop did,
//! so a surface that never answers cannot stop the game.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How often a frame waiting for a drawable checks again.
pub const RETRY: Duration = Duration::from_millis(1);

/// How long a due frame waits for a drawable before it runs anyway.
pub const MAX_DEFERRAL: Duration = Duration::from_millis(100);

/// The period used until the monitor reports a refresh rate.
pub const DEFAULT_PERIOD: Duration = Duration::from_nanos(16_666_667);

/// What the event loop does after `about_to_wait`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// Request a redraw; the frame runs in the `RedrawRequested` that follows.
    Redraw,
    /// Sleep until the instant, or until an event arrives.
    WaitUntil(Instant),
    /// Sleep until an event arrives (a requested redraw is on its way).
    Wait,
}

/// When the next windowed frame is due, and whether one is put off waiting for
/// a drawable.
#[derive(Debug)]
pub struct FramePacer {
    period: Duration,
    deadline: Instant,
    deferred_since: Option<Instant>,
    redraw_requested: bool,
}

impl FramePacer {
    /// A pacer whose first frame is due at `now`.
    pub fn new(now: Instant) -> Self {
        Self {
            period: DEFAULT_PERIOD,
            deadline: now,
            deferred_since: None,
            redraw_requested: false,
        }
    }

    /// Sets the refresh period from a monitor's rate in millihertz; `None` or a
    /// rate under 1 Hz keeps the current period.
    pub fn set_refresh_millihertz(&mut self, rate: Option<u32>) {
        if let Some(rate) = rate.filter(|rate| *rate >= 1000) {
            self.period = Duration::from_secs_f64(1000.0 / f64::from(rate));
        }
    }

    #[cfg(test)]
    pub fn period(&self) -> Duration {
        self.period
    }

    /// Decides the next step after the event loop's `about_to_wait`.
    /// `drawable_ready` reports whether the next frame could acquire its
    /// drawable without waiting.
    pub fn step(&mut self, now: Instant, drawable_ready: bool) -> Step {
        if self.redraw_requested {
            return Step::Wait;
        }
        if now < self.deadline {
            return Step::WaitUntil(self.deadline);
        }
        if !drawable_ready {
            let since = *self.deferred_since.get_or_insert(now);
            if now.duration_since(since) < MAX_DEFERRAL {
                return Step::WaitUntil(now + RETRY);
            }
        }
        self.deferred_since = None;
        self.redraw_requested = true;
        Step::Redraw
    }

    /// Records that a frame is starting at `now` and moves the deadline one
    /// period on. A frame that started more than a period late is due a period
    /// from now instead, so a stall is not followed by a burst of catch-up frames.
    pub fn frame_started(&mut self, now: Instant) {
        self.redraw_requested = false;
        self.deferred_since = None;
        let next = self.deadline + self.period;
        self.deadline = if next <= now { now + self.period } else { next };
    }

    /// Makes the next frame due a [`RETRY`] from `now`, for a frame that parked
    /// before finishing and polls again rather than waiting out a period.
    pub fn retry_soon(&mut self, now: Instant) {
        self.deadline = now + RETRY;
    }

    /// Whether the event loop is waiting for a redraw this pacer asked for.
    pub fn redraw_requested(&self) -> bool {
        self.redraw_requested
    }
}

/// Something a worker thread can block on to produce the next drawable.
pub trait Acquire: Send + Sync + 'static {
    type Output: Send + 'static;
    fn acquire(&self) -> Self::Output;
}

/// Acquires drawables on a worker thread, one at a time, so the main thread
/// asks whether one is ready instead of waiting for it.
pub struct Acquirer<A: Acquire> {
    requests: Option<Sender<()>>,
    results: Receiver<A::Output>,
    ready: Option<A::Output>,
    in_flight: bool,
    worker: Option<JoinHandle<()>>,
}

impl<A: Acquire> Acquirer<A> {
    pub fn new(source: std::sync::Arc<A>) -> Self {
        let (requests, wanted) = mpsc::channel::<()>();
        let (deliver, results) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("tecs-acquire".to_owned())
            .spawn(move || {
                while wanted.recv().is_ok() {
                    if deliver.send(source.acquire()).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn the drawable acquisition thread");
        Self {
            requests: Some(requests),
            results,
            ready: None,
            in_flight: false,
            worker: Some(worker),
        }
    }

    /// Starts acquiring the next drawable unless one is already held or on its way.
    pub fn request(&mut self) {
        if self.ready.is_none() && !self.in_flight {
            if let Some(requests) = &self.requests {
                self.in_flight = requests.send(()).is_ok();
            }
        }
    }

    /// Reports whether a drawable is in hand, starting an acquisition if none is.
    pub fn is_ready(&mut self) -> bool {
        self.request();
        if self.in_flight {
            match self.results.try_recv() {
                Ok(output) => {
                    self.ready = Some(output);
                    self.in_flight = false;
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => self.in_flight = false,
            }
        }
        self.ready.is_some()
    }

    /// Takes the acquired drawable, waiting for it if it is not in hand yet.
    /// Returns `None` only when the worker has stopped.
    pub fn take(&mut self) -> Option<A::Output> {
        self.request();
        if self.ready.is_none() && self.in_flight {
            self.ready = self.results.recv().ok();
            self.in_flight = false;
        }
        self.ready.take()
    }

    /// Waits for an acquisition in flight and drops whatever was acquired, so
    /// the surface can be reconfigured with nothing outstanding.
    pub fn settle(&mut self) {
        if self.in_flight {
            let _ = self.results.recv();
            self.in_flight = false;
        }
        self.ready = None;
    }
}

impl<A: Acquire> Drop for Acquirer<A> {
    fn drop(&mut self) {
        self.settle();
        self.requests = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn a_frame_is_due_once_per_period() {
        let t0 = Instant::now();
        let mut pacer = FramePacer::new(t0);
        pacer.set_refresh_millihertz(Some(50_000));
        assert_eq!(pacer.period(), ms(20));
        assert_eq!(pacer.step(t0, true), Step::Redraw);
        assert_eq!(pacer.step(t0, true), Step::Wait);
        pacer.frame_started(t0);
        assert_eq!(pacer.step(t0 + ms(5), true), Step::WaitUntil(t0 + ms(20)));
        assert_eq!(pacer.step(t0 + ms(20), true), Step::Redraw);
        pacer.frame_started(t0 + ms(21));
        assert_eq!(pacer.step(t0 + ms(22), true), Step::WaitUntil(t0 + ms(40)));
    }

    #[test]
    fn a_frame_without_a_drawable_is_put_off_and_bounded() {
        let t0 = Instant::now();
        let mut pacer = FramePacer::new(t0);
        assert_eq!(pacer.step(t0, false), Step::WaitUntil(t0 + RETRY));
        assert_eq!(
            pacer.step(t0 + ms(3), false),
            Step::WaitUntil(t0 + ms(3) + RETRY)
        );
        assert_eq!(pacer.step(t0 + ms(4), true), Step::Redraw);
        pacer.frame_started(t0 + ms(4));
        let due = t0 + DEFAULT_PERIOD;
        assert_eq!(pacer.step(due, false), Step::WaitUntil(due + RETRY));
        assert_eq!(pacer.step(due + MAX_DEFERRAL, false), Step::Redraw);
    }

    #[test]
    fn a_stall_does_not_bring_catch_up_frames() {
        let t0 = Instant::now();
        let mut pacer = FramePacer::new(t0);
        pacer.set_refresh_millihertz(Some(100_000));
        pacer.step(t0, true);
        pacer.frame_started(t0 + ms(250));
        assert_eq!(
            pacer.step(t0 + ms(251), true),
            Step::WaitUntil(t0 + ms(260))
        );
    }

    #[test]
    fn a_parked_frame_polls_again_soon() {
        let t0 = Instant::now();
        let mut pacer = FramePacer::new(t0);
        pacer.step(t0, true);
        pacer.frame_started(t0);
        pacer.retry_soon(t0 + ms(2));
        assert_eq!(
            pacer.step(t0 + ms(2), true),
            Step::WaitUntil(t0 + ms(2) + RETRY)
        );
        assert_eq!(pacer.step(t0 + ms(3), true), Step::Redraw);
    }

    #[test]
    fn an_unknown_refresh_rate_keeps_the_period() {
        let mut pacer = FramePacer::new(Instant::now());
        pacer.set_refresh_millihertz(None);
        pacer.set_refresh_millihertz(Some(0));
        assert_eq!(pacer.period(), DEFAULT_PERIOD);
        pacer.set_refresh_millihertz(Some(120_000));
        assert_eq!(pacer.period(), Duration::from_secs_f64(1.0 / 120.0));
    }

    /// Hands out numbered drawables, each only once the test releases it.
    struct Gate {
        released: Mutex<Receiver<()>>,
        count: Mutex<u32>,
    }

    impl Acquire for Gate {
        type Output = u32;
        fn acquire(&self) -> u32 {
            self.released.lock().unwrap().recv().unwrap();
            let mut count = self.count.lock().unwrap();
            *count += 1;
            *count
        }
    }

    fn gated() -> (Acquirer<Gate>, Sender<()>) {
        let (release, released) = mpsc::channel();
        let gate = Arc::new(Gate {
            released: Mutex::new(released),
            count: Mutex::new(0),
        });
        (Acquirer::new(gate), release)
    }

    fn ready_within(acquirer: &mut Acquirer<Gate>, limit: Duration) -> bool {
        let start = Instant::now();
        while start.elapsed() < limit {
            if acquirer.is_ready() {
                return true;
            }
            std::thread::sleep(ms(1));
        }
        false
    }

    #[test]
    fn the_main_thread_asks_rather_than_waits() {
        let (mut acquirer, release) = gated();
        assert!(!acquirer.is_ready());
        std::thread::sleep(ms(10));
        assert!(!acquirer.is_ready(), "nothing was released");
        release.send(()).unwrap();
        assert!(ready_within(&mut acquirer, ms(2000)));
        assert_eq!(acquirer.take(), Some(1));
        assert!(!acquirer.is_ready());
        release.send(()).unwrap();
        assert_eq!(acquirer.take(), Some(2), "take waits when it has to");
    }

    #[test]
    fn settle_drops_the_drawable_in_hand_and_in_flight() {
        let (mut acquirer, release) = gated();
        release.send(()).unwrap();
        assert!(ready_within(&mut acquirer, ms(2000)));
        acquirer.settle();
        assert!(!acquirer.is_ready());
        release.send(()).unwrap();
        acquirer.settle();
        release.send(()).unwrap();
        assert_eq!(
            acquirer.take(),
            Some(3),
            "the settled ones were dropped, not kept"
        );
    }

    #[test]
    fn dropping_stops_the_worker() {
        let (mut acquirer, release) = gated();
        acquirer.request();
        release.send(()).unwrap();
        drop(acquirer);
        assert!(release.send(()).is_err(), "the worker is gone");
    }
}
