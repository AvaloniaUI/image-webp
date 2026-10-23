//! A part of a decode that runs on its own, one unit after another (the token partitions of a
//! lossy bitstream, an alpha plane), done ahead of the decoder on its caller's helper threads.
//! The decoder spawns no thread: it hands the work to a [`Runner`] while the runner has an idle
//! helper, and does the units itself otherwise. Units come out in order, and at most `ahead` of
//! them are done and not yet taken. The decoder waits only for the unit a running helper is on.

use std::collections::VecDeque;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

/// Hands work to an idle helper thread of the caller and says whether one took it. Work no
/// helper takes is dropped unrun, and the decoder does it itself.
pub type Runner = Arc<dyn Fn(Box<dyn FnOnce() + Send>) -> bool + Send + Sync>;

/// Units made one after another, each from the state the one before left.
pub(crate) trait Produce: Send + 'static {
    type Unit: Send + 'static;

    /// The next unit, in `spare` (a unit handed back) where there is one; `None` after the last.
    fn produce(&mut self, spare: Option<Self::Unit>) -> Option<Self::Unit>;
}

pub(crate) struct Ahead<P: Produce> {
    shared: Arc<Shared<P>>,
    runner: Runner,
}

struct Shared<P: Produce> {
    state: Mutex<State<P>>,
    ready: Condvar,
    ahead: usize,
}

struct State<P: Produce> {
    /// `None` while a helper produces with it.
    producer: Option<Box<P>>,
    units: VecDeque<std::thread::Result<Option<P::Unit>>>,
    /// Units taken and handed back, for the producer to reuse.
    spares: Vec<P::Unit>,
    /// Helpers handed the work and not yet back.
    running: usize,
    /// The decoder waits for a unit.
    waiting: bool,
    /// The last unit, or a panic, is made.
    finished: bool,
    stop: bool,
}

impl<P: Produce> Shared<P> {
    fn lock(&self) -> MutexGuard<'_, State<P>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Waits for a helper to put a unit or go. Where none runs, none ever will: the producer is lost.
    fn wait<'a>(&self, state: MutexGuard<'a, State<P>>) -> MutexGuard<'a, State<P>> {
        assert!(
            state.running > 0,
            "ahead: waiting with no helper running ({} units ready, producer {}, finished {})",
            state.units.len(),
            if state.producer.is_some() { "free" } else { "taken" },
            state.finished
        );
        let mut state = state;
        state.waiting = true;
        let mut state = self.ready.wait(state).unwrap_or_else(PoisonError::into_inner);
        state.waiting = false;
        state
    }

    fn help(self: Arc<Self>) {
        let helping = Helping(&self);
        let mut state = self.lock();
        let Some(mut producer) = state.producer.take() else {
            return;
        };
        loop {
            if state.stop || state.units.len() >= self.ahead {
                state.producer = Some(producer);
                break;
            }
            let spare = state.spares.pop();
            drop(state);
            let unit = catch_unwind(AssertUnwindSafe(|| producer.produce(spare)));
            state = self.lock();
            let more = matches!(unit, Ok(Some(_)));
            state.units.push_back(unit);
            if state.waiting {
                self.ready.notify_one();
            }
            if !more {
                state.finished = true;
                state.producer = Some(producer);
                break;
            }
        }
        drop(state);
        drop(helping);
    }
}

/// A helper counted in `running` until it returns or unwinds.
struct Helping<'a, P: Produce>(&'a Shared<P>);

impl<P: Produce> Drop for Helping<'_, P> {
    fn drop(&mut self) {
        self.0.lock().running -= 1;
        self.0.ready.notify_all();
    }
}

impl<P: Produce> Ahead<P> {
    /// Units of `producer`, at most `ahead` (at least 1) of them done ahead on `runner`'s helpers.
    pub(crate) fn new(producer: P, runner: Runner, ahead: usize) -> Self {
        let mut ahead = Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    producer: Some(Box::new(producer)),
                    units: VecDeque::new(),
                    spares: Vec::new(),
                    running: 0,
                    waiting: false,
                    finished: false,
                    stop: false,
                }),
                ready: Condvar::new(),
                ahead: ahead.max(1),
            }),
            runner,
        };
        let call = Self::calls_helper(&ahead.shared, &mut ahead.shared.lock());
        if call {
            ahead.call_helper();
        }
        ahead
    }

    /// The next unit, `None` after the last; `spent`, a unit taken before, is handed back for
    /// the producer to reuse.
    pub(crate) fn next(&mut self, spent: Option<P::Unit>) -> Option<P::Unit> {
        let shared = &*self.shared;
        let mut state = shared.lock();
        state.spares.extend(spent);
        let unit = loop {
            if let Some(unit) = state.units.pop_front() {
                break unit;
            }
            if state.finished {
                return None;
            }
            if let Some(mut producer) = state.producer.take() {
                let spare = state.spares.pop();
                drop(state);
                let unit = catch_unwind(AssertUnwindSafe(|| producer.produce(spare)));
                state = shared.lock();
                match unit {
                    Ok(Some(_)) => state.producer = Some(producer),
                    _ => state.finished = true,
                }
                break unit;
            }
            state = shared.wait(state);
        };
        let call = Self::calls_helper(shared, &mut state);
        drop(state);
        if call {
            self.call_helper();
        }
        unit.unwrap_or_else(|panic| resume_unwind(panic))
    }

    /// Whether to hand the producer to a helper: none has it and there is room ahead. Counts
    /// the helper it is handed to.
    fn calls_helper(shared: &Shared<P>, state: &mut State<P>) -> bool {
        let call = state.running == 0 && state.producer.is_some() && !state.finished && state.units.len() < shared.ahead;
        state.running += usize::from(call);
        call
    }

    /// Hands the producer to an idle helper, counted by [`calls_helper`](Self::calls_helper).
    fn call_helper(&mut self) {
        let helper = self.shared.clone();
        if !(self.runner)(Box::new(move || helper.help())) {
            self.shared.lock().running -= 1;
        }
    }
}

impl<P: Produce> Drop for Ahead<P> {
    /// Waits until no helper works on a unit and frees what the units hold, so that it is gone
    /// once the decoder is.
    fn drop(&mut self) {
        let mut state = self.shared.lock();
        state.stop = true;
        while state.running > 0 {
            state = self.shared.wait(state);
        }
        let held = (state.producer.take(), std::mem::take(&mut state.units), std::mem::take(&mut state.spares));
        drop(state);
        drop(held);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// A runner with `threads` helpers, each a thread of its own while it works.
    pub(crate) fn helpers(threads: usize) -> Runner {
        let idle = Arc::new(AtomicUsize::new(threads));
        Arc::new(move |work: Box<dyn FnOnce() + Send>| {
            let mut n = idle.load(Ordering::Acquire);
            loop {
                if n == 0 {
                    return false;
                }
                match idle.compare_exchange_weak(n, n - 1, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => break,
                    Err(now) => n = now,
                }
            }
            let idle = idle.clone();
            std::thread::spawn(move || {
                work();
                idle.fetch_add(1, Ordering::AcqRel);
            });
            true
        })
    }

    /// Counts to `count`, spinning unevenly per unit, and panics at `panic_at`.
    struct Counter {
        next: usize,
        count: usize,
        panic_at: usize,
        made: Arc<AtomicUsize>,
    }

    impl Produce for Counter {
        type Unit = Vec<usize>;

        fn produce(&mut self, spare: Option<Vec<usize>>) -> Option<Vec<usize>> {
            if self.next == self.count {
                return None;
            }
            let i = self.next;
            assert!(i != self.panic_at, "unit {i}");
            for _ in 0..(i * 7919 % 13) * 300 {
                std::hint::spin_loop();
            }
            self.next += 1;
            self.made.fetch_add(1, Ordering::SeqCst);
            let mut unit = spare.unwrap_or_default();
            unit.clear();
            unit.push(i);
            Some(unit)
        }
    }

    fn counter(count: usize) -> Counter {
        Counter { next: 0, count, panic_at: usize::MAX, made: Arc::default() }
    }

    #[test]
    fn hands_out_every_unit_in_order() {
        for round in 0..300 {
            let runner = helpers(round % 3);
            let mut a = Ahead::new(counter(40), runner, 1 + round % 5);
            let mut spent = None;
            for i in 0..40 {
                let unit = a.next(spent).unwrap();
                assert_eq!(unit, [i], "round {round}");
                spent = Some(unit);
            }
            assert_eq!(a.next(spent), None, "round {round}");
            assert_eq!(a.next(None), None, "round {round}");
        }
    }

    #[test]
    fn a_runner_that_works_in_place() {
        let runner: Runner = Arc::new(|work: Box<dyn FnOnce() + Send>| {
            work();
            true
        });
        let units: Vec<usize> = {
            let mut a = Ahead::new(counter(25), runner, 3);
            std::iter::from_fn(|| a.next(None)).map(|u| u[0]).collect()
        };
        assert_eq!(units, (0..25).collect::<Vec<_>>());
    }

    #[test]
    fn never_more_than_ahead_past_the_unit_taken() {
        let c = counter(60);
        let made = c.made.clone();
        let mut a = Ahead::new(c, helpers(1), 4);
        for i in 0..60 {
            std::thread::sleep(Duration::from_micros(300));
            assert!(made.load(Ordering::SeqCst) <= i + 4, "unit {i}: {} made", made.load(Ordering::SeqCst));
            assert_eq!(a.next(None).unwrap(), [i]);
        }
    }

    #[test]
    fn a_panic_reaches_the_decoder_at_its_unit() {
        for threads in [0, 1] {
            let mut c = counter(10);
            c.panic_at = 6;
            let mut a = Ahead::new(c, helpers(threads), 3);
            for i in 0..6 {
                assert_eq!(a.next(None).unwrap(), [i]);
            }
            let panic = catch_unwind(AssertUnwindSafe(|| a.next(None))).unwrap_err();
            assert_eq!(panic.downcast_ref::<String>().map(String::as_str), Some("unit 6"));
            assert_eq!(a.next(None), None);
        }
    }

    #[test]
    fn dropped_early_waits_for_its_helper() {
        let c = counter(1000);
        let made = c.made.clone();
        let mut a = Ahead::new(c, helpers(1), 50);
        assert_eq!(a.next(None).unwrap(), [0]);
        drop(a);
        let at_drop = made.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(made.load(Ordering::SeqCst), at_drop);
    }
}
