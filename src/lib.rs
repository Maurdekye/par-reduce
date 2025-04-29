#![feature(mpmc_channel)]
use std::{
    sync::{atomic::{AtomicBool, AtomicUsize, Ordering}, mpmc::channel},
    thread::{self, available_parallelism},
};

fn par_reduce_inner<I, F>(it: I, f: F) -> Option<I::Item>
where
    I: Iterator,
    I::Item: Send + Sync,
    F: Fn(I::Item, I::Item) -> I::Item + Send + Sync,
{
    let in_flight = AtomicUsize::new(0);
    let work_finished = AtomicBool::new(false);
    thread::scope(|scope| {
        let (outbox, inbox) = channel();
        let num_cpus: usize = available_parallelism().map(usize::from).unwrap_or(1);
        let workers: Vec<_> = (0..num_cpus)
            .map(|_| {
                let inbox = inbox.clone();
                let outbox = outbox.clone();
                let f = &f;
                let in_flight = &in_flight;
                let work_finished = &work_finished;
                scope.spawn(move || {
                    let mut stock = None;
                    while let Some(elem) = inbox.recv().unwrap() {
                        stock = match stock {
                            Some(other) => {
                                outbox.send(Some(f(elem, other))).unwrap();
                                None
                            }
                            None => Some(elem),
                        };
                        let in_flight = in_flight.fetch_sub(1, Ordering::SeqCst);
                        let work_finished = work_finished.load(Ordering::SeqCst);
                        if work_finished && in_flight <= 1 {
                            (1..num_cpus).for_each(|_| outbox.send(None).unwrap());
                            break;
                        }
                    }
                    stock
                })
            })
            .collect();
        it.for_each(|elem| {
            in_flight.fetch_add(1, Ordering::SeqCst);
            outbox.send(Some(elem)).unwrap()
        });
        work_finished.store(true, Ordering::SeqCst);
        drop((inbox, outbox));
        workers
            .into_iter()
            .filter_map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    })
    .into_iter()
    .reduce(f)
}

pub trait ParReduce: Iterator {
    /// Reduce an iterator in parallel.
    ///
    /// ```
    /// use par_reduce::*;
    /// let sums = [(0, 1), (5, 6), (16, 2), (8, 9)]
    /// .into_iter()
    /// .par_reduce(|a, b| (a.0 + b.0, a.1 + b.1))
    /// .unwrap();
    /// assert_eq!(sums, (0 + 5 + 16 + 8, 1 + 6 + 2 + 9));
    /// ```
    fn par_reduce<F>(self, f: F) -> Option<Self::Item>
    where
        Self::Item: Send + Sync,
        F: Fn(Self::Item, Self::Item) -> Self::Item + Send + Sync;
}

impl<I> ParReduce for I
where
    I: Iterator,
    I::Item: Send + Sync,
{
    fn par_reduce<F>(self, f: F) -> Option<Self::Item>
    where
        Self::Item: Send + Sync,
        F: Fn(Self::Item, Self::Item) -> Self::Item + Send + Sync,
    {
        par_reduce_inner(self, f)
    }
}

#[test]
fn test() {
    let sums = [(0, 1), (5, 6), (16, 2), (8, 9)]
        .into_iter()
        .par_reduce(|a, b| (a.0 + b.0, a.1 + b.1))
        .unwrap();
    assert_eq!(sums, (0 + 5 + 16 + 8, 1 + 6 + 2 + 9));
}
