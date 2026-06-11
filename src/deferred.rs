//! An externally-settlable promise: a [`Sequence`] that suspends a running Lua
//! execution until host (Rust) code settles it.
//!
//! This is the piccolo-side bridge for "call a host async function, suspend the
//! script, resume when the backing Rust future completes," without a manual coroutine
//! dance. Return [`Deferred::sequence`] from a [`Callback`](crate::Callback) via
//! [`CallbackReturn::Sequence`](crate::CallbackReturn::Sequence) so that *calling* the
//! callback suspends its caller; later, host code calls [`Deferred::resolve`] or
//! [`Deferred::reject`], and the next [`Executor::step`](crate::Executor::step) resumes
//! the caller with the value (or raises the error).
//!
//! The settled value crosses from Rust into the GC as a [`StashedValue`]: the host
//! stashes it (e.g. with [`Context::stash`](crate::Context::stash)) so it stays rooted
//! in the registry until the awaiting sequence fetches it. That registry stashing is
//! piccolo's existing "hold a `'gc` value in `'static` Rust across steps" mechanism;
//! `Deferred` is just the suspend/resume protocol layered on top of it.

use std::cell::RefCell;
use std::pin::Pin;
use std::rc::Rc;

use gc_arena::{Collect, Mutation};

use crate::{
    BoxSequence, Context, Error, Execution, Sequence, SequencePoll, Stack, StashedError,
    StashedValue,
};

enum State {
    Pending,
    Resolved(StashedValue),
    Rejected(StashedError),
}

/// A handle to an externally-settlable promise.
///
/// Cheaply clonable; every clone shares one settlement state, so the host can hold one
/// clone while the [`Sequence`] returned by [`Deferred::sequence`] holds another.
#[derive(Clone)]
pub struct Deferred {
    state: Rc<RefCell<State>>,
}

impl Deferred {
    /// Create a fresh, unsettled `Deferred`.
    pub fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(State::Pending)),
        }
    }

    /// Whether this `Deferred` has been resolved or rejected.
    pub fn is_settled(&self) -> bool {
        !matches!(&*self.state.borrow(), State::Pending)
    }

    /// Resolve with `value`, which the awaiting [`Sequence`] returns to its caller on
    /// the next executor step. `value` must be registry-stashed (e.g. via
    /// [`Context::stash`](crate::Context::stash)) so it stays GC-rooted until consumed.
    /// A second settle (resolve or reject) is ignored, so this is safe to call twice.
    pub fn resolve(&self, value: StashedValue) {
        let mut state = self.state.borrow_mut();
        if matches!(&*state, State::Pending) {
            *state = State::Resolved(value);
        }
    }

    /// Reject with `error`, which the awaiting [`Sequence`] raises into its caller on
    /// the next executor step. A second settle is ignored.
    pub fn reject(&self, error: StashedError) {
        let mut state = self.state.borrow_mut();
        if matches!(&*state, State::Pending) {
            *state = State::Rejected(error);
        }
    }

    /// Build the [`Sequence`] that awaits settlement. Return it from a callback via
    /// [`CallbackReturn::Sequence`](crate::CallbackReturn::Sequence); the running
    /// [`Executor`](crate::Executor) re-polls it each step, yielding
    /// [`SequencePoll::Pending`] until this `Deferred` is settled.
    pub fn sequence<'gc>(&self, mc: &Mutation<'gc>) -> BoxSequence<'gc> {
        BoxSequence::new(
            mc,
            DeferredSequence {
                state: self.state.clone(),
            },
        )
    }
}

impl Default for Deferred {
    fn default() -> Self {
        Self::new()
    }
}

// The shared state holds only `'static` stashed handles, so the sequence carries no
// `'gc` data and needs no tracing.
#[derive(Collect)]
#[collect(no_drop)]
struct DeferredSequence {
    #[collect(require_static)]
    state: Rc<RefCell<State>>,
}

impl<'gc> Sequence<'gc> for DeferredSequence {
    fn poll(
        self: Pin<&mut Self>,
        ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        // Shared read of a non-pinned `Rc` field; nothing structural is pinned.
        let state = self.state.borrow();
        match &*state {
            State::Pending => Ok(SequencePoll::Pending),
            State::Resolved(value) => {
                let value = ctx.fetch(value);
                stack.replace(ctx, value);
                Ok(SequencePoll::Return)
            }
            State::Rejected(error) => Err(ctx.fetch(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Callback, CallbackReturn, Closure, Executor, Fuel, Lua, Value};

    // A bounded fuel for each manual step: large enough to run the tiny scripts, small
    // enough that a misbehaving pending loop would return promptly rather than spin.
    fn step_once(lua: &mut Lua, executor: &crate::StashedExecutor) -> bool {
        lua.enter(|ctx| {
            let ex = ctx.fetch(executor);
            ex.step(ctx, &mut Fuel::with(4096)).unwrap()
        })
    }

    #[test]
    fn deferred_suspends_then_resumes_with_value() {
        let mut lua = Lua::core();
        let deferred = Deferred::new();

        // A host function that suspends its caller on the deferred.
        {
            let deferred = deferred.clone();
            lua.enter(|ctx| {
                let cb = Callback::from_fn(&ctx, move |ctx, _exec, _stack| {
                    Ok(CallbackReturn::Sequence(deferred.sequence(&ctx)))
                });
                ctx.set_global("awaitHost", cb);
            });
        }

        // `return awaitHost()` suspends until the host settles the deferred.
        let executor = lua
            .try_enter(|ctx| {
                let closure = Closure::load(ctx, None, &b"return awaitHost()"[..])?;
                Ok(ctx.stash(Executor::start(ctx, closure.into(), ())))
            })
            .unwrap();

        // First step: the call suspends (deferred still pending), so not finished.
        assert!(!step_once(&mut lua, &executor), "call must suspend while pending");
        assert!(!deferred.is_settled());

        // Host settles it between steps, stashing the value so it stays rooted.
        lua.enter(|ctx| {
            deferred.resolve(ctx.stash(Value::Integer(42)));
        });
        assert!(deferred.is_settled());

        // Next step: the sequence sees Resolved, returns the value, executor finishes.
        assert!(step_once(&mut lua, &executor), "call resumes once settled");

        // The resolved value is the script's result.
        lua.enter(|ctx| {
            let ex = ctx.fetch(&executor);
            let result: i64 = ex.take_result::<i64>(ctx).unwrap().unwrap();
            assert_eq!(result, 42);
        });
    }

    #[test]
    fn deferred_reject_errors_the_execution() {
        let mut lua = Lua::core();
        let deferred = Deferred::new();
        {
            let deferred = deferred.clone();
            lua.enter(|ctx| {
                let cb = Callback::from_fn(&ctx, move |ctx, _exec, _stack| {
                    Ok(CallbackReturn::Sequence(deferred.sequence(&ctx)))
                });
                ctx.set_global("awaitHost", cb);
            });
        }

        let executor = lua
            .try_enter(|ctx| {
                let closure = Closure::load(ctx, None, &b"return awaitHost()"[..])?;
                Ok(ctx.stash(Executor::start(ctx, closure.into(), ())))
            })
            .unwrap();

        assert!(!step_once(&mut lua, &executor));

        // Reject with a Lua error value.
        lua.enter(|ctx| {
            deferred.reject(StashedError::Lua(ctx.stash(Value::Integer(7))));
        });

        assert!(step_once(&mut lua, &executor));

        // The rejection surfaces as an execution error, not a value.
        lua.enter(|ctx| {
            let ex = ctx.fetch(&executor);
            let result = ex.take_result::<Value>(ctx).unwrap();
            assert!(result.is_err(), "rejection surfaces as an execution error");
        });
    }
}
