#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics;
using System.Threading;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Vector {
    /// <summary>
    ///     A function that serves one sub-env of an <see cref="AsyncVectorEnv{TObs, TAct}"/> (DR-38, Gymnasium's
    ///     <c>worker</c>): it builds the sub-env, then answers the parent's commands through its pipe until a close, as
    ///     Gymnasium's worker process does. <see cref="AsyncVectorEnv.AsyncWorker"/> is the default one; a custom worker
    ///     usually wraps it, changing what it needs.
    /// </summary>
    /// <param name="index">The sub-env's index.</param>
    /// <param name="envFn">The function that builds the sub-env; the worker calls it and owns, closes and disposes what it returns.</param>
    /// <param name="pipe">
    ///     The worker's handle on its end of the pipe: it receives <c>(string Command, object? Data)</c> value tuples and
    ///     answers each with an <c>(object? Result, bool Success)</c> value tuple, in order. The commands are Gymnasium's:
    ///     <c>reset</c>, <c>reset-noop</c>, <c>step</c>, <c>close</c>, <c>_call</c>, <c>_setattr</c>, <c>_check_spaces</c>.
    /// </param>
    /// <param name="parentPipe">The worker's own handle on the parent's end, which Gymnasium's worker closes first; closing it leaves the parent's handle open.</param>
    /// <param name="sharedMemory">The shared observation buffer to write observations into (<see cref="VectorUtils.WriteToSharedMemory"/>), or <see langword="null"/> without shared memory.</param>
    /// <param name="errorQueue">Where a failing worker reports its exception before answering <c>(null, false)</c>, as Gymnasium's error queue.</param>
    /// <param name="autoresetMode">When an ended sub-env resets: the vector env's mode.</param>
    /// <param name="semaphore">The <c>max_concurrency</c> permits shared by every worker, to hold around <c>reset</c> and <c>step</c>, or <see langword="null"/>.</param>
    /// <remarks>
    ///     <para>
    ///     The signature is Gymnasium's eight arguments, so the arity check Gymnasium runs when the env is built is a
    ///     compile-time fact here. Each custom worker runs on a thread of its own (a background thread unless the vector
    ///     env's <c>daemon</c> is off), whatever the execution mode, as each of Gymnasium's runs in a process of its own.
    ///     </para>
    ///     <para>
    ///     Footgun: unlike a process, the worker shares the parent's memory. What crosses the pipe is passed by reference,
    ///     and the shared buffer is written in place (lock on <paramref name="sharedMemory"/> around a write, as the
    ///     default worker does, since writing touches the buffer's shared bookkeeping). A worker that returns without
    ///     answering a command leaves the parent's receive to report the end of the stream
    ///     (<see cref="System.IO.EndOfStreamException"/>, carrying what escaped the worker, if anything).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/async_vector_env.py#L823">gymnasium/vector/async_vector_env.py</see>.</para>
    /// </remarks>
    public delegate void AsyncVectorWorker(int index, Func<IEnv> envFn, PipeConnection pipe, PipeConnection parentPipe, object? sharedMemory,
        ConcurrentQueue<AsyncWorkerError> errorQueue, AutoresetMode autoresetMode, SemaphoreSlim? semaphore);

    /// <summary>
    ///     A worker's failure, as Gymnasium's error queue holds it: the worker's index, the exception (rethrown to the
    ///     vector env's caller as it was thrown) and its text with the stack trace (Python's <c>traceback.format_exc()</c>).
    /// </summary>
    /// <param name="Index">The failing worker's sub-env.</param>
    /// <param name="Exception">The exception.</param>
    /// <param name="Trace">The exception's text with its .NET stack trace, ending with a newline as Python's does; logged by the parent.</param>
    public readonly record struct AsyncWorkerError(int Index, Exception Exception, string Trace);

    /// <summary>A worker's answer to one command, as Gymnasium's worker sends it through its pipe: the result, and whether the command succeeded.</summary>
    /// <param name="Value">The result, or <see langword="null"/>.</param>
    /// <param name="Success">Whether the command succeeded.</param>
    internal readonly record struct AsyncWorkerMessage(object? Value, bool Success);

    /// <summary>
    ///     What a worker loop keeps between commands (the local variables of Gymnasium's worker): the sub-env, the pending
    ///     autoreset, the last observation sent, and whether the loop ended. Touched only by the code serving the worker.
    /// </summary>
    internal sealed class AsyncWorkerState {
        /// <summary>The state of the worker serving sub-env <paramref name="index"/>.</summary>
        /// <param name="index">The sub-env.</param>
        public AsyncWorkerState(int index) {
            Index = index;
        }

        /// <summary>The sub-env's index.</summary>
        public int Index { get; }

        /// <summary>The sub-env the worker built, or <see langword="null"/> before it is built and once it is closed.</summary>
        public IEnv? Env { get; set; }

        /// <summary>Whether the sub-env's episode ended and the next step resets it (the worker's <c>autoreset</c>).</summary>
        public bool Autoreset { get; set; }

        /// <summary>The last observation sent (the worker's <c>observation</c>), what a masked-out reset answers; <see langword="null"/> with shared memory.</summary>
        public object? Observation { get; set; }

        /// <summary>Whether the worker loop ended (a failure or a close).</summary>
        public bool Exited { get; set; }
    }

    /// <summary>
    ///     The body of Gymnasium's worker loop, shared by the two ways <see cref="AsyncVectorEnv{TObs, TAct}"/> serves its
    ///     sub-envs: its executors post each command to <see cref="Run"/> (DR-22's modes), and the public default worker,
    ///     <see cref="AsyncVectorEnv.AsyncWorker"/>, runs <see cref="Serve"/> over a pipe (DR-38). One body, so a custom
    ///     worker that wraps the default one behaves exactly as the vector env does without it.
    /// </summary>
    internal static class AsyncWorkerCommands {
        /// <summary>The command that resets the sub-env.</summary>
        public const string Reset = "reset";

        /// <summary>The command that answers a masked-out reset with the last observation.</summary>
        public const string ResetNoop = "reset-noop";

        /// <summary>The command that steps (or autoresets) the sub-env.</summary>
        public const string Step = "step";

        /// <summary>The command that closes the sub-env and ends the worker.</summary>
        public const string Close = "close";

        /// <summary>The command that reads or calls a sub-env member by its Python name.</summary>
        public const string Call = "_call";

        /// <summary>The command that sets a sub-env attribute through its wrappers.</summary>
        public const string SetAttr = "_setattr";

        /// <summary>The command that compares the sub-env's spaces with the batch's.</summary>
        public const string CheckSpaces = "_check_spaces";

        /// <summary>
        ///     Gymnasium's worker loop over a pipe (<c>_async_worker</c>): builds the sub-env, closes its handle on the parent's
        ///     end, then receives each command, runs it and sends the answer, until a close or a failure ends it; the sub-env
        ///     is closed and disposed on the way out.
        /// </summary>
        /// <param name="index">The sub-env's index.</param>
        /// <param name="envFn">Builds the sub-env.</param>
        /// <param name="pipe">The worker's end of the pipe.</param>
        /// <param name="parentPipe">The worker's handle on the parent's end, closed first.</param>
        /// <param name="sharedMemory">The shared observation buffer, or <see langword="null"/>.</param>
        /// <param name="errorQueue">Where failures are reported.</param>
        /// <param name="autoresetMode">The autoreset mode.</param>
        /// <param name="semaphore">The concurrency permits, or <see langword="null"/>.</param>
        /// <exception cref="Exception">
        ///     What <paramref name="envFn"/> throws (outside the loop's error handling, as in Gymnasium: the worker ends
        ///     before answering anything), or what a failed answer after a broken pipe throws.
        /// </exception>
        public static void Serve(int index, Func<IEnv> envFn, PipeConnection pipe, PipeConnection parentPipe, object? sharedMemory,
            ConcurrentQueue<AsyncWorkerError> errorQueue, AutoresetMode autoresetMode, SemaphoreSlim? semaphore) {
            // env = env_fn(): before the try, as Gymnasium's worker; a failing function ends the worker without an answer.
            var worker = new AsyncWorkerState(index) { Env = envFn() };
            // observation_space = env.observation_space: the shared buffer is written by the sub-env's own space.
            Space observationSpace = ObjectSpace.Unview(worker.Env.ObservationSpace)!;
            parentPipe.Close();
            try {
                while (true) {
                    (string command, object? data) = ((string, object?))pipe.Recv()!;
                    AsyncWorkerMessage message = Run(worker, command, data, observationSpace, sharedMemory, autoresetMode, semaphore, errorQueue);
                    pipe.Send((message.Value, message.Success));
                    // A close ends the loop after its answer, as Gymnasium's break; a failure ended the worker inside Run.
                    if (worker.Exited)
                        break;
                }
            } catch (Exception e) {
                // A broken pipe or a malformed command: reported as Gymnasium's except clause reports any error.
                errorQueue.Enqueue(new AsyncWorkerError(index, e, e + "\n"));
                pipe.Send(((object?)null, false));
            } finally {
                CloseEnv(worker);
            }
        }

        /// <summary>
        ///     One command of the worker loop: its result as the message the parent receives. An exception ends the worker: it
        ///     is queued with its trace, the sub-env is closed, and the message reports the failure.
        /// </summary>
        /// <param name="worker">The worker's state.</param>
        /// <param name="command">The command.</param>
        /// <param name="data">The command's data.</param>
        /// <param name="observationSpace">The space observations are written to <paramref name="sharedMemory"/> with.</param>
        /// <param name="sharedMemory">The shared observation buffer, or <see langword="null"/> to send observations instead.</param>
        /// <param name="autoresetMode">The autoreset mode.</param>
        /// <param name="semaphore">The concurrency permits around <c>reset</c> and <c>step</c>, or <see langword="null"/>.</param>
        /// <param name="errorQueue">Where a failure is reported.</param>
        /// <returns>The message: the result and <see langword="true"/>, or <see langword="null"/> and <see langword="false"/> after an error.</returns>
        public static AsyncWorkerMessage Run(AsyncWorkerState worker, string command, object? data, Space observationSpace, object? sharedMemory,
            AutoresetMode autoresetMode, SemaphoreSlim? semaphore, ConcurrentQueue<AsyncWorkerError> errorQueue) {
            if (worker.Exited) {
                // The worker ended earlier (an error or a close): its pipe has nothing more to say.
                errorQueue.Enqueue(new AsyncWorkerError(worker.Index, new RuntimeError($"Worker-{worker.Index} has already ended"), $"Worker-{worker.Index} has already ended\n"));
                return new AsyncWorkerMessage(null, false);
            }

            bool permitHeld = semaphore is not null && command is Reset or Step;
            try {
                if (permitHeld)
                    semaphore!.Wait();
                try {
                    IEnv env = worker.Env!;
                    switch (command) {
                        case Reset: {
                            (BigInteger? seed, IReadOnlyDictionary<string, object?>? options) = ((BigInteger?, IReadOnlyDictionary<string, object?>?))data!;
                            (object? observation, Info info) = env.Reset(seed, options);
                            if (sharedMemory is not null) {
                                WriteShared(observationSpace, worker.Index, observation, sharedMemory);
                                observation = null;
                            }

                            // A full reset ends any pending autoreset in both copy paths (DR-42). Gymnasium's worker clears it
                            // only in its shared-memory branch, so without shared memory its next step would reset this
                            // sub-env a second time with a reward of 0; Gym.NET deviates on purpose so sync and async agree.
                            worker.Autoreset = false;

                            worker.Observation = observation;
                            // Released before the result is handed over, as Gymnasium releases before the blocking send.
                            if (permitHeld) {
                                permitHeld = false;
                                semaphore!.Release();
                            }

                            return new AsyncWorkerMessage(new object?[] { observation, info }, true);
                        }
                        case ResetNoop:
                            return new AsyncWorkerMessage(new object?[] { worker.Observation, new Info() }, true);
                        case Step:
                            return new AsyncWorkerMessage(StepOne(worker, env, data, observationSpace, sharedMemory, autoresetMode, semaphore, ref permitHeld), true);
                        case Close:
                            worker.Exited = true;
                            CloseEnv(worker);
                            return new AsyncWorkerMessage(null, true);
                        case Call: {
                            (string name, object?[] args) = ((string, object?[]))data!;
                            if (name is Reset or Step or Close or SetAttr or CheckSpaces)
                                throw new ValueError($"Trying to call function `{name}` with `call`, use `{name}` directly instead.");
                            return new AsyncWorkerMessage(SyncVectorEnv<object?, object?>.CallOne(env, name, args), true);
                        }
                        case SetAttr: {
                            (string name, object? value) = ((string, object?))data!;
                            env.SetWrapperAttr(name, value);
                            return new AsyncWorkerMessage(null, true);
                        }
                        case CheckSpaces: {
                            (object observationMode, Space singleObservationSpace, Space singleActionSpace) = ((object, Space, Space))data!;
                            bool sameObservation = observationMode is "same"
                                ? Equals(ObjectSpace.Unview(singleObservationSpace), ObjectSpace.Unview(env.ObservationSpace))
                                : SpaceUtils.IsSpaceDtypeShapeEquiv(ObjectSpace.Unview(singleObservationSpace)!, ObjectSpace.Unview(env.ObservationSpace)!);
                            bool sameAction = Equals(ObjectSpace.Unview(singleActionSpace), ObjectSpace.Unview(env.ActionSpace));
                            return new AsyncWorkerMessage(new[] { sameObservation, sameAction }, true);
                        }
                        default:
                            throw new RuntimeError($"Received unknown command `{command}`. Must be one of [`reset`, `step`, `close`, `_call`, `_setattr`, `_check_spaces`].");
                    }
                } finally {
                    if (permitHeld)
                        semaphore!.Release();
                }
            } catch (Exception e) {
                // error_queue.put((index, type, value, trace)); pipe.send((None, False)); the worker ends, closing its env.
                errorQueue.Enqueue(new AsyncWorkerError(worker.Index, e, e + "\n"));
                worker.Exited = true;
                CloseEnv(worker);
                return new AsyncWorkerMessage(null, false);
            }
        }

        /// <summary>
        ///     The worker's <c>finally: env.close()</c>, then the env's disposal: the worker built it and owns it. Safe to call
        ///     twice (the second finds no env).
        /// </summary>
        /// <param name="worker">The worker's state.</param>
        public static void CloseEnv(AsyncWorkerState worker) {
            IEnv? env = worker.Env;
            worker.Env = null;
            if (env is null)
                return;
            try {
                env.Close();
                env.Dispose();
            } catch (Exception) {
                // A failing close in Gymnasium's worker ends a process nobody listens to; here it must not end the worker's
                // thread, and the sub-env is unusable either way.
            }
        }

        /// <summary>A <c>step</c> command by the autoreset mode (the worker's <c>"step"</c> branch), its observation written to the shared buffer when there is one.</summary>
        /// <param name="worker">The worker's state.</param>
        /// <param name="env">Its sub-env.</param>
        /// <param name="action">The sub-env's action.</param>
        /// <param name="observationSpace">The space the shared buffer is written with.</param>
        /// <param name="sharedMemory">The shared buffer, or <see langword="null"/>.</param>
        /// <param name="autoresetMode">The autoreset mode.</param>
        /// <param name="semaphore">The concurrency permits, or <see langword="null"/>.</param>
        /// <param name="permitHeld">Whether the worker holds a concurrency permit; released (and cleared) before the result is returned.</param>
        /// <returns><c>(observation, reward, terminated, truncated, info)</c>, the observation <see langword="null"/> when it went to the shared buffer.</returns>
        /// <exception cref="AssertionError">Thrown in <c>DISABLED</c> mode with a pending autoreset (Gymnasium's bare <c>assert</c>; it never holds one).</exception>
        private static object?[] StepOne(AsyncWorkerState worker, IEnv env, object? action, Space observationSpace, object? sharedMemory, AutoresetMode autoresetMode,
            SemaphoreSlim? semaphore, ref bool permitHeld) {
            object? observation;
            double reward;
            bool terminated, truncated;
            Info info;
            if (autoresetMode == AutoresetMode.NextStep) {
                if (worker.Autoreset) {
                    // The episode ended at the previous step: reset now, ignoring the action, with a reward of 0.
                    (observation, info) = env.Reset();
                    reward = 0;
                    terminated = false;
                    truncated = false;
                } else {
                    (observation, reward, terminated, truncated, info) = env.Step(action);
                }

                worker.Autoreset = terminated || truncated;
            } else if (autoresetMode == AutoresetMode.SameStep) {
                (observation, reward, terminated, truncated, info) = env.Step(action);
                if (terminated || truncated) {
                    (object? resetObservation, Info resetInfo) = env.Reset();
                    // {"final_info": info, "final_obs": observation, **reset_info}: in that key order.
                    var merged = new Info { ["final_info"] = info, ["final_obs"] = observation };
                    foreach (KeyValuePair<string, object?> entry in resetInfo)
                        merged[entry.Key] = entry.Value;
                    info = merged;
                    observation = resetObservation;
                }
            } else {
                if (worker.Autoreset)
                    throw new AssertionError("");
                (observation, reward, terminated, truncated, info) = env.Step(action);
            }

            if (sharedMemory is not null) {
                WriteShared(observationSpace, worker.Index, observation, sharedMemory);
                observation = null;
            }

            worker.Observation = observation;
            if (permitHeld) {
                permitHeld = false;
                semaphore!.Release();
            }

            return new object?[] { observation, reward, terminated, truncated, info };
        }

        /// <summary>
        ///     Writes a sub-env's observation into its slot of the shared buffer, one writer at a time: each slot is disjoint,
        ///     but making the slot views touches the buffer's shared bookkeeping. The lock is the buffer itself, the one object
        ///     every worker of a vector env (a custom one included) is given.
        /// </summary>
        /// <param name="observationSpace">The space the buffer was laid out by.</param>
        /// <param name="index">The sub-env.</param>
        /// <param name="observation">Its observation; read, not kept.</param>
        /// <param name="sharedMemory">The shared buffer.</param>
        private static void WriteShared(Space observationSpace, int index, object? observation, object sharedMemory) {
            lock (sharedMemory)
                VectorUtils.WriteToSharedMemory(observationSpace, index, observation, sharedMemory);
        }
    }
}
