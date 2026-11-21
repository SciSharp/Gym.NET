#nullable enable
using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Threading;
using System.Threading.Tasks;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Vector {
    /// <summary>
    ///     A vector env that runs its sub-envs in parallel, each call on a worker thread (Gymnasium's <c>AsyncVectorEnv</c>,
    ///     whose workers are processes): the same results as <see cref="SyncVectorEnv{TObs, TAct}"/> gathered in sub-env order,
    ///     for envs whose steps are long enough to be worth a thread. Typed by the batched types; <see cref="AsyncVectorEnv"/>
    ///     is the untyped one.
    /// </summary>
    /// <typeparam name="TObs">The batched observation type: the element type of the single observation space batched.</typeparam>
    /// <typeparam name="TAct">The batched action type.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Gymnasium's protocol stays: <see cref="ResetAsync(BigInteger?, IReadOnlyDictionary{string, object})"/>,
    ///     <see cref="StepAsync"/> and <see cref="CallAsync"/> send a call to every sub-env and return at once; the matching
    ///     <see cref="ResetWait"/>, <see cref="StepWait"/> and <see cref="CallWait"/> collect the results, optionally within a
    ///     timeout (<see cref="TimeoutException"/>, Python's <c>multiprocessing.TimeoutError</c>); one call is pending at a
    ///     time (<see cref="AsyncState"/>), and <see cref="Reset(BigInteger?, IReadOnlyDictionary{string, object})"/> and
    ///     <see cref="Step"/> are the two halves back to back. Each sub-env is built by its worker from <see cref="EnvFns"/>,
    ///     owned by it and closed by it, after one extra build of the first function in the constructor to read the metadata
    ///     and spaces, as Gymnasium's dummy env.
    ///     </para>
    ///     <para>
    ///     Where the calls run is <see cref="ExecutionMode"/> (DR-22): the thread pool, dedicated threads shared by the
    ///     sub-envs, or one thread per sub-env; a sub-env rendering to a window gets its own thread in pool mode. Results are
    ///     gathered in sub-env order whatever finishes first, so every mode gives the same bits. <see cref="MaxConcurrency"/>
    ///     limits how many sub-envs run <c>reset</c> or <c>step</c> at once. With <see cref="SharedMemory"/> (the default)
    ///     the workers write their observations straight into one buffer (<see cref="VectorUtils.CreateSharedMemory"/>), as
    ///     Gymnasium's processes write into shared memory; threads share memory anyway, so the option changes the copy path,
    ///     not the results, except where Gymnasium's own two paths differ (see the deviations).
    ///     </para>
    ///     <para>
    ///     A sub-env that raises ends its worker, as Gymnasium's process does: the wait that collects the call logs Gymnasium's
    ///     three error lines (the worker's index, the exception with its .NET stack trace where Python prints its traceback,
    ///     and the last line) and rethrows the last failing sub-env's exception as it was thrown. That sub-env is closed and
    ///     detached, so any later call that needs it fails as Python's does on the dropped pipe (<see cref="AttributeError"/>
    ///     <c>'NoneType' object has no attribute 'send'</c>); closing still works. Errors are collected in the order the
    ///     sub-envs failed, as Python's error queue receives them.
    ///     </para>
    ///     <para>
    ///     Ownership (D19): with <see cref="Copy"/> every observation batch returned is a new deep copy the caller owns;
    ///     without it, the batch is this env's own buffer (the shared one when <see cref="SharedMemory"/> is on), overwritten
    ///     by the next call. Rewards and flags are always new arrays. Not thread-safe: one thread drives the vector env.
    ///     </para>
    ///     <para>
    ///     Footgun: the sub-envs run at once on different threads of one process, where Gymnasium's processes each receive a
    ///     copy of whatever the functions capture. A function must therefore build everything mutable its sub-env uses (a
    ///     space, a generator, a buffer) rather than share one object among sub-envs: two threads sampling one space race on
    ///     its generator. The actions given to <see cref="StepAsync"/> are read by the workers after it returns; keep them
    ///     until the matching wait.
    ///     </para>
    ///     <para>
    ///     Deviations, each where Python's processes have no thread counterpart: <c>context</c> is accepted and has nothing
    ///     to choose (threads have no start method); <c>daemon</c> decides whether the dedicated threads are background
    ///     threads (pool threads always are); a custom <c>worker</c> (<see cref="AsyncVectorWorker"/>, DR-38) runs on a thread
    ///     of its own over a <see cref="PipeConnection"/> pair, whatever the execution mode; a sub-env function that throws
    ///     makes the constructor throw that exception (Python's parent sees its worker's pipe close), except under a custom
    ///     worker, where the parent's receive reports the end of the stream as Python's does
    ///     (<see cref="EndOfStreamException"/>, carrying the exception); <c>close(terminate=True)</c> cannot kill a thread, so it stops waiting and lets
    ///     each worker close its sub-env as soon as the call it runs returns, and <see cref="IDisposable.Dispose"/> waits for
    ///     that; a full reset clears a <c>NEXT_STEP</c> worker's pending autoreset in both copy paths (DR-42), where
    ///     Gymnasium's worker clears it only with shared memory, so its next step without shared memory resets that sub-env
    ///     a second time, with a reward of 0 (Gym.NET's sync and async envs agree in every sequence instead). Kept from
    ///     Gymnasium on purpose: a <c>Text</c> or <c>OneOf</c> shared buffer is read once, at construction; a
    ///     <c>SAME_STEP</c> info puts <c>final_info</c> before
    ///     <c>final_obs</c>; <c>reset_mask</c> leaves the flags of the unmasked sub-envs alone. As in
    ///     <see cref="SyncVectorEnv{TObs, TAct}"/>, a <c>reset_mask</c> stays in the caller's options dict, and
    ///     <see cref="VectorEnv{TObs, TAct}.NpRandom"/> and <see cref="VectorEnv{TObs, TAct}.NpRandomSeed"/> keep the base's
    ///     meaning (<c>GetAttr("np_random")</c> gives the sub-envs').
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/async_vector_env.py#L56">gymnasium/vector/async_vector_env.py</see>.</para>
    /// </remarks>
    public class AsyncVectorEnv<TObs, TAct> : VectorEnv<TObs, TAct> {
        /// <summary>The worker command that resets the sub-env.</summary>
        private const string ResetCommand = AsyncWorkerCommands.Reset;

        /// <summary>The worker command that answers a masked-out reset with the last observation.</summary>
        private const string ResetNoopCommand = AsyncWorkerCommands.ResetNoop;

        /// <summary>The worker command that steps (or autoresets) the sub-env.</summary>
        private const string StepCommand = AsyncWorkerCommands.Step;

        /// <summary>The worker command that closes the sub-env and ends the worker.</summary>
        private const string CloseCommand = AsyncWorkerCommands.Close;

        /// <summary>The worker command that reads or calls a sub-env member by its Python name.</summary>
        private const string CallCommand = AsyncWorkerCommands.Call;

        /// <summary>The worker command that sets a sub-env attribute through its wrappers.</summary>
        private const string SetAttrCommand = AsyncWorkerCommands.SetAttr;

        /// <summary>The worker command that compares the sub-env's spaces with the batch's.</summary>
        private const string CheckSpacesCommand = AsyncWorkerCommands.CheckSpaces;

        /// <summary>The single action space batched, untyped: what <c>iterate</c> splits the actions with.</summary>
        private readonly Space _batchedActionSpace;

        /// <summary>The single observation space batched, untyped (this env's whether or not <typeparamref name="TObs"/> views it).</summary>
        private readonly Space _batchedObservationSpace;

        /// <summary>The workers, one per sub-env, in order (Python's processes and their pipes).</summary>
        private readonly Worker[] _workers = Array.Empty<Worker>();

        /// <summary>The dedicated threads the workers run on, in <see cref="AsyncExecutionMode.DedicatedThreads"/> and <see cref="AsyncExecutionMode.ThreadPerEnv"/> modes, and for thread-bound sub-envs.</summary>
        private readonly List<WorkerThread> _threads = new();

        /// <summary>The permits of <see cref="MaxConcurrency"/>, or <see langword="null"/> for no limit; shared by every worker.</summary>
        private readonly SemaphoreSlim? _semaphore;

        /// <summary>The failures reported by the workers, in the order they failed (Python's <c>error_queue</c>); a custom worker is handed it.</summary>
        private readonly ConcurrentQueue<AsyncWorkerError> _errorQueue = new();

        /// <summary>The shared observation buffer (Python's <c>_obs_buffer</c>), or <see langword="null"/> without <see cref="SharedMemory"/>.</summary>
        private readonly object? _sharedMemory;

        /// <summary>The spaces this env made (copies of the first sub-env's, and the other sub-envs' in <c>"different"</c> mode); disposed with it.</summary>
        private readonly List<Space> _ownedSpaces = new();

        /// <summary>The close work posted without waiting by a terminating close, which disposal waits for.</summary>
        private readonly List<Task> _closing = new();

        /// <summary>
        ///     The observation batch (Python's <c>observations</c>): the shared buffer read as batches (live views), or the
        ///     buffer the sub-envs' observations are concatenated into without <see cref="SharedMemory"/>.
        /// </summary>
        private object? _observations;

        /// <summary>The call pending on the workers (Python's <c>_state</c>).</summary>
        private AsyncState _state = AsyncState.Default;

        /// <summary>
        ///     Builds the vector env (<c>AsyncVectorEnv(env_fns, shared_memory, copy, observation_mode=..., autoreset_mode=...,
        ///     max_concurrency=...)</c>): the first function once more for the metadata and spaces, then one worker per function,
        ///     each building its sub-env on its own thread, and the check that every sub-env's spaces fit the batch.
        /// </summary>
        /// <param name="envFns">One function per sub-env; each is called by its worker, which owns and closes what it returns. The first is also called once by the constructor (closed at once).</param>
        /// <param name="sharedMemory">Whether the workers write their observations into one shared buffer (the default); a space with no shared layout is refused when on.</param>
        /// <param name="copy">Whether every returned observation batch is a deep copy (the default) or the env's own buffer.</param>
        /// <param name="observationMode">
        ///     <c>"same"</c> (<see langword="null"/>, the default): every sub-env has the first's observation space, batched;
        ///     <c>"different"</c>: the spaces may differ but share shape and dtype (each function is called once more to read
        ///     its space); or a pair (<see langword="object"/>?[] or a value tuple) of the batched and the single space to use.
        /// </param>
        /// <param name="autoresetMode">An <see cref="Gym.Vector.AutoresetMode"/> or its value as a string; <see langword="null"/> is <see cref="Gym.Vector.AutoresetMode.NextStep"/>.</param>
        /// <param name="maxConcurrency">How many sub-envs may run <c>reset</c> or <c>step</c> at once: a positive integer (any .NET integer or a 0-d integer array), or <see langword="null"/> for no limit.</param>
        /// <param name="executionMode">Where the calls run (DR-22); <see cref="AsyncExecutionMode.ThreadPool"/> by default.</param>
        /// <param name="threadCount">The number of threads of <see cref="AsyncExecutionMode.DedicatedThreads"/> (default <see cref="Environment.ProcessorCount"/>); refused in the other modes.</param>
        /// <param name="context">Gymnasium's <c>multiprocessing</c> start method (<c>"spawn"</c>, <c>"fork"</c>, ...): accepted and kept, with nothing to choose, since the workers are threads.</param>
        /// <param name="daemon">
        ///     Whether the dedicated threads (of <see cref="AsyncExecutionMode.DedicatedThreads"/>, <see cref="AsyncExecutionMode.ThreadPerEnv"/>,
        ///     a thread-bound sub-env, a custom worker) are background threads, which never keep the process alive (the
        ///     default, as Gymnasium's daemon processes); <see langword="false"/> makes them foreground threads the process
        ///     waits for. Pool threads are background threads either way.
        /// </param>
        /// <param name="worker">
        ///     A custom worker (DR-38) to serve each sub-env instead of the built-in one, on a thread of its own over a pipe,
        ///     whatever <paramref name="executionMode"/> says; usually a wrapper around <see cref="AsyncVectorEnv.AsyncWorker"/>.
        ///     <see langword="null"/> (the default) keeps the built-in workers of <paramref name="executionMode"/>.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="envFns"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">
        ///     Thrown for an unknown autoreset or observation mode, a <paramref name="maxConcurrency"/> that is not a positive
        ///     integer, a misplaced or non-positive <paramref name="threadCount"/>, or a shared buffer for a space that has none,
        ///     with Gymnasium's messages.
        /// </exception>
        /// <exception cref="IndexError">Thrown when there are no sub-envs (<c>list index out of range</c>).</exception>
        /// <exception cref="TypeError">Thrown when a pair observation mode holds something else than spaces, a batched space's elements are not <typeparamref name="TObs"/>/<typeparamref name="TAct"/>, or a shared buffer is asked for a <c>Graph</c> or <c>Sequence</c> space.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch, with Gymnasium's messages.</exception>
        /// <exception cref="EndOfStreamException">Thrown under a custom worker that ended without answering (a sub-env function that throws, carried as the inner exception), as Python's parent sees <c>EOFError</c>.</exception>
        /// <exception cref="OSError">Thrown under a custom worker that ended before the first command reached it (<c>Broken pipe</c>, Python's <c>BrokenPipeError</c>), carrying its exception likewise.</exception>
        /// <exception cref="Exception">Whatever a sub-env function throws, rethrown after the other workers were shut down.</exception>
        public AsyncVectorEnv(IEnumerable<Func<IEnv>> envFns, bool sharedMemory = true, bool copy = true, object? observationMode = null, object? autoresetMode = null,
            object? maxConcurrency = null, AsyncExecutionMode executionMode = AsyncExecutionMode.ThreadPool, int? threadCount = null, string? context = null,
            bool daemon = true, AsyncVectorWorker? worker = null) {
            if (envFns is null)
                throw new ArgumentNullException(nameof(envFns));
            EnvFns = envFns.ToArray();
            SharedMemory = sharedMemory;
            Copy = copy;
            ObservationMode = observationMode ?? "same";
            AutoresetMode = AutoresetModes.From(autoresetMode);
            MaxConcurrency = CheckMaxConcurrency(maxConcurrency);
            ExecutionMode = executionMode;
            ThreadCount = CheckThreadCount(executionMode, threadCount);
            Context = context;
            Daemon = daemon;
            WorkerFunction = worker;
            NumEnvs = EnvFns.Count;
            if (EnvFns.Count == 0)
                throw new IndexError("list index out of range");

            try {
                // Gymnasium builds a dummy env from the first function to read the metadata and the spaces, then drops it.
                // Its spaces outlive it there; here they are copied, since disposing the env disposes its own.
                using (IEnv dummyEnv = EnvFns[0]()) {
                    // Copy the metadata such that we don't modify the sub-environment's (class-level) metadata.
                    Metadata = new Metadata(dummyEnv.Metadata) { ["autoreset_mode"] = AutoresetMode };
                    RenderMode = dummyEnv.RenderMode;
                    SingleActionSpace = Own(ObjectSpace.Unview(dummyEnv.ActionSpace)!.DeepCopy());
                    _batchedActionSpace = VectorUtils.BatchSpace(SingleActionSpace, NumEnvs);
                    ActionSpace = Typed<TAct>(_batchedActionSpace, "action_space");

                    (Space batched, Space single) = ObservationSpaces(ObservationMode, dummyEnv);
                    _batchedObservationSpace = batched;
                    SingleObservationSpace = single;
                    ObservationSpace = Typed<TObs>(batched, "observation_space");
                }

                if (SharedMemory) {
                    try {
                        _sharedMemory = VectorUtils.CreateSharedMemory(SingleObservationSpace, NumEnvs);
                        _observations = VectorUtils.ReadFromSharedMemory(SingleObservationSpace, _sharedMemory, NumEnvs);
                    } catch (CustomSpaceError e) {
                        throw new ValueError("Using `AsyncVector(..., shared_memory=True)` caused an error, you can disable this feature with `shared_memory=False` however this is slower.", e);
                    }
                } else {
                    _observations = VectorUtils.CreateEmptyArray(SingleObservationSpace, NumEnvs, (shape, dtype) => np.zeros(shape, dtype));
                }

                // A shared semaphore to limit the number of environments executing at any one time.
                _semaphore = MaxConcurrency is { } permits ? new SemaphoreSlim(permits, permits) : null;
                // A custom worker serves its sub-env over a pipe on a thread of its own; the built-in one through an executor.
                _workers = Enumerable.Range(0, NumEnvs).Select(index => worker is null ? new Worker(index, ExecutorFor(index)) : new Worker(index)).ToArray();
                StartWorkers();
                _state = AsyncState.Default;
                CheckSpaces();
            } catch {
                // Nothing else holds the workers, the buffers or the spaces yet: release them before the error propagates, as
                // Python's half-built env is closed with terminate=True when collected.
                Abandon();
                throw;
            }
        }

        /// <summary>The functions the sub-envs are built from (<c>env_fns</c>), in order.</summary>
        public IReadOnlyList<Func<IEnv>> EnvFns { get; }

        /// <summary>Whether the workers write their observations into one shared buffer (<c>shared_memory</c>).</summary>
        public bool SharedMemory { get; }

        /// <summary>Whether returned observation batches are deep copies (<c>copy</c>).</summary>
        public bool Copy { get; }

        /// <summary>The observation mode given (<c>observation_mode</c>): <c>"same"</c>, <c>"different"</c>, or the pair of batched and single spaces.</summary>
        public object ObservationMode { get; }

        /// <summary>When ended sub-envs reset (<c>autoreset_mode</c>), also in <c>metadata["autoreset_mode"]</c>; applied by each worker.</summary>
        public AutoresetMode AutoresetMode { get; }

        /// <summary>How many sub-envs may run <c>reset</c> or <c>step</c> at once (<c>max_concurrency</c>), or <see langword="null"/> for no limit; other calls are not limited.</summary>
        public int? MaxConcurrency { get; }

        /// <summary>Where the workers' calls run (DR-22).</summary>
        public AsyncExecutionMode ExecutionMode { get; }

        /// <summary>The number of dedicated threads in <see cref="AsyncExecutionMode.DedicatedThreads"/> mode, or <see langword="null"/> in the other modes.</summary>
        public int? ThreadCount { get; }

        /// <summary>The start method given (<c>context</c>), kept with nothing to choose: the workers are threads.</summary>
        public string? Context { get; }

        /// <summary>Whether the dedicated threads are background threads (<c>daemon</c>); pool threads always are.</summary>
        public bool Daemon { get; }

        /// <summary>The custom worker serving each sub-env (<c>worker</c>, DR-38), or <see langword="null"/> for the built-in workers.</summary>
        [PyName("worker")]
        public AsyncVectorWorker? WorkerFunction { get; }

        /// <summary>The pending call (Python's <c>_state</c>): <see cref="AsyncState.Default"/> when none.</summary>
        public AsyncState State => _state;

        /// <summary>Resets every sub-env in parallel and waits for them (<c>reset(seed=seed, options=options)</c>).</summary>
        /// <param name="seed">The first sub-env's seed (sub-env <c>i</c> gets <c>seed + i</c>), or <see langword="null"/> to keep every generator.</param>
        /// <param name="options">The options passed to every reset sub-env; a <c>reset_mask</c> entry resets only the masked ones.</param>
        /// <returns>The batched observations (a copy unless <see cref="Copy"/> is off) and the merged infos.</returns>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="AlreadyPendingCallError">Thrown while another call is pending.</exception>
        /// <exception cref="TypeError">Thrown when <c>reset_mask</c> is not an array or not bool.</exception>
        /// <exception cref="ValueError">Thrown when <c>reset_mask</c> has another shape or no <see langword="true"/>.</exception>
        /// <exception cref="Exception">The last failing sub-env's exception, after the error lines are logged.</exception>
        public override VectorResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ResetAsync(seed, options);
            return ResetWait();
        }

        /// <summary>Resets every sub-env with a seed each and waits for them (<c>reset(seed=[...])</c>).</summary>
        /// <param name="seeds">One seed per sub-env, <see langword="null"/> keeping that sub-env's generator.</param>
        /// <param name="options">The options passed to every reset sub-env, with an optional <c>reset_mask</c>.</param>
        /// <returns>The batched observations and the merged infos.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="seeds"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown when there is not one seed per sub-env, with Gymnasium's message, or for a bad <c>reset_mask</c>.</exception>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="AlreadyPendingCallError">Thrown while another call is pending.</exception>
        /// <exception cref="TypeError">Thrown for a <c>reset_mask</c> that is not a bool array.</exception>
        /// <exception cref="Exception">The last failing sub-env's exception, after the error lines are logged.</exception>
        public override VectorResetResult<TObs> Reset(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options = null) {
            ResetAsync(seeds, options);
            return ResetWait();
        }

        /// <summary>
        ///     Sends <c>reset</c> to every sub-env and returns at once (<c>reset_async(seed, options)</c>); <see cref="ResetWait"/>
        ///     collects the results. With <c>options["reset_mask"]</c>, the masked-out sub-envs answer with their last
        ///     observation instead.
        /// </summary>
        /// <param name="seed">The first sub-env's seed (sub-env <c>i</c> gets <c>seed + i</c>), or <see langword="null"/>.</param>
        /// <param name="options">The options, with an optional <c>reset_mask</c> (a <c>(num_envs,)</c> bool array with a <see langword="true"/>).</param>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="AlreadyPendingCallError">Thrown while another call is pending: <c>Calling `reset_async` while waiting for a pending call to `step` to complete</c>.</exception>
        /// <exception cref="TypeError">Thrown when <c>reset_mask</c> is not an array or not bool.</exception>
        /// <exception cref="ValueError">Thrown when <c>reset_mask</c> has another shape or no <see langword="true"/>.</exception>
        /// <exception cref="AttributeError">Thrown when a sub-env's worker ended with an error earlier (the pipe Python dropped).</exception>
        public void ResetAsync(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            AssertIsRunning();
            var seeds = new BigInteger?[NumEnvs];
            for (int i = 0; i < seeds.Length; i++)
                seeds[i] = seed is null ? null : seed.Value + i;
            ResetAsyncCore(seeds, options);
        }

        /// <summary>Sends <c>reset</c> with a seed per sub-env and returns at once (<c>reset_async(seed=[...])</c>).</summary>
        /// <param name="seeds">One seed per sub-env.</param>
        /// <param name="options">The options, with an optional <c>reset_mask</c>.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="seeds"/> is <see langword="null"/>.</exception>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="ValueError">Thrown when there is not one seed per sub-env (Gymnasium's message, before the pending-call check), or for a bad <c>reset_mask</c>.</exception>
        /// <exception cref="AlreadyPendingCallError">Thrown while another call is pending.</exception>
        /// <exception cref="TypeError">Thrown for a <c>reset_mask</c> that is not a bool array.</exception>
        /// <exception cref="AttributeError">Thrown when a sub-env's worker ended with an error earlier.</exception>
        public void ResetAsync(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options = null) {
            AssertIsRunning();
            if (seeds is null)
                throw new ArgumentNullException(nameof(seeds));
            if (seeds.Count != NumEnvs)
                throw new ValueError($"If seeds are passed as a list the length must match num_envs={NumEnvs} but got length={seeds.Count}.");
            ResetAsyncCore(seeds, options);
        }

        /// <summary>Collects the results of <see cref="ResetAsync(BigInteger?, IReadOnlyDictionary{string, object})"/> (<c>reset_wait(timeout)</c>).</summary>
        /// <param name="timeout">Seconds to wait for every sub-env, or <see langword="null"/> to wait as long as it takes.</param>
        /// <returns>The batched observations (a copy unless <see cref="Copy"/> is off) and the merged infos.</returns>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="NoAsyncCallError">Thrown when no reset is pending: <c>Calling `reset_wait` without any prior call to `reset_async`.</c></exception>
        /// <exception cref="TimeoutException">Thrown when a sub-env did not answer in time; the call stays with the workers, and the state returns to <see cref="AsyncState.Default"/>.</exception>
        /// <exception cref="AttributeError">Thrown when a sub-env's worker ended with an error earlier.</exception>
        /// <exception cref="Exception">The last failing sub-env's exception, after the error lines are logged.</exception>
        public VectorResetResult<TObs> ResetWait(double? timeout = null) {
            AssertIsRunning();
            if (!ReferenceEquals(_state, AsyncState.WaitingReset))
                throw new NoAsyncCallError("Calling `reset_wait` without any prior call to `reset_async`.", AsyncState.WaitingReset.Value);
            if (!PollPipeEnvs(timeout)) {
                _state = AsyncState.Default;
                throw new TimeoutException($"The call to `reset_wait` has timed out after {PyBuiltins.Str(timeout)} second(s).");
            }

            AsyncWorkerMessage[] messages = _workers.Select(worker => Recv(worker)).ToArray();
            RaiseIfErrors(messages.Select(message => message.Success).ToArray());

            var infos = new Info();
            var observations = new object?[NumEnvs];
            for (int i = 0; i < messages.Length; i++) {
                var result = (object?[])messages[i].Value!;
                observations[i] = result[0];
                infos = AddInfo(infos, (Info)result[1]!, i);
            }

            if (!SharedMemory)
                _observations = VectorUtils.Concatenate(SingleObservationSpace, observations, _observations);
            _state = AsyncState.Default;
            return new VectorResetResult<TObs>(ObservationsOut(), infos);
        }

        /// <summary>Steps every sub-env in parallel and waits for them (<c>step(actions)</c>).</summary>
        /// <param name="actions">The batched actions, one per sub-env, split by the batched action space (<c>iterate</c>).</param>
        /// <returns>The batched observations (a copy unless <see cref="Copy"/> is off), new reward and flag arrays, and the merged infos.</returns>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="AlreadyPendingCallError">Thrown while another call is pending.</exception>
        /// <exception cref="ValueError">Thrown when the actions are not one per sub-env (Python's <c>zip(strict=True)</c> text), after the paired sub-envs were sent theirs.</exception>
        /// <exception cref="Exception">The last failing sub-env's exception, after the error lines are logged.</exception>
        public override VectorStepResult<TObs> Step(TAct actions) {
            StepAsync(actions);
            return StepWait();
        }

        /// <summary>Sends each sub-env its action and returns at once (<c>step_async(actions)</c>); <see cref="StepWait"/> collects the results.</summary>
        /// <param name="actions">The batched actions, split by the batched action space; read by the workers after this returns, so keep them until the wait.</param>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="AlreadyPendingCallError">Thrown while another call is pending: <c>Calling `step_async` while waiting for a pending call to `reset` to complete.</c></exception>
        /// <exception cref="ValueError">Thrown when the actions are not one per sub-env, after the paired sub-envs were sent theirs (their results then wait in the workers' queues, as in Python's pipes).</exception>
        /// <exception cref="AttributeError">Thrown when a sub-env's worker ended with an error earlier.</exception>
        public void StepAsync(TAct actions) {
            AssertIsRunning();
            if (!ReferenceEquals(_state, AsyncState.Default))
                throw new AlreadyPendingCallError($"Calling `step_async` while waiting for a pending call to `{_state.Value}` to complete.", _state.Value);

            using IEnumerator<object?> iterActions = VectorUtils.Iterate(_batchedActionSpace, actions).GetEnumerator();
            // zip(self.parent_pipes, iter_actions, strict=True): each pair is sent as it is formed.
            for (int i = 0;; i++) {
                if (i == NumEnvs) {
                    if (iterActions.MoveNext())
                        throw new ValueError("zip() argument 2 is longer than argument 1");
                    break;
                }

                if (!iterActions.MoveNext())
                    throw new ValueError("zip() argument 2 is shorter than argument 1");
                Send(_workers[i], StepCommand, iterActions.Current);
            }

            _state = AsyncState.WaitingStep;
        }

        /// <summary>Collects the results of <see cref="StepAsync"/> (<c>step_wait(timeout)</c>).</summary>
        /// <param name="timeout">Seconds to wait for every sub-env, or <see langword="null"/> to wait as long as it takes.</param>
        /// <returns>The batched observations (a copy unless <see cref="Copy"/> is off), new float64 rewards and bool flags, and the merged infos.</returns>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="NoAsyncCallError">Thrown when no step is pending: <c>Calling `step_wait` without any prior call to `step_async`.</c></exception>
        /// <exception cref="TimeoutException">Thrown when a sub-env did not answer in time; the state returns to <see cref="AsyncState.Default"/>.</exception>
        /// <exception cref="AttributeError">Thrown when a sub-env's worker ended with an error earlier.</exception>
        /// <exception cref="Exception">The last failing sub-env's exception, after the error lines are logged.</exception>
        public VectorStepResult<TObs> StepWait(double? timeout = null) {
            AssertIsRunning();
            if (!ReferenceEquals(_state, AsyncState.WaitingStep))
                throw new NoAsyncCallError("Calling `step_wait` without any prior call to `step_async`.", AsyncState.WaitingStep.Value);
            if (!PollPipeEnvs(timeout)) {
                _state = AsyncState.Default;
                throw new TimeoutException($"The call to `step_wait` has timed out after {PyBuiltins.Str(timeout)} second(s).");
            }

            var observations = new List<object?>(NumEnvs);
            var rewards = new List<double>(NumEnvs);
            var terminations = new List<bool>(NumEnvs);
            var truncations = new List<bool>(NumEnvs);
            var infos = new Info();
            var successes = new bool[NumEnvs];
            for (int envIdx = 0; envIdx < NumEnvs; envIdx++) {
                AsyncWorkerMessage message = Recv(_workers[envIdx]);
                successes[envIdx] = message.Success;
                if (!message.Success)
                    continue;
                var envStepReturn = (object?[])message.Value!;
                observations.Add(envStepReturn[0]);
                rewards.Add((double)envStepReturn[1]!);
                terminations.Add((bool)envStepReturn[2]!);
                truncations.Add((bool)envStepReturn[3]!);
                infos = AddInfo(infos, (Info)envStepReturn[4]!, envIdx);
            }

            RaiseIfErrors(successes);

            if (!SharedMemory)
                _observations = VectorUtils.Concatenate(SingleObservationSpace, observations, _observations);
            _state = AsyncState.Default;
            return new VectorStepResult<TObs>(ObservationsOut(), np.array(rewards.ToArray()), np.array(terminations.ToArray()), np.array(truncations.ToArray()), infos);
        }

        /// <summary>
        ///     Calls a sub-env member by its Python name on every sub-env, in parallel, and waits (<c>call(name, *args)</c>):
        ///     an attribute's value (a delegate one is called with the arguments), or a method, called with the arguments.
        /// </summary>
        /// <param name="name">The Python name: <c>render</c>, <c>gravity</c>, a run-time attribute.</param>
        /// <param name="args">The positional arguments for a callable member.</param>
        /// <returns>One result per sub-env, in order.</returns>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="AlreadyPendingCallError">Thrown while another call is pending.</exception>
        /// <exception cref="Exception">The last failing sub-env's exception (an unknown attribute's <see cref="AttributeError"/>, <c>reset</c>/<c>step</c>/<c>close</c> refused with <see cref="ValueError"/>), after the error lines are logged.</exception>
        public IReadOnlyList<object?> Call(string name, params object?[] args) {
            CallAsync(name, args);
            return CallWait();
        }

        /// <summary>Sends a member call to every sub-env and returns at once (<c>call_async(name, *args)</c>); <see cref="CallWait"/> collects the results.</summary>
        /// <param name="name">The Python name.</param>
        /// <param name="args">The positional arguments.</param>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="AlreadyPendingCallError">Thrown while another call is pending: <c>Calling `call_async` while waiting for a pending call to `step` to complete.</c></exception>
        /// <exception cref="AttributeError">Thrown when a sub-env's worker ended with an error earlier.</exception>
        public void CallAsync(string name, params object?[] args) {
            AssertIsRunning();
            if (!ReferenceEquals(_state, AsyncState.Default))
                throw new AlreadyPendingCallError($"Calling `call_async` while waiting for a pending call to `{_state.Value}` to complete.", _state.Value);
            foreach (Worker worker in _workers)
                Send(worker, CallCommand, (name, args ?? Array.Empty<object?>()));
            _state = AsyncState.WaitingCall;
        }

        /// <summary>Collects the results of <see cref="CallAsync"/> (<c>call_wait(timeout)</c>).</summary>
        /// <param name="timeout">Seconds to wait for every sub-env, or <see langword="null"/> to wait as long as it takes.</param>
        /// <returns>One result per sub-env, in order.</returns>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="NoAsyncCallError">Thrown when no call is pending: <c>Calling `call_wait` without any prior call to `call_async`.</c></exception>
        /// <exception cref="TimeoutException">Thrown when a sub-env did not answer in time; the state returns to <see cref="AsyncState.Default"/>.</exception>
        /// <exception cref="AttributeError">Thrown when a sub-env's worker ended with an error earlier.</exception>
        /// <exception cref="Exception">The last failing sub-env's exception, after the error lines are logged.</exception>
        public IReadOnlyList<object?> CallWait(double? timeout = null) {
            AssertIsRunning();
            if (!ReferenceEquals(_state, AsyncState.WaitingCall))
                throw new NoAsyncCallError("Calling `call_wait` without any prior call to `call_async`.", AsyncState.WaitingCall.Value);
            if (!PollPipeEnvs(timeout)) {
                _state = AsyncState.Default;
                throw new TimeoutException($"The call to `call_wait` has timed out after {PyBuiltins.Str(timeout)} second(s).");
            }

            AsyncWorkerMessage[] messages = _workers.Select(worker => Recv(worker)).ToArray();
            RaiseIfErrors(messages.Select(message => message.Success).ToArray());
            _state = AsyncState.Default;
            return messages.Select(message => message.Value).ToArray();
        }

        /// <summary>Reads an attribute of every sub-env (<c>get_attr(name)</c>), as <see cref="Call"/> does without arguments.</summary>
        /// <param name="name">The Python name.</param>
        /// <returns>One value per sub-env.</returns>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="AlreadyPendingCallError">Thrown while another call is pending.</exception>
        /// <exception cref="Exception">The last failing sub-env's exception (an unknown attribute's <see cref="AttributeError"/>), after the error lines are logged.</exception>
        public IReadOnlyList<object?> GetAttr(string name) => Call(name);

        /// <summary>
        ///     Sets an attribute of every sub-env through its wrappers and waits (<c>set_attr(name, values)</c>): a list or tuple
        ///     gives one value per sub-env, anything else is set on all of them.
        /// </summary>
        /// <param name="name">The Python name.</param>
        /// <param name="values">A list or <see langword="object"/>?[] of one value per sub-env, or one value for all (an array is one value).</param>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="ValueError">Thrown when a list's length is not the number of sub-envs, with Gymnasium's message.</exception>
        /// <exception cref="AlreadyPendingCallError">Thrown while another call is pending.</exception>
        /// <exception cref="Exception">The last failing sub-env's exception, after the error lines are logged.</exception>
        public void SetAttr(string name, object? values) {
            AssertIsRunning();
            IList perEnv;
            if (values is IList list and not NDArray) {
                perEnv = list;
            } else {
                var repeated = new object?[NumEnvs];
                Array.Fill(repeated, values);
                perEnv = repeated;
            }

            if (perEnv.Count != NumEnvs)
                throw new ValueError($"Values must be a list or tuple with length equal to the number of environments. Got `{perEnv.Count}` values for {NumEnvs} environments.");
            if (!ReferenceEquals(_state, AsyncState.Default))
                throw new AlreadyPendingCallError($"Calling `set_attr` while waiting for a pending call to `{_state.Value}` to complete.", _state.Value);

            for (int i = 0; i < _workers.Length; i++)
                Send(_workers[i], SetAttrCommand, (name, perEnv[i]));
            RaiseIfErrors(_workers.Select(worker => Recv(worker).Success).ToArray());
        }

        /// <summary>Renders every sub-env (<c>render()</c>, Gymnasium's <c>call("render")</c>), in parallel.</summary>
        /// <returns>One result per sub-env, each by its own render mode.</returns>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        /// <exception cref="AlreadyPendingCallError">Thrown while another call is pending.</exception>
        public override IReadOnlyList<RenderResult>? Render() => Call("render").Select(frame => (RenderResult)frame!).ToArray();

        /// <summary>
        ///     Closes the vector env (<c>close(timeout=..., terminate=...)</c>): a pending call is collected first (within
        ///     <paramref name="timeout"/>, with Gymnasium's warning), then every worker closes its sub-env and ends.
        /// </summary>
        /// <param name="timeout">Seconds to wait for a pending call; when it runs out, the close becomes a terminating one.</param>
        /// <param name="terminate">
        ///     Whether to stop at once: Python kills its workers; threads cannot be killed, so each worker closes its sub-env as
        ///     soon as the call it is running returns, and this returns without waiting for that (disposal waits).
        /// </param>
        /// <exception cref="Exception">A pending call's sub-env exception, as Gymnasium's close lets it through (the env stays open).</exception>
        public void Close(double? timeout, bool terminate = false) {
            if (Closed)
                return;
            CloseExtras(timeout, terminate);
            Closed = true;
        }

        /// <summary>Closes without a timeout or termination: what <see cref="VectorEnv{TObs, TAct}.Close"/> runs (<c>close_extras()</c>).</summary>
        protected override void CloseExtras() => CloseExtras(null, false);

        /// <summary>
        ///     Closes (Python's <c>__del__</c> closes an open one with <c>terminate=True</c>), waits for every worker to close its
        ///     sub-env, then releases the threads, the buffers and the spaces this env made.
        /// </summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="VectorEnv{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            if (!Closed) {
                try {
                    Close(null, terminate: true);
                } catch (Exception e) {
                    // A pending call's sub-env error has no caller left to reach; Python prints such an error as ignored.
                    Logger.Error($"Exception ignored while disposing {this}: {e}");
                    Closed = true;
                }
            }

            base.Dispose(disposing);
            if (!disposing)
                return;
            Abandon();
        }

        /// <summary>The reset of <c>reset_async</c> once the seeds are one per sub-env: the pending-call check, the mask, then the sends.</summary>
        /// <param name="seeds">One seed per sub-env.</param>
        /// <param name="options">The options, possibly with <c>reset_mask</c>.</param>
        /// <exception cref="AlreadyPendingCallError">Thrown while another call is pending.</exception>
        /// <exception cref="TypeError">Thrown when <c>reset_mask</c> is not an array or not bool.</exception>
        /// <exception cref="ValueError">Thrown when <c>reset_mask</c> has another shape or no <see langword="true"/>.</exception>
        /// <exception cref="AttributeError">Thrown when a sub-env's worker ended with an error earlier.</exception>
        private void ResetAsyncCore(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options) {
            if (!ReferenceEquals(_state, AsyncState.Default))
                throw new AlreadyPendingCallError($"Calling `reset_async` while waiting for a pending call to `{_state.Value}` to complete", _state.Value);

            if (options is not null && options.TryGetValue("reset_mask", out object? maskValue)) {
                // The check hands back the caller's own mask, which the caller keeps and disposes.
#pragma warning disable NDW012
                NDArray resetMask = CheckResetMask(maskValue);
#pragma warning restore NDW012
                // The sub-envs get the options without the mask, as after Python's options.pop("reset_mask").
                var subOptions = new Info(options.Where(entry => entry.Key != "reset_mask"));
                for (int i = 0; i < _workers.Length; i++) {
                    if (resetMask.GetBoolean(i))
                        Send(_workers[i], ResetCommand, (seeds[i], (IReadOnlyDictionary<string, object?>?)subOptions));
                    else
                        Send(_workers[i], ResetNoopCommand, null);
                }
            } else {
                for (int i = 0; i < _workers.Length; i++)
                    Send(_workers[i], ResetCommand, (seeds[i], options));
            }

            _state = AsyncState.WaitingReset;
        }

        /// <summary>
        ///     <c>close_extras(timeout, terminate)</c>: a pending call is collected (its results discarded) within the timeout,
        ///     zero when terminating, a timeout turning the close into a terminating one; then the workers close their sub-envs,
        ///     waited for unless terminating, and the dedicated threads end.
        /// </summary>
        /// <param name="timeout">Seconds to wait for a pending call.</param>
        /// <param name="terminate">Whether to stop without waiting for the workers.</param>
        /// <exception cref="Exception">A pending call's sub-env exception, which Gymnasium does not catch here either.</exception>
        private void CloseExtras(double? timeout, bool terminate) {
            timeout = terminate ? 0 : timeout;
            try {
                if (!ReferenceEquals(_state, AsyncState.Default)) {
                    Logger.Warn($"Calling `close` while waiting for a pending call to `{_state.Value}` to complete.");
                    // getattr(self, f"{self._state.value}_wait")(timeout): the results are dropped, so their arrays are released.
                    switch (_state.Value) {
                        case ResetCommand:
                            Release(ResetWait(timeout));
                            break;
                        case StepCommand:
                            Release(StepWait(timeout));
                            break;
                        default:
                            CallWait(timeout);
                            break;
                    }
                }
            } catch (TimeoutException) {
                terminate = true;
            }

            if (terminate) {
                // Threads cannot be killed: each live worker closes its sub-env once the call it runs (if any) returns, and
                // nothing here waits for it; disposal does.
                foreach (Worker worker in _workers) {
                    if (worker.Detached || worker.ClosePosted)
                        continue;
                    worker.ClosePosted = true;
                    if (worker.ParentPipe is { } pipe) {
                        // A custom worker closes its sub-env once it reads the close; disposal joins its thread.
                        TrySendClose(worker, pipe);
                        continue;
                    }

                    _closing.Add(worker.Executor!.Post(() => Run(worker, CloseCommand, null)));
                }
            } else {
                foreach (Worker worker in _workers) {
                    if (!worker.Detached)
                        Send(worker, CloseCommand, null);
                }

                // pipe.recv() and process.join(): every message still queued (a timed-out call's included) and the close itself.
                foreach (Worker worker in _workers) {
                    if (worker.Detached)
                        continue;
                    while ((worker.ParentPipe is null ? worker.Pipe.Count : worker.Pending) > 0)
                        Recv(worker);
                }

                // pipe.close() and process.join() for the custom workers, which have ended their loops by now.
                foreach (Worker worker in _workers) {
                    worker.ParentPipe?.Close();
                    worker.Thread?.Join();
                }
            }

            foreach (WorkerThread thread in _threads) {
                thread.Complete();
                if (!terminate)
                    thread.Join();
            }
        }

        /// <summary>
        ///     Builds every sub-env on its worker, all at once, and waits for them; in thread-pool mode, then moves each sub-env
        ///     that renders to a window onto a dedicated thread (DR-22), since a window belongs to the thread that opens it.
        /// </summary>
        /// <exception cref="Exception">The lowest-indexed sub-env function's exception, when one throws.</exception>
        private void StartWorkers() {
            if (WorkerFunction is { } custom) {
                StartCustomWorkers(custom);
                return;
            }

            Task<AsyncWorkerMessage>[] started = _workers.Select(worker => worker.Executor!.Post(() => {
                // env = env_fn(): the worker builds its own sub-env, on its own thread.
                worker.State.Env = EnvFns[worker.Index]();
                return new AsyncWorkerMessage(null, true);
            })).ToArray();
            Exception? failure = null;
            foreach (Task<AsyncWorkerMessage> start in started) {
                try {
                    start.GetAwaiter().GetResult();
                } catch (Exception e) {
                    // Every start is waited for, so no worker is still building when the constructor gives up; the first
                    // failure in sub-env order is the one reported.
                    failure ??= e;
                }
            }

            if (failure is not null)
                ExceptionDispatchInfo.Capture(failure).Throw();

            if (ExecutionMode != AsyncExecutionMode.ThreadPool)
                return;
            foreach (Worker worker in _workers) {
                // Created on a pool thread, the env has opened no window yet: windows open at the first human render.
                if (worker.State.Env!.RenderMode is "human")
                    worker.Executor = NewThread($"Worker<{PyAttributes.PyTypeName(GetType())}>-{worker.Index}");
            }
        }

        /// <summary>
        ///     Starts one thread per sub-env running <paramref name="custom"/>, as Gymnasium starts one process per sub-env:
        ///     each gets a pipe to the parent and handles of its own on both ends, and the parent then closes its copy of the
        ///     worker's end. The sub-envs are built by the workers, so a failure shows at the first receive.
        /// </summary>
        /// <param name="custom">The custom worker.</param>
        private void StartCustomWorkers(AsyncVectorWorker custom) {
            foreach (Worker worker in _workers) {
                (PipeConnection parentPipe, PipeConnection childPipe) = PipeConnection.Pipe();
                // A process inherits handles of its own on both ends; so does the worker thread, so its closes are its own.
                PipeConnection workerPipe = childPipe.Duplicate(), workerParentPipe = parentPipe.Duplicate();
                int index = worker.Index;
                Func<IEnv> envFn = EnvFns[index];
                var thread = new Thread(() => {
                    try {
                        custom(index, envFn, workerPipe, workerParentPipe, _sharedMemory, _errorQueue, AutoresetMode, _semaphore);
                    } catch (Exception e) {
                        // What a process would print as its traceback on dying: kept for the parent's end-of-stream report.
                        worker.Escaped = e;
                    } finally {
                        // A process's handles close when it exits; a thread's must be closed for it.
                        workerPipe.Close();
                        workerParentPipe.Close();
                    }
                }) {
                    IsBackground = Daemon,
                    Name = $"Worker<{PyAttributes.PyTypeName(GetType())}>-{index}",
                };
                worker.ParentPipe = parentPipe;
                worker.Thread = thread;
                thread.Start();
                childPipe.Close();
            }
        }

        /// <summary>
        ///     <c>_check_spaces()</c>: every worker compares its sub-env's spaces with the batch's (equal spaces in
        ///     <c>"same"</c> mode, else a shared shape and dtype; equal action spaces).
        /// </summary>
        /// <exception cref="RuntimeError">Thrown when a sub-env does not fit, with Gymnasium's messages.</exception>
        /// <exception cref="Exception">A sub-env's exception while comparing, after the error lines are logged.</exception>
        private void CheckSpaces() {
            AssertIsRunning();
            foreach (Worker worker in _workers)
                Send(worker, CheckSpacesCommand, (ObservationMode, SingleObservationSpace, SingleActionSpace));
            AsyncWorkerMessage[] messages = _workers.Select(worker => Recv(worker)).ToArray();
            RaiseIfErrors(messages.Select(message => message.Success).ToArray());
            bool[][] results = messages.Select(message => (bool[])message.Value!).ToArray();

            if (!results.All(result => result[0])) {
                if (ObservationMode is "same")
                    throw new RuntimeError("AsyncVectorEnv(..., observation_mode='same') however some of the sub-environments observation spaces are not equivalent. If this is intentional, use `observation_mode='different'` instead.");
                throw new RuntimeError("AsyncVectorEnv(..., observation_mode='different' or custom space) however the sub-environment's observation spaces do not share a common shape and dtype.");
            }

            if (!results.All(result => result[1]))
                throw new RuntimeError($"Some environments have an action space different from `{SingleActionSpace}`. In order to batch actions, the action spaces from all environments must be equal.");
        }

        /// <summary>
        ///     One worker command, run on the worker's executor: the shared worker core (<see cref="AsyncWorkerCommands.Run"/>,
        ///     Gymnasium's <c>_async_worker</c> loop body) with this env's buffer, mode, permits and error queue. The shared
        ///     buffer is written with the batch's single observation space.
        /// </summary>
        /// <param name="worker">The worker.</param>
        /// <param name="command">The command.</param>
        /// <param name="data">The command's data.</param>
        /// <returns>The message: the result and <see langword="true"/>, or <see langword="null"/> and <see langword="false"/> after an error.</returns>
        private AsyncWorkerMessage Run(Worker worker, string command, object? data) =>
            AsyncWorkerCommands.Run(worker.State, command, data, SingleObservationSpace, _sharedMemory, AutoresetMode, _semaphore, _errorQueue);

        /// <summary>Sends a command to a worker (<c>pipe.send((command, data))</c>): its result queues in the worker's pipe, in order.</summary>
        /// <param name="worker">The worker.</param>
        /// <param name="command">The command.</param>
        /// <param name="data">Its data.</param>
        /// <exception cref="AttributeError">Thrown when the worker ended with an error earlier: Python's pipe is <c>None</c> then.</exception>
        /// <exception cref="OSError">
        ///     Thrown when a custom worker's pipe is broken (the worker ended), as Python's <c>send</c> fails with
        ///     <c>BrokenPipeError</c>; what escaped the worker, if anything, is the inner exception.
        /// </exception>
        private void Send(Worker worker, string command, object? data) {
            if (worker.Detached)
                throw new AttributeError("'NoneType' object has no attribute 'send'");
            if (command is CloseCommand)
                worker.ClosePosted = true;
            if (worker.ParentPipe is { } pipe) {
                try {
                    pipe.Send((command, data));
                } catch (OSError e) when (worker.Escaped is { } escaped) {
                    // A worker that died before this send closed its end; the pipe's lock orders its last writes first.
                    throw new OSError(e.Message, escaped);
                }

                worker.Pending++;
                return;
            }

            worker.Pipe.Enqueue(worker.Executor!.Post(() => Run(worker, command, data)));
        }

        /// <summary>Receives a worker's oldest pending result (<c>pipe.recv()</c>), waiting for it.</summary>
        /// <param name="worker">The worker.</param>
        /// <returns>The message.</returns>
        /// <exception cref="AttributeError">Thrown when the worker ended with an error earlier.</exception>
        /// <exception cref="InvalidOperationException">Thrown when nothing was sent to the worker (Python's <c>recv</c> would block forever).</exception>
        /// <exception cref="EndOfStreamException">Thrown, as Python's <c>EOFError</c>, when a custom worker ended without answering; what escaped it, if anything, is the inner exception.</exception>
        /// <exception cref="InvalidCastException">Thrown when a custom worker answered with something else than an <c>(object?, bool)</c> value tuple.</exception>
        private static AsyncWorkerMessage Recv(Worker worker) {
            if (worker.Detached)
                throw new AttributeError("'NoneType' object has no attribute 'recv'");
            if (worker.ParentPipe is { } pipe) {
                if (worker.Pending == 0)
                    throw new InvalidOperationException($"Nothing was sent to Worker-{worker.Index}, so there is nothing to receive.");
                object? received;
                try {
                    received = pipe.Recv();
                } catch (EndOfStreamException e) when (worker.Escaped is { } escaped) {
                    // The pipe's lock orders the worker's last writes before this receive, so Escaped is complete here.
                    throw new EndOfStreamException(e.Message, escaped);
                }

                worker.Pending--;
                (object? value, bool success) = ((object?, bool))received!;
                return new AsyncWorkerMessage(value, success);
            }

            if (worker.Pipe.Count == 0)
                throw new InvalidOperationException($"Nothing was sent to Worker-{worker.Index}, so there is nothing to receive.");
            // The work runs on the pool or a worker thread and never needs the caller's context, so waiting here cannot deadlock.
            return worker.Pipe.Dequeue().GetAwaiter().GetResult();
        }

        /// <summary>
        ///     <c>_poll_pipe_envs(timeout)</c>: without a timeout, <see langword="true"/> at once (the receives then wait);
        ///     otherwise whether every worker answered before the deadline.
        /// </summary>
        /// <param name="timeout">Seconds, or <see langword="null"/>.</param>
        /// <returns>Whether the results can be received.</returns>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>.</exception>
        private bool PollPipeEnvs(double? timeout) {
            AssertIsRunning();
            if (timeout is null)
                return true;

            long deadline = Stopwatch.GetTimestamp() + (long)(timeout.Value * Stopwatch.Frequency);
            foreach (Worker worker in _workers) {
                double delta = Math.Max((double)(deadline - Stopwatch.GetTimestamp()) / Stopwatch.Frequency, 0);
                if (worker.Detached)
                    return false;
                if (worker.ParentPipe is { } pipe) {
                    // A custom worker's pipe polls as Python's does; an ended worker counts as readable (its end of stream).
                    if (!pipe.Poll(delta))
                        return false;
                    continue;
                }

                if (worker.Pipe.Count == 0) {
                    // pipe.poll(delta) waits the whole delta for data that was never asked for.
                    Thread.Sleep(TimeSpan.FromSeconds(delta));
                    return false;
                }

                if (Task.WaitAny(new Task[] { worker.Pipe.Peek() }, TimeSpan.FromSeconds(delta)) < 0)
                    return false;
            }

            return true;
        }

        /// <summary>
        ///     <c>_raise_if_errors(successes)</c>: for every failed worker, in the order they failed, Gymnasium's two error lines
        ///     and the worker detached; the last failure's exception is rethrown after the closing line.
        /// </summary>
        /// <param name="successes">One flag per worker.</param>
        /// <exception cref="Exception">The last failing sub-env's exception, as it was thrown.</exception>
        /// <exception cref="RuntimeError">Thrown when a worker reported a failure but queued no error (an invariant broken by a command sent to an ended worker).</exception>
        private void RaiseIfErrors(IReadOnlyList<bool> successes) {
            if (successes.All(success => success))
                return;

            int numErrors = NumEnvs - successes.Count(success => success);
            for (int i = 0; i < numErrors; i++) {
                if (!_errorQueue.TryDequeue(out AsyncWorkerError error))
                    throw new RuntimeError("A worker reported a failure without queuing its error.");
                Logger.Error($"Received the following error from Worker-{error.Index} - Shutting it down");
                Logger.Error(error.Trace);
                _workers[error.Index].Detached = true;
                if (i == numErrors - 1) {
                    Logger.Error("Raising the last exception back to the main process.");
                    _state = AsyncState.Default;
                    ExceptionDispatchInfo.Capture(error.Exception).Throw();
                }
            }
        }

        /// <summary><c>_assert_is_running()</c>.</summary>
        /// <exception cref="ClosedEnvironmentError">Thrown after <see cref="VectorEnv{TObs, TAct}.Close"/>, with Gymnasium's message.</exception>
        private void AssertIsRunning() {
            if (Closed)
                throw new ClosedEnvironmentError($"Trying to operate on `{PyAttributes.PyTypeName(GetType())}`, after a call to `close()`.");
        }

        /// <summary>The observation batch handed out: a deep copy the caller owns, or the buffer itself without <see cref="Copy"/>.</summary>
        /// <returns>The batch.</returns>
        private TObs ObservationsOut() {
            object? batch = Copy ? PyCopy.DeepCopy(_observations) : _observations;
            return (TObs)batch!;
        }

        /// <summary>The executor a worker starts on: a pool chain, a shared dedicated thread, or a thread of its own.</summary>
        /// <param name="index">The worker.</param>
        /// <returns>The executor.</returns>
        private IWorkerExecutor ExecutorFor(int index) {
            string name = $"Worker<{PyAttributes.PyTypeName(GetType())}>";
            switch (ExecutionMode) {
                case AsyncExecutionMode.DedicatedThreads: {
                    int slot = index % ThreadCount!.Value;
                    // The threads are made in slot order as their first sub-env asks, so slot t is _threads[t].
                    while (_threads.Count <= slot)
                        NewThread($"{name}-thread-{_threads.Count}");
                    return _threads[slot];
                }
                case AsyncExecutionMode.ThreadPerEnv:
                    return NewThread($"{name}-{index}");
                default:
                    return new PoolExecutor();
            }
        }

        /// <summary>
        ///     Sends a custom worker its close without waiting for the answer, when its pipe still takes it: a worker that already
        ///     ended has nothing to close.
        /// </summary>
        /// <param name="worker">The worker.</param>
        /// <param name="pipe">The parent's end of its pipe.</param>
        private static void TrySendClose(Worker worker, PipeConnection pipe) {
            try {
                pipe.Send((CloseCommand, (object?)null));
                worker.Pending++;
            } catch (OSError) {
                // The worker ended and closed its end: nothing listens, and nothing is left to close.
            }
        }

        /// <summary>Starts a dedicated worker thread, remembered so closing ends it; a background thread unless <see cref="Daemon"/> is off.</summary>
        /// <param name="name">The thread's name.</param>
        /// <returns>The thread.</returns>
        private WorkerThread NewThread(string name) {
            var thread = new WorkerThread(name, Daemon);
            _threads.Add(thread);
            return thread;
        }

        /// <summary>
        ///     Releases everything this env made, without the close protocol: the sub-envs (closed by their workers, waited
        ///     for), the threads, the semaphore, the buffers and the spaces. Used by disposal and by a constructor that fails.
        /// </summary>
        private void Abandon() {
            // A constructor that failed, or a disposal of an env whose close never ran, still owes each worker its close; the
            // detached ones closed their sub-envs when they failed.
            foreach (Worker worker in _workers) {
                if (worker.Detached || worker.ClosePosted)
                    continue;
                worker.ClosePosted = true;
                if (worker.ParentPipe is { } pipe) {
                    TrySendClose(worker, pipe);
                    continue;
                }

                _closing.Add(worker.Executor!.Post(() => Run(worker, CloseCommand, null)));
            }

            // A custom worker reads its close (or the end of its pipe, once the parent's handle closes) and ends; its thread
            // is waited for, as Gymnasium joins its processes.
            foreach (Worker worker in _workers) {
                worker.ParentPipe?.Close();
                worker.Thread?.Join();
            }

            foreach (Task closing in _closing) {
                try {
                    closing.Wait();
                } catch (AggregateException) {
                    // The close work reports its own failures through the error queue; nothing is left to tell.
                }
            }

            _closing.Clear();
            foreach (WorkerThread thread in _threads)
                thread.Dispose();
            _threads.Clear();
            _semaphore?.Dispose();
            if (SharedMemory) {
                // The batch read from the shared buffer is views of it; the buffer goes with them.
                Release(_observations);
                VectorUtils.ReleaseSharedMemory(_sharedMemory);
            } else {
                Release(_observations);
            }

            _observations = null;
            _batchedObservationSpace?.Dispose();
            _batchedActionSpace?.Dispose();
            foreach (Space space in _ownedSpaces)
                space.Dispose();
            _ownedSpaces.Clear();
        }

        /// <summary>Keeps a space this env made, so disposal releases it.</summary>
        /// <param name="space">The space.</param>
        /// <returns>The space.</returns>
        private Space Own(Space space) {
            _ownedSpaces.Add(space);
            return space;
        }

        /// <summary>
        ///     The batched and single observation spaces of an observation mode: the first sub-env's space batched
        ///     (<c>"same"</c>), every sub-env's space batched together (<c>"different"</c>, each function called once more), or
        ///     the pair given.
        /// </summary>
        /// <param name="mode">The mode.</param>
        /// <param name="dummyEnv">The env built from the first function.</param>
        /// <returns>The batched space and the single one.</returns>
        /// <exception cref="TypeError">Thrown when a pair holds something else than spaces.</exception>
        /// <exception cref="ValueError">Thrown for any other mode.</exception>
        private (Space Batched, Space Single) ObservationSpaces(object mode, IEnv dummyEnv) {
            object?[]? pair = mode switch {
                object?[] array when array.Length == 2 => array,
                ITuple { Length: 2 } tuple => new[] { tuple[0], tuple[1] },
                _ => null,
            };
            if (pair is not null) {
                if (pair[0] is not Space batched || pair[1] is not Space single)
                    throw new TypeError($"Expected both elements of observation_mode to be Spaces, got {PyBuiltins.TypeRepr(pair[0])} and {PyBuiltins.TypeRepr(pair[1])}");
                // Python unpacks the pair as (observation_space, single_observation_space); the caller keeps both.
                return (batched, single);
            }

            if (mode is "same") {
                Space single = Own(ObjectSpace.Unview(dummyEnv.ObservationSpace)!.DeepCopy());
                return (VectorUtils.BatchSpace(single, NumEnvs), single);
            }

            if (mode is "different") {
                // [env().observation_space for env in self.env_fns]: each env is built for its space and dropped; the spaces
                // are copied, since disposing an env disposes its own.
                var envSpaces = new Space[EnvFns.Count];
                for (int i = 0; i < envSpaces.Length; i++) {
                    using IEnv env = EnvFns[i]();
                    envSpaces[i] = Own(ObjectSpace.Unview(env.ObservationSpace)!.DeepCopy());
                }

                return (VectorUtils.BatchDifferingSpaces(envSpaces), envSpaces[0]);
            }

            throw new ValueError($"Invalid `observation_mode`, expected: 'same' or 'different' or tuple of single and batch observation space, actual got {PyBuiltins.Str(mode)}");
        }

        /// <summary>Gymnasium's checks of <c>options["reset_mask"]</c>: an array, of shape <c>(num_envs,)</c>, of bool, with a <see langword="true"/>.</summary>
        /// <param name="value">The mask given.</param>
        /// <returns>The mask.</returns>
        /// <exception cref="TypeError">Thrown when it is not an array, or not bool.</exception>
        /// <exception cref="ValueError">Thrown when its shape is wrong or nothing is masked.</exception>
        private NDArray CheckResetMask(object? value) {
            if (value is not NDArray resetMask)
                throw new TypeError($"`options['reset_mask']` must be a numpy array, got {PyBuiltins.TypeRepr(value)}");
            if (resetMask.ndim != 1 || resetMask.shape[0] != NumEnvs)
                throw new ValueError($"`options['reset_mask']` must have shape `({NumEnvs},)`, got {PyBuiltins.Repr(resetMask.shape.Select(dim => (object?)dim).ToArray())}");
            if (resetMask.typecode != NPTypeCode.Boolean)
                throw new TypeError($"`options['reset_mask']` must have `dtype=np.bool_`, got {resetMask.dtype.name}");
            if (!np.any(resetMask))
                throw new ValueError($"`options['reset_mask']` must contain a boolean array with at least one True value, got reset_mask={PyBuiltins.Str(resetMask)}");
            return resetMask;
        }

        /// <summary>A batched space as the typed property needs it: as it is, or viewed as <see langword="object"/> elements for the untyped env.</summary>
        /// <typeparam name="T">The batched element type.</typeparam>
        /// <param name="space">The batched space.</param>
        /// <param name="attribute">The Python attribute, for the error.</param>
        /// <returns>The typed space.</returns>
        /// <exception cref="TypeError">Thrown when the space's elements are not <typeparamref name="T"/>.</exception>
        private Space<T> Typed<T>(Space space, string attribute) {
            if (space is Space<T> typed)
                return typed;
            if (typeof(T) == typeof(object))
                return (Space<T>)(object)new ObjectSpace(space);
            throw new TypeError($"{PyAttributes.PyTypeName(GetType())}<{typeof(TObs).Name}, {typeof(TAct).Name}> needs an {attribute} of {typeof(T).Name} elements, but batching gives {space}; use the untyped AsyncVectorEnv or the batched types");
        }

        /// <summary>
        ///     Gymnasium's <c>max_concurrency</c> checks: <see langword="null"/>, or an integer (Python's <c>int</c> or
        ///     <c>np.integer</c>, so a bool too) of at least 1.
        /// </summary>
        /// <param name="value">The value given.</param>
        /// <returns>The limit, or <see langword="null"/>.</returns>
        /// <exception cref="ValueError">Thrown for a value of another type or below 1, with Gymnasium's messages.</exception>
        private static int? CheckMaxConcurrency(object? value) {
            if (value is null)
                return null;
            BigInteger? integer = value switch {
                bool flag => flag ? 1 : 0,
                sbyte or byte or short or ushort or int or uint or long or ulong => new BigInteger(Convert.ToDecimal(value, System.Globalization.CultureInfo.InvariantCulture)),
                BigInteger big => big,
                NDArray { ndim: 0 } array when array.typecode is NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32
                    or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 => new BigInteger(Convert.ToDecimal(array.GetAtIndex(0), System.Globalization.CultureInfo.InvariantCulture)),
                _ => null,
            };
            if (integer is null)
                throw new ValueError($"`max_concurrency` must be an integer or None, got {PyBuiltins.TypeRepr(value)}.");
            if (integer < 1)
                throw new ValueError($"`max_concurrency` must be a positive or None, got {PyBuiltins.Str(value)}.");
            // More permits than sub-envs limit nothing; the semaphore's count is an int.
            return integer > int.MaxValue ? int.MaxValue : (int)integer.Value;
        }

        /// <summary>The dedicated thread count: <see cref="Environment.ProcessorCount"/> by default in <see cref="AsyncExecutionMode.DedicatedThreads"/> mode, none in the others.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="threadCount">The count given, or <see langword="null"/>.</param>
        /// <returns>The count, or <see langword="null"/> outside <see cref="AsyncExecutionMode.DedicatedThreads"/>.</returns>
        /// <exception cref="ValueError">Thrown for a count outside that mode, or one below 1.</exception>
        private static int? CheckThreadCount(AsyncExecutionMode mode, int? threadCount) {
            if (mode != AsyncExecutionMode.DedicatedThreads) {
                if (threadCount is not null)
                    throw new ValueError($"`thread_count` applies to AsyncExecutionMode.DedicatedThreads only, not {mode}.");
                return null;
            }

            int count = threadCount ?? Environment.ProcessorCount;
            if (count < 1)
                throw new ValueError($"`thread_count` must be a positive integer, got {count}.");
            return count;
        }

        /// <summary>Disposes the arrays of a dropped result: its observations, rewards, flags and infos.</summary>
        /// <param name="result">The result.</param>
        private static void Release(VectorStepResult<TObs> result) {
            Release(result.Observations);
            result.Rewards.Dispose();
            result.Terminations.Dispose();
            result.Truncations.Dispose();
            Release(result.Infos);
        }

        /// <summary>Disposes the arrays of a dropped reset result.</summary>
        /// <param name="result">The result.</param>
        private static void Release(VectorResetResult<TObs> result) {
            Release(result.Observations);
            Release(result.Infos);
        }

        /// <summary>Disposes the arrays of a value: an array, or the arrays inside a tuple or dict.</summary>
        /// <param name="value">The value.</param>
        private static void Release(object? value) {
            switch (value) {
                case NDArray array:
                    array.Dispose();
                    break;
                case object?[] items:
                    foreach (object? item in items)
                        Release(item);
                    break;
                case IReadOnlyDictionary<string, object?> dict:
                    foreach (object? item in dict.Values)
                        Release(item);
                    break;
            }
        }

        /// <summary>
        ///     One sub-env's worker (Python's worker process and the parent's end of its pipe). A built-in worker posts each
        ///     command to its executor and keeps the worker loop's state (<see cref="State"/>, touched only on the executor) and
        ///     the results in order (<see cref="Pipe"/>, touched only by the caller's thread); a custom worker (DR-38) runs on
        ///     its own <see cref="Thread"/> and talks through <see cref="ParentPipe"/>.
        /// </summary>
        private sealed class Worker {
            /// <summary>A built-in worker for sub-env <paramref name="index"/>.</summary>
            /// <param name="index">The sub-env.</param>
            /// <param name="executor">Where its commands run.</param>
            public Worker(int index, IWorkerExecutor executor) {
                Index = index;
                Executor = executor;
                State = new AsyncWorkerState(index);
            }

            /// <summary>A custom worker's slot for sub-env <paramref name="index"/>; its pipe and thread are set when it starts.</summary>
            /// <param name="index">The sub-env.</param>
            public Worker(int index) {
                Index = index;
                State = new AsyncWorkerState(index);
            }

            /// <summary>The sub-env's index.</summary>
            public int Index { get; }

            /// <summary>Where a built-in worker's commands run, replaced once, before any command, for a thread-bound sub-env in pool mode; <see langword="null"/> for a custom worker.</summary>
            public IWorkerExecutor? Executor { get; set; }

            /// <summary>The built-in worker loop's state (the sub-env, the pending autoreset, the last observation); unused by a custom worker, which keeps its own.</summary>
            public AsyncWorkerState State { get; }

            /// <summary>A built-in worker's results, oldest first (the pipe's unread messages).</summary>
            public Queue<Task<AsyncWorkerMessage>> Pipe { get; } = new();

            /// <summary>The parent's end of a custom worker's pipe, or <see langword="null"/> for a built-in worker.</summary>
            public PipeConnection? ParentPipe { get; set; }

            /// <summary>The thread a custom worker runs on, or <see langword="null"/> for a built-in worker.</summary>
            public Thread? Thread { get; set; }

            /// <summary>The commands sent to a custom worker and not answered yet; read and written by the caller only.</summary>
            public int Pending { get; set; }

            /// <summary>What escaped a custom worker when it ended (its process's dying traceback), or <see langword="null"/>; written by its thread before its pipe closes.</summary>
            public Exception? Escaped { get; set; }

            /// <summary>Whether the caller dropped the pipe after the worker failed (Python sets it to <c>None</c>).</summary>
            public bool Detached { get; set; }

            /// <summary>Whether the caller already sent the worker its close (through the pipe or, terminating, without waiting); read and written by the caller only.</summary>
            public bool ClosePosted { get; set; }
        }

        /// <summary>Runs a worker's commands one after another and hands back each result.</summary>
        private interface IWorkerExecutor {
            /// <summary>Queues a command after the ones already queued.</summary>
            /// <param name="work">The command; it does not throw (a worker reports failures as messages).</param>
            /// <returns>Its result, once it ran.</returns>
            Task<AsyncWorkerMessage> Post(Func<AsyncWorkerMessage> work);
        }

        /// <summary>
        ///     <see cref="AsyncExecutionMode.ThreadPool"/>: each command runs as a pool task continuing the previous one, so a
        ///     sub-env's commands never overlap while different sub-envs' run in parallel.
        /// </summary>
        private sealed class PoolExecutor : IWorkerExecutor {
            /// <summary>The last command queued.</summary>
            private Task _tail = Task.CompletedTask;

            /// <summary>Queues a command after the previous one, on the thread pool.</summary>
            /// <param name="work">The command.</param>
            /// <returns>Its result.</returns>
            public Task<AsyncWorkerMessage> Post(Func<AsyncWorkerMessage> work) {
                Task<AsyncWorkerMessage> next = _tail.ContinueWith(_ => work(), CancellationToken.None, TaskContinuationOptions.DenyChildAttach, TaskScheduler.Default);
                _tail = next;
                return next;
            }
        }

        /// <summary>A long-lived background thread that runs the commands posted to it in order, for every worker pinned to it.</summary>
        private sealed class WorkerThread : IWorkerExecutor, IDisposable {
            /// <summary>The commands not run yet.</summary>
            private readonly BlockingCollection<Action> _queue = new();

            /// <summary>The thread.</summary>
            private readonly Thread _thread;

            /// <summary>Starts the thread.</summary>
            /// <param name="name">Its name, for debuggers.</param>
            /// <param name="background">Whether it is a background thread (Gymnasium's <c>daemon</c>): one that never keeps the process alive.</param>
            public WorkerThread(string name, bool background) {
                _thread = new Thread(Loop) { IsBackground = background, Name = name };
                _thread.Start();
            }

            /// <summary>Queues a command after the ones already queued on this thread.</summary>
            /// <param name="work">The command.</param>
            /// <returns>Its result; completed off this thread, so a waiter never runs on it.</returns>
            /// <exception cref="InvalidOperationException">Thrown after <see cref="Complete"/>.</exception>
            public Task<AsyncWorkerMessage> Post(Func<AsyncWorkerMessage> work) {
                var completion = new TaskCompletionSource<AsyncWorkerMessage>(TaskCreationOptions.RunContinuationsAsynchronously);
                _queue.Add(() => {
                    try {
                        completion.SetResult(work());
                    } catch (Exception e) {
                        completion.SetException(e);
                    }
                });
                return completion.Task;
            }

            /// <summary>Accepts no more commands; the thread ends after the queued ones.</summary>
            public void Complete() => _queue.CompleteAdding();

            /// <summary>Waits for the thread to end (after <see cref="Complete"/>).</summary>
            public void Join() => _thread.Join();

            /// <summary>Ends the thread after its queued commands, waits for it, and releases the queue.</summary>
            public void Dispose() {
                _queue.CompleteAdding();
                _thread.Join();
                _queue.Dispose();
            }

            /// <summary>The thread's body: the commands, in order, until the queue is completed.</summary>
            private void Loop() {
                foreach (Action command in _queue.GetConsumingEnumerable())
                    command();
            }
        }
    }

    /// <summary>
    ///     <see cref="AsyncVectorEnv{TObs, TAct}"/> untyped: observations and actions are <see langword="object"/>s, whatever
    ///     the spaces, as Gymnasium's own. The untyped <c>MakeVec(..., vectorizationMode: "async")</c> builds this one.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/async_vector_env.py#L56">gymnasium/vector/async_vector_env.py</see>.</remarks>
    public class AsyncVectorEnv : AsyncVectorEnv<object?, object?> {
        /// <summary>Builds the workers and the batched spaces; see <see cref="AsyncVectorEnv{TObs, TAct}"/>'s constructor.</summary>
        /// <param name="envFns">One function per sub-env, each called by its worker; the first also once by the constructor.</param>
        /// <param name="sharedMemory">Whether the workers write their observations into one shared buffer.</param>
        /// <param name="copy">Whether every returned observation batch is a deep copy.</param>
        /// <param name="observationMode"><c>"same"</c> (<see langword="null"/>), <c>"different"</c>, or the pair of batched and single spaces.</param>
        /// <param name="autoresetMode">An <see cref="AutoresetMode"/> or its value; <see langword="null"/> is <c>NEXT_STEP</c>.</param>
        /// <param name="maxConcurrency">How many sub-envs may run <c>reset</c> or <c>step</c> at once, or <see langword="null"/>.</param>
        /// <param name="executionMode">Where the calls run (DR-22).</param>
        /// <param name="threadCount">The number of threads of <see cref="AsyncExecutionMode.DedicatedThreads"/>.</param>
        /// <param name="context">Gymnasium's start method: accepted and kept, with nothing to choose.</param>
        /// <param name="daemon">Whether the dedicated threads are background threads (the default).</param>
        /// <param name="worker">A custom worker serving each sub-env on a thread of its own (DR-38), or <see langword="null"/> for the built-in workers.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="envFns"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for an unknown mode, a bad <c>max_concurrency</c> or thread count, or a space with no shared layout.</exception>
        /// <exception cref="IndexError">Thrown when there are no sub-envs.</exception>
        /// <exception cref="TypeError">Thrown when a pair observation mode holds something else than spaces, or a <c>Graph</c>/<c>Sequence</c> space is shared.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch.</exception>
        /// <exception cref="EndOfStreamException">Thrown under a custom worker that ended without answering.</exception>
        public AsyncVectorEnv(IEnumerable<Func<IEnv>> envFns, bool sharedMemory = true, bool copy = true, object? observationMode = null, object? autoresetMode = null,
            object? maxConcurrency = null, AsyncExecutionMode executionMode = AsyncExecutionMode.ThreadPool, int? threadCount = null, string? context = null,
            bool daemon = true, AsyncVectorWorker? worker = null)
            : base(envFns, sharedMemory, copy, observationMode, autoresetMode, maxConcurrency, executionMode, threadCount, context, daemon, worker) { }

        /// <summary>
        ///     The default worker (Gymnasium's <c>_async_worker</c>, DR-38), public so a custom <see cref="AsyncVectorWorker"/>
        ///     can wrap it: builds the sub-env with <paramref name="envFn"/>, closes its handle on <paramref name="parentPipe"/>,
        ///     then answers each <c>(command, data)</c> received on <paramref name="pipe"/> with <c>(result, success)</c>, in
        ///     order, until a <c>close</c> or a failure; the sub-env is closed and disposed on the way out. The vector env's
        ///     built-in workers run this same loop body through their executors.
        /// </summary>
        /// <param name="index">The sub-env's index.</param>
        /// <param name="envFn">Builds the sub-env, which this worker owns.</param>
        /// <param name="pipe">The worker's end of the pipe.</param>
        /// <param name="parentPipe">The worker's handle on the parent's end, closed first, as Gymnasium's worker does.</param>
        /// <param name="sharedMemory">The shared observation buffer to write into, or <see langword="null"/> to send observations through the pipe.</param>
        /// <param name="errorQueue">Where a failure is reported before the <c>(null, false)</c> answer.</param>
        /// <param name="autoresetMode">When an ended sub-env resets.</param>
        /// <param name="semaphore">The concurrency permits held around <c>reset</c> and <c>step</c>, or <see langword="null"/>.</param>
        /// <exception cref="Exception">
        ///     What <paramref name="envFn"/> throws, before any answer (as in Gymnasium, where the process then dies and the
        ///     parent sees the end of the pipe), or what answering through a broken pipe throws.
        /// </exception>
        /// <remarks>
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/async_vector_env.py#L823">gymnasium/vector/async_vector_env.py</see>.
        /// </remarks>
        public static void AsyncWorker(int index, Func<IEnv> envFn, PipeConnection pipe, PipeConnection parentPipe, object? sharedMemory,
            ConcurrentQueue<AsyncWorkerError> errorQueue, AutoresetMode autoresetMode, SemaphoreSlim? semaphore) =>
            AsyncWorkerCommands.Serve(index, envFn, pipe, parentPipe, sharedMemory, errorQueue, autoresetMode, semaphore);
    }
}
