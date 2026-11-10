#nullable enable
namespace Gym.Vector {
    /// <summary>
    ///     Where an <see cref="AsyncVectorEnv{TObs, TAct}"/> runs its sub-envs' calls (DR-22). Gymnasium runs each sub-env in
    ///     its own process; Gym.NET runs them on threads of the calling process, and the mode decides which. Every mode gives
    ///     the same results, gathered in sub-env order, so the choice is about throughput and thread affinity only.
    /// </summary>
    /// <remarks>
    ///     A sub-env that renders to a window (<c>render_mode == "human"</c>) is bound to the thread that opened it; in
    ///     <see cref="ThreadPool"/> mode such a sub-env gets a dedicated thread of its own automatically, the others stay on
    ///     the pool.
    /// </remarks>
    public enum AsyncExecutionMode {
        /// <summary>
        ///     The default: each call runs as a thread-pool task, one sub-env's calls one after another. Cheapest for many
        ///     short steps; a sub-env never keeps a thread while it waits for its next call.
        /// </summary>
        ThreadPool,

        /// <summary>
        ///     A fixed set of long-lived threads (<c>threadCount</c>, by default <see cref="System.Environment.ProcessorCount"/>),
        ///     sub-env <c>i</c> pinned to thread <c>i % threadCount</c> for its life: the sub-envs sharing a thread run one after
        ///     another, so fewer threads than sub-envs bound the parallelism. Suits envs that keep thread-local state.
        /// </summary>
        DedicatedThreads,

        /// <summary>
        ///     One long-lived thread per sub-env, as Gymnasium has one process per sub-env: the most parallel, and a thread per
        ///     sub-env for as long as the vector env is open.
        /// </summary>
        ThreadPerEnv,
    }
}
