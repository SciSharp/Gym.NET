#nullable enable
using Gym.Utils;

namespace Gym.Envs.Registration {
    /// <summary>
    ///     How <c>MakeVec</c> vectorizes an env: <see cref="Sync"/> steps copies made with <c>Make</c> one after another
    ///     (<c>SyncVectorEnv</c>), <see cref="Async"/> runs them in parallel on worker threads (<c>AsyncVectorEnv</c>, DR-22),
    ///     and <see cref="VectorEntryPoint"/> builds the env's own vectorized class from its spec's <c>vector_entry_point</c>
    ///     (<c>CartPoleVectorEnv</c>), which is also the default whenever a spec has one.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A C# enum (DR-77, DR-84) that stands for Gymnasium's Python <c>Enum</c> (<see cref="PyEnumAttribute"/>), as
    ///     <see cref="Gym.Vector.AutoresetMode"/> does: <see cref="VectorizeModes"/> gives a member's Python name, value,
    ///     <c>str</c> and <c>repr</c> (<c>&lt;VectorizeMode.SYNC: 'sync'&gt;</c>) and parses <c>VectorizeMode("sync")</c>;
    ///     <c>MakeVec</c> takes a member or its string.
    ///     </para>
    ///     <para>
    ///     Footgun: a member's <see cref="object.ToString"/> is its C# name (<c>"Sync"</c>), not Python's value
    ///     (<c>"sync"</c>); use <see cref="VectorizeModes.Value"/> for the string Gymnasium records.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L245">gymnasium/envs/registration.py</see>.</para>
    /// </remarks>
    [PyEnum("VectorizeMode")]
    public enum VectorizeMode {
        /// <summary>Gymnasium's <c>VectorizeMode.ASYNC</c> (<c>"async"</c>): sub-envs in parallel, on worker threads in Gym.NET (DR-22).</summary>
        [PyName("ASYNC"), PyValue("async")] Async,

        /// <summary>Gymnasium's <c>VectorizeMode.SYNC</c> (<c>"sync"</c>): sub-envs one after another on the calling thread.</summary>
        [PyName("SYNC"), PyValue("sync")] Sync,

        /// <summary>Gymnasium's <c>VectorizeMode.VECTOR_ENTRY_POINT</c> (<c>"vector_entry_point"</c>): the env's own vectorized class.</summary>
        [PyName("VECTOR_ENTRY_POINT"), PyValue("vector_entry_point")] VectorEntryPoint,
    }
}
