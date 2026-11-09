#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Utils;
using NumSharp;

namespace Gym.Envs.Registration {
    /// <summary>
    ///     How <c>MakeVec</c> vectorizes an env: <see cref="Sync"/> steps copies made with <c>Make</c> one after another
    ///     (<c>SyncVectorEnv</c>), <see cref="Async"/> would run them in parallel (<c>AsyncVectorEnv</c>, P6.2, not available
    ///     yet), and <see cref="VectorEntryPoint"/> builds the env's own vectorized class from its spec's
    ///     <c>vector_entry_point</c> (<c>CartPoleVectorEnv</c>), which is also the default whenever a spec has one.
    /// </summary>
    /// <remarks>
    ///     A class with three instances, as <see cref="Gym.Vector.AutoresetMode"/>, so it reads, prints and parses as Python's
    ///     <c>Enum</c>: <c>VectorizeMode("sync")</c> is <see cref="Of"/>, and a string converts implicitly. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L245">gymnasium/envs/registration.py</see>.
    /// </remarks>
    public sealed class VectorizeMode {
        /// <summary>Gymnasium's <c>VectorizeMode.ASYNC</c> (<c>"async"</c>): sub-envs in parallel. Not available until P6.2.</summary>
        public static readonly VectorizeMode Async = new("ASYNC", "async");

        /// <summary>Gymnasium's <c>VectorizeMode.SYNC</c> (<c>"sync"</c>): sub-envs one after another on the calling thread.</summary>
        public static readonly VectorizeMode Sync = new("SYNC", "sync");

        /// <summary>Gymnasium's <c>VectorizeMode.VECTOR_ENTRY_POINT</c> (<c>"vector_entry_point"</c>): the env's own vectorized class.</summary>
        public static readonly VectorizeMode VectorEntryPoint = new("VECTOR_ENTRY_POINT", "vector_entry_point");

        /// <summary>The members in Gymnasium's declaration order, which errors list.</summary>
        private static readonly VectorizeMode[] Members = { Async, Sync, VectorEntryPoint };

        /// <summary>Creates one of the three members.</summary>
        /// <param name="name">The Python member name.</param>
        /// <param name="value">The Python value.</param>
        private VectorizeMode(string name, string value) {
            Name = name;
            Value = value;
        }

        /// <summary>Python's <c>.name</c>: <c>ASYNC</c>, <c>SYNC</c> or <c>VECTOR_ENTRY_POINT</c>.</summary>
        public string Name { get; }

        /// <summary>Python's <c>.value</c>: <c>"async"</c>, <c>"sync"</c> or <c>"vector_entry_point"</c>, what a <c>MakeVec</c> spec records.</summary>
        public string Value { get; }

        /// <summary>Python's <c>str(mode)</c>: <c>VectorizeMode.SYNC</c>.</summary>
        public string Str => $"VectorizeMode.{Name}";

        /// <summary>Every member, in Gymnasium's order.</summary>
        public static IReadOnlyList<VectorizeMode> All => Members;

        /// <summary>Python's <c>VectorizeMode(value)</c>: the member with that value (not its name).</summary>
        /// <param name="value">The value.</param>
        /// <returns>The member.</returns>
        /// <exception cref="ValueError">Thrown for any other value: <c>'x' is not a valid VectorizeMode</c>.</exception>
        public static VectorizeMode Of(string? value) =>
            Members.FirstOrDefault(member => string.Equals(member.Value, value, StringComparison.Ordinal)) ?? throw new ValueError($"{PyBuiltins.Repr(value)} is not a valid VectorizeMode");

        /// <summary>A string converts to the member with that value.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The member.</returns>
        /// <exception cref="ValueError">Thrown when no member has that value.</exception>
        public static implicit operator VectorizeMode(string value) => Of(value);

        /// <summary>Python's <c>repr</c>: <c>&lt;VectorizeMode.SYNC: 'sync'&gt;</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => $"<VectorizeMode.{Name}: '{Value}'>";
    }
}
