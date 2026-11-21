#nullable enable
using System;
using System.Collections.Generic;
using Gym.Utils;
using NumSharp;

namespace Gym.Envs.Registration {
    /// <summary>
    ///     Python's view of <see cref="VectorizeMode"/>'s members (DR-84): their <c>.name</c>, <c>.value</c>, <c>str</c> and
    ///     <c>repr</c>, and Gymnasium's <c>VectorizeMode(value)</c> call. A C# enum cannot carry members of its own, so they
    ///     live here, beside it.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L245">gymnasium/envs/registration.py</see>.</remarks>
    public static class VectorizeModes {
        /// <summary>Every member, in Gymnasium's order: <c>ASYNC</c>, <c>SYNC</c>, <c>VECTOR_ENTRY_POINT</c>, the order <c>make_vec</c>'s error lists them in.</summary>
        /// <remarks>A shared, read-only list.</remarks>
        public static IReadOnlyList<VectorizeMode> All => PyEnums.Members<VectorizeMode>();

        /// <summary>Python's <c>VectorizeMode(value)</c>: the member with that value (<c>"sync"</c>), never its name (<c>"SYNC"</c>).</summary>
        /// <param name="value">The value.</param>
        /// <returns>The member.</returns>
        /// <exception cref="ValueError">Thrown for any other value: <c>'x' is not a valid VectorizeMode</c>.</exception>
        public static VectorizeMode Of(string? value) => PyEnums.Of<VectorizeMode>(value);

        /// <summary>Python's <c>.name</c>: <c>ASYNC</c>, <c>SYNC</c> or <c>VECTOR_ENTRY_POINT</c>.</summary>
        /// <param name="mode">The member.</param>
        /// <returns>The name.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        public static string Name(VectorizeMode mode) => PyEnums.Name(mode);

        /// <summary>Python's <c>.value</c>: <c>"async"</c>, <c>"sync"</c> or <c>"vector_entry_point"</c>, what a <c>MakeVec</c> spec records.</summary>
        /// <param name="mode">The member.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        public static string Value(VectorizeMode mode) => PyEnums.Value(mode);

        /// <summary>Python's <c>str(mode)</c>: <c>VectorizeMode.SYNC</c>.</summary>
        /// <param name="mode">The member.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        public static string Str(VectorizeMode mode) => PyEnums.Str(mode);

        /// <summary>Python's <c>repr(mode)</c>: <c>&lt;VectorizeMode.SYNC: 'sync'&gt;</c>.</summary>
        /// <param name="mode">The member.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        public static string Repr(VectorizeMode mode) => PyEnums.Repr(mode);
    }
}
