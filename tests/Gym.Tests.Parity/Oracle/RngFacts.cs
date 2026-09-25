using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp;

namespace Gym.Tests.Parity.Oracle {
    /// <summary>
    ///     Describes a NumSharp generator's state in the shape NumPy's <c>bit_generator.state</c> has, so the harness can
    ///     compare Gym.NET's generator with Gymnasium's <c>np_random</c> by the generic <see cref="ByteContract"/>.
    /// </summary>
    /// <remarks>
    ///     The state catches a missing or extra draw at the call where it happens, even before any drawn value differs
    ///     (plan §5): two PCG64 generators agree on <c>state</c> exactly when they made the same number of 64-bit draws from the
    ///     same seed, and on <c>has_uint32</c>/<c>uinteger</c> when they also agree on the buffered half of a 32-bit draw.
    /// </remarks>
    public static class RngFacts {
        /// <summary>
        ///     Describes <paramref name="generator"/>'s bit generator the way NumPy's <c>PCG64.state</c> property returns it:
        ///     <c>{'bit_generator': 'PCG64', 'state': {'state': int, 'inc': int}, 'has_uint32': int, 'uinteger': int}</c>,
        ///     keys in that order, integers as Python ints.
        /// </summary>
        /// <param name="generator">The generator; only read.</param>
        /// <returns>The description.</returns>
        /// <exception cref="NotSupportedException">Thrown for a bit generator other than PCG64, which no Gymnasium env uses.</exception>
        public static ValueFacts Of(Generator generator) {
            if (generator.bit_generator is not PCG64 pcg)
                throw new NotSupportedException($"Only PCG64 states are described; this generator uses {generator.bit_generator.GetType().Name}.");
            PCG64.Pcg64StateData state = pcg.state;
            return ValueFacts.FromDict(new[] {
                Entry("bit_generator", ValueFacts.FromText("PCG64")),
                Entry("state", ValueFacts.FromDict(new[] {
                    Entry("state", PythonInt(state.state)),
                    Entry("inc", PythonInt(state.inc)),
                })),
                // NumPy stores the flag as a C int, which the property returns as a Python int, not a bool.
                Entry("has_uint32", PythonInt(state.has_uint32 ? 1 : 0)),
                Entry("uinteger", PythonInt(state.uinteger)),
            });
        }

        /// <summary>A dictionary entry with a string key.</summary>
        /// <param name="key">The key.</param>
        /// <param name="value">The value.</param>
        /// <returns>The entry.</returns>
        private static KeyValuePair<ValueFacts, ValueFacts> Entry(string key, ValueFacts value) => new(ValueFacts.FromText(key), value);

        /// <summary>An integer as a Python <c>int</c> (no dtype), which is what the state dictionary holds.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The description.</returns>
        private static ValueFacts PythonInt(BigInteger value) => ValueFacts.FromInteger(value, "", "int");
    }
}
