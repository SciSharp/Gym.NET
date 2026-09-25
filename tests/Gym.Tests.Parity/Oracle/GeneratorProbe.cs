using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NumSharp;

namespace Gym.Tests.Parity.Oracle {
    /// <summary>
    ///     Finds the <see cref="Generator"/> a Gym.NET env or space draws from, so its state can be compared with Gymnasium's
    ///     <c>np_random</c>. Gym.NET does not expose it yet (Gymnasium's <c>np_random</c> property arrives with P1), so this
    ///     reads the one instance field of type <see cref="Generator"/> by reflection.
    /// </summary>
    /// <remarks>
    ///     The field is found by type, not by name (<c>random</c> in CartPole, a property's backing field in LunarLander,
    ///     <c>NpRandom</c> in the spaces), so renames do not silently switch the check off. When the product exposes
    ///     <c>NpRandom</c> publicly, this probe should give way to it.
    /// </remarks>
    public static class GeneratorProbe {
        /// <summary>Finds the single generator of <paramref name="target"/>.</summary>
        /// <param name="target">The env or space.</param>
        /// <param name="generator">The generator, when <see langword="true"/> is returned.</param>
        /// <param name="reason">Why none was found, when <see langword="false"/> is returned.</param>
        /// <returns><see langword="true"/> when exactly one non-null generator field exists.</returns>
        public static bool TryFind(object target, out Generator? generator, out string reason) {
            var fields = new List<FieldInfo>();
            for (Type? type = target.GetType(); type != null && type != typeof(object); type = type.BaseType)
                fields.AddRange(type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly).Where(f => f.FieldType == typeof(Generator)));

            if (fields.Count != 1) {
                generator = null;
                reason = fields.Count == 0
                    ? $"{target.GetType().Name} has no Generator field"
                    : $"{target.GetType().Name} has {fields.Count} Generator fields ({string.Join(", ", fields.Select(f => f.Name))}), so which one is np_random is ambiguous";
                return false;
            }

            generator = (Generator?)fields[0].GetValue(target);
            reason = generator is null ? $"{target.GetType().Name}.{fields[0].Name} is null (not seeded yet)" : "";
            return generator != null;
        }

        /// <summary>Describes the state of <paramref name="target"/>'s generator, failing the test when it has none.</summary>
        /// <param name="target">The env or space.</param>
        /// <returns>The state, in <c>bit_generator.state</c> form.</returns>
        /// <exception cref="InvalidOperationException">Thrown when no single generator is found.</exception>
        public static ValueFacts StateOf(object target) {
            if (!TryFind(target, out Generator? generator, out string reason))
                throw new InvalidOperationException(reason);
            return RngFacts.Of(generator!);
        }
    }
}
