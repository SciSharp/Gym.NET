#nullable enable
using System;
using System.Collections.Generic;
using Gym.Utils;
using NumSharp;

namespace Gym.Vector {
    /// <summary>
    ///     Python's view of <see cref="AutoresetMode"/>'s members (DR-84): their <c>.name</c>, <c>.value</c>, <c>str</c> and
    ///     <c>repr</c>, Gymnasium's <c>AutoresetMode(value)</c> call, and the reading of a vector env's <c>autoreset_mode</c>
    ///     argument. A C# enum cannot carry members of its own, so they live here, beside it.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/vector_env.py#L37">gymnasium/vector/vector_env.py</see>.</remarks>
    public static class AutoresetModes {
        /// <summary>Every member, in Gymnasium's order: <c>NEXT_STEP</c>, <c>SAME_STEP</c>, <c>DISABLED</c> (<c>list(AutoresetMode)</c>).</summary>
        /// <remarks>A shared, read-only list.</remarks>
        public static IReadOnlyList<AutoresetMode> All => PyEnums.Members<AutoresetMode>();

        /// <summary>
        ///     Python's <c>AutoresetMode(value)</c>: the member with that value. Only the value is accepted (<c>"NextStep"</c>),
        ///     not the Python name (<c>"NEXT_STEP"</c>) nor the C# one, as Python's <c>Enum</c> call accepts values only.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The member.</returns>
        /// <exception cref="ValueError">Thrown for any other value: <c>'x' is not a valid AutoresetMode</c>.</exception>
        public static AutoresetMode Of(string? value) => PyEnums.Of<AutoresetMode>(value);

        /// <summary>
        ///     The member an argument stands for, as Gymnasium's vector envs read their <c>autoreset_mode</c>
        ///     (<c>mode if isinstance(mode, AutoresetMode) else AutoresetMode(mode)</c>).
        /// </summary>
        /// <param name="mode">A member, its value as a string, or <see langword="null"/> for the default <see cref="AutoresetMode.NextStep"/>.</param>
        /// <returns>The member.</returns>
        /// <exception cref="ValueError">
        ///     Thrown for a string no member has as its value, a member the enum does not declare (a cast integer), or another
        ///     kind of value (Python's <c>Enum</c> lookup fails for it too): <c>'x' is not a valid AutoresetMode</c>.
        /// </exception>
        public static AutoresetMode From(object? mode) => mode switch {
            null => AutoresetMode.NextStep,
            AutoresetMode member when Enum.IsDefined(member) => member,
            string value => Of(value),
            _ => throw new ValueError($"{PyBuiltins.Repr(mode)} is not a valid AutoresetMode"),
        };

        /// <summary>Python's <c>.name</c>: <c>NEXT_STEP</c>, <c>SAME_STEP</c> or <c>DISABLED</c>.</summary>
        /// <param name="mode">The member.</param>
        /// <returns>The name.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        public static string Name(AutoresetMode mode) => PyEnums.Name(mode);

        /// <summary>Python's <c>.value</c>: <c>"NextStep"</c>, <c>"SameStep"</c> or <c>"Disabled"</c>, what a <c>make_vec</c> spec's kwargs may hold.</summary>
        /// <param name="mode">The member.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        public static string Value(AutoresetMode mode) => PyEnums.Value(mode);

        /// <summary>
        ///     Python's <c>str(mode)</c>, what an f-string shows: <c>AutoresetMode.SAME_STEP</c>. It differs from the
        ///     <c>repr</c> (<see cref="Repr"/>), which a printed dict shows, so messages that Gymnasium formats with an f-string
        ///     use this one.
        /// </summary>
        /// <param name="mode">The member.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        public static string Str(AutoresetMode mode) => PyEnums.Str(mode);

        /// <summary>Python's <c>repr(mode)</c>: <c>&lt;AutoresetMode.NEXT_STEP: 'NextStep'&gt;</c>, what a printed metadata dict shows.</summary>
        /// <param name="mode">The member.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        public static string Repr(AutoresetMode mode) => PyEnums.Repr(mode);
    }
}
