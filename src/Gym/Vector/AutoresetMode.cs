#nullable enable
using System;
using System.Collections.Generic;
using Gym.Utils;
using NumSharp;

namespace Gym.Vector {
    /// <summary>
    ///     When a vector env resets a sub-env whose episode ended: on the next step (<see cref="NextStep"/>, Gymnasium's
    ///     default), within the same step (<see cref="SameStep"/>, the ended episode's last observation and info moved into
    ///     <c>final_obs</c>/<c>final_info</c>), or never (<see cref="Disabled"/>, the caller resets through
    ///     <c>reset_mask</c>). A vector env states its mode in <c>metadata["autoreset_mode"]</c>, and training code must handle
    ///     the mode it gets: the same step's observation means different things in each.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A class with three instances rather than a C# <see langword="enum"/>, so it behaves as Python's <c>Enum</c> does
    ///     where it shows: compared by identity, named <c>NEXT_STEP</c>, valued <c>"NextStep"</c>, printed
    ///     <c>&lt;AutoresetMode.NEXT_STEP: 'NextStep'&gt;</c> inside a printed metadata dict, and parsed from its value only
    ///     (<c>AutoresetMode("NextStep")</c>; the member name is refused, as Python refuses it). A string converts implicitly,
    ///     so <c>autoresetMode: "SameStep"</c> reads as Python's call does.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/vector_env.py#L37">gymnasium/vector/vector_env.py</see>.</para>
    /// </remarks>
    public sealed class AutoresetMode {
        /// <summary>Gymnasium's <c>AutoresetMode.NEXT_STEP</c>: the step after an episode's end resets the sub-env and returns its first observation, with a reward of 0.</summary>
        public static readonly AutoresetMode NextStep = new("NEXT_STEP", "NextStep");

        /// <summary>Gymnasium's <c>AutoresetMode.SAME_STEP</c>: the ending step resets at once, returning the new episode's first observation and the ended one's last in <c>final_obs</c>.</summary>
        public static readonly AutoresetMode SameStep = new("SAME_STEP", "SameStep");

        /// <summary>Gymnasium's <c>AutoresetMode.DISABLED</c>: no reset at all; stepping an ended sub-env without resetting it through <c>reset_mask</c> fails.</summary>
        public static readonly AutoresetMode Disabled = new("DISABLED", "Disabled");

        /// <summary>The members in Gymnasium's declaration order, which <see cref="Of"/> searches and its error lists.</summary>
        private static readonly AutoresetMode[] Members = { NextStep, SameStep, Disabled };

        /// <summary>Creates one of the three members.</summary>
        /// <param name="name">The Python member name.</param>
        /// <param name="value">The Python value.</param>
        private AutoresetMode(string name, string value) {
            Name = name;
            Value = value;
        }

        /// <summary>Python's <c>.name</c>: <c>NEXT_STEP</c>, <c>SAME_STEP</c> or <c>DISABLED</c>.</summary>
        public string Name { get; }

        /// <summary>Python's <c>.value</c>: <c>"NextStep"</c>, <c>"SameStep"</c> or <c>"Disabled"</c>, what a <c>make_vec</c> spec records.</summary>
        public string Value { get; }

        /// <summary>
        ///     Python's <c>str(mode)</c>, what an f-string shows: <c>AutoresetMode.SAME_STEP</c>. It differs from the
        ///     <c>repr</c> (<see cref="ToString"/>), which a printed dict shows, so messages that Gymnasium formats with an
        ///     f-string use this one.
        /// </summary>
        public string Str => $"AutoresetMode.{Name}";

        /// <summary>Every member, in Gymnasium's order.</summary>
        public static IReadOnlyList<AutoresetMode> All => Members;

        /// <summary>
        ///     Python's <c>AutoresetMode(value)</c>: the member with that value. Only the value is accepted (<c>"NextStep"</c>),
        ///     not the name (<c>"NEXT_STEP"</c>), as Python's <c>Enum</c> call does.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The member.</returns>
        /// <exception cref="ValueError">Thrown for any other value: <c>'x' is not a valid AutoresetMode</c>.</exception>
        public static AutoresetMode Of(string? value) {
            foreach (AutoresetMode member in Members) {
                if (string.Equals(member.Value, value, StringComparison.Ordinal))
                    return member;
            }

            throw new ValueError($"{PyBuiltins.Repr(value)} is not a valid AutoresetMode");
        }

        /// <summary>
        ///     The member an argument stands for, as Gymnasium's constructors read it
        ///     (<c>mode if isinstance(mode, AutoresetMode) else AutoresetMode(mode)</c>).
        /// </summary>
        /// <param name="mode">A member, its value as a string, or <see langword="null"/> for the default <see cref="NextStep"/>.</param>
        /// <returns>The member.</returns>
        /// <exception cref="ValueError">Thrown for a string no member has as its value, or another kind of value (Python's <c>Enum</c> lookup fails for it too).</exception>
        public static AutoresetMode From(object? mode) => mode switch {
            null => NextStep,
            AutoresetMode member => member,
            string value => Of(value),
            _ => throw new ValueError($"{PyBuiltins.Repr(mode)} is not a valid AutoresetMode"),
        };

        /// <summary>A string converts to the member with that value, so <c>autoresetMode: "SameStep"</c> reads as Python's call.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The member.</returns>
        /// <exception cref="ValueError">Thrown when no member has that value.</exception>
        public static implicit operator AutoresetMode(string value) => Of(value);

        /// <summary>Python's <c>repr</c> of the member: <c>&lt;AutoresetMode.NEXT_STEP: 'NextStep'&gt;</c>, what a printed metadata dict shows.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => $"<AutoresetMode.{Name}: '{Value}'>";
    }
}
