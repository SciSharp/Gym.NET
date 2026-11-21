#nullable enable
using Gym.Utils;

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
    ///     A C# enum (DR-77, DR-84) that stands for Gymnasium's Python <c>Enum</c> (<see cref="PyEnumAttribute"/>): Python's
    ///     view of a member is <see cref="AutoresetModes"/>'s, named <c>NEXT_STEP</c>, valued <c>"NextStep"</c>, printed
    ///     <c>&lt;AutoresetMode.NEXT_STEP: 'NextStep'&gt;</c> inside a printed metadata dict (<see cref="PyBuiltins.Repr"/>),
    ///     and parsed from its value only (<see cref="AutoresetModes.Of"/>; the member name is refused, as Python refuses it).
    ///     The vector envs' constructors take a member or its string (<c>autoresetMode: "SameStep"</c>), as Python's do.
    ///     </para>
    ///     <para>
    ///     Footgun: a member's <see cref="object.ToString"/> is its C# name (<c>"NextStep"</c>), not Python's <c>str</c>
    ///     (<c>"AutoresetMode.NEXT_STEP"</c>); messages format members through <see cref="AutoresetModes.Str"/>, and a member
    ///     boxed in a metadata dict compares with <see cref="object.Equals(object)"/> or a pattern
    ///     (<c>mode is AutoresetMode.SameStep</c>), never with <see cref="object.ReferenceEquals"/>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/vector_env.py#L37">gymnasium/vector/vector_env.py</see>.</para>
    /// </remarks>
    [PyEnum("AutoresetMode")]
    public enum AutoresetMode {
        /// <summary>Gymnasium's <c>AutoresetMode.NEXT_STEP</c> (<c>"NextStep"</c>): the step after an episode's end resets the sub-env and returns its first observation, with a reward of 0.</summary>
        [PyName("NEXT_STEP"), PyValue("NextStep")] NextStep,

        /// <summary>Gymnasium's <c>AutoresetMode.SAME_STEP</c> (<c>"SameStep"</c>): the ending step resets at once, returning the new episode's first observation and the ended one's last in <c>final_obs</c>.</summary>
        [PyName("SAME_STEP"), PyValue("SameStep")] SameStep,

        /// <summary>Gymnasium's <c>AutoresetMode.DISABLED</c> (<c>"Disabled"</c>): no reset at all; stepping an ended sub-env without resetting it through <c>reset_mask</c> fails.</summary>
        [PyName("DISABLED"), PyValue("Disabled")] Disabled,
    }
}
