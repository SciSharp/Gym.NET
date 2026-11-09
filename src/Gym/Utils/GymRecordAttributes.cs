#nullable enable
using System;

namespace Gym.Utils {
    /// <summary>
    ///     Asks Gym.NET's source generator for an options record (DR-34): on a <c>partial</c> env or wrapper class it writes a
    ///     <c>&lt;Class&gt;Options</c> record with one property per parameter of the public constructor <c>make</c> binds to (a
    ///     wrapper's <c>env</c> excepted), its <see cref="IGymOptions.ToKwargs"/>, and a constructor of the class that takes
    ///     the record. <c>Gymnasium.Make("CartPole-v1", new CartPoleEnvOptions { RenderMode = "rgb_array" })</c> is then
    ///     checked by the compiler where a dictionary of kwargs is not.
    /// </summary>
    /// <remarks>
    ///     The record follows the constructor on every build, so it cannot drift from it. Only the properties that are set
    ///     become kwargs: the registered spec's values and the constructor's defaults stand for the others. A class that is
    ///     not <c>partial</c> fails to compile (CS0260), as the generator adds the constructor to it.
    /// </remarks>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public sealed class GymOptionsAttribute : Attribute { }

    /// <summary>
    ///     Makes a <c>partial</c> positional record a Python tuple (DR-13): the generator writes implicit conversions to and from
    ///     <see langword="object"/>?[] (Gym.NET's tuple, what a <c>Tuple</c> space samples), Python's <c>==</c> member by member,
    ///     and Python's <c>repr</c>, <c>(1, 2)</c>. <c>[GymTuple] public partial record BlackjackObservation(NDArray PlayerSum,
    ///     NDArray DealerCard, NDArray UsableAce);</c> reads a Blackjack observation by name.
    /// </summary>
    /// <remarks>
    ///     The conversions share the members' values (arrays are not copied). Equality is Python's, not the record's:
    ///     arrays compare by value, so the hash is the member count, the only thing equal records always share.
    /// </remarks>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public sealed class GymTupleAttribute : Attribute { }

    /// <summary>
    ///     Makes a <c>partial</c> positional record a Python dict (DR-13): the generator writes implicit conversions to and from
    ///     <c>Info</c> (what a <c>Dict</c> space samples), keyed by each member's Python name (<c>[PyName]</c>, else its
    ///     snake_case), Python's <c>==</c> and <c>repr</c>.
    /// </summary>
    /// <remarks>Converting from a dict without one of the keys raises <see cref="NumSharp.KeyError"/>, as Python's indexing does.</remarks>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public sealed class GymDictAttribute : Attribute { }
}
