// Gym.Internal is 0.2's code, [Obsolete] since 1.0 and removed in 2.0 with Gym.Legacy (DR-56): it keeps no nullable
// context (DR-54), rather than 89 annotations for code on its way out.
#nullable disable
using System;

namespace Gym.Collections {
    /// <summary>
    ///     Thrown by the legacy <see cref="Dict"/>'s inline-pairs constructor and <see cref="Dict.AddInline"/> when the
    ///     arguments are not key-value pairs: an odd count, a <see langword="null"/> key, or a key that is not exactly a
    ///     <see cref="string"/>. The pairs before the bad one are already in the dictionary when it is thrown.
    /// </summary>
    /// <remarks>
    ///     Obsolete with <see cref="Dict"/> (DR-56) and removed in 2.0. <see cref="Gym.Envs.Info"/> needs no such error:
    ///     its collection initializer (<c>new Info { ["key"] = value }</c>) is checked when it compiles.
    /// </remarks>
    [Serializable]
    [Obsolete("Thrown only by the legacy Dict's inline pairs; Gym.Envs.Info needs none (new Info { [\"key\"] = value } is checked when it compiles). Removed in 2.0 with Gym.Legacy.")]
    public class DictInlineInitializationException : Exception {
        /// <summary>An error without a message of its own (.NET's default text).</summary>
        public DictInlineInitializationException() { }

        /// <summary>An error saying which rule the inline pairs broke.</summary>
        /// <param name="message">The rule broken, as <see cref="Dict.AddInline"/> words it.</param>
        public DictInlineInitializationException(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="inner">The underlying cause.</param>
        public DictInlineInitializationException(string message, Exception inner) : base(message, inner) { }
    }
}
