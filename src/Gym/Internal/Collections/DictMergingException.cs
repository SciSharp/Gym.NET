// Gym.Internal is 0.2's code, [Obsolete] since 1.0 and removed in 2.0 with Gym.Legacy (DR-56): it keeps no nullable
// context (DR-54), rather than 89 annotations for code on its way out.
#nullable disable
using System;

namespace Gym.Collections {
    /// <summary>
    ///     Thrown by the legacy deep merge (<see cref="Gym.Merging.Merger"/>, <see cref="Dict.Merge(System.Collections.IDictionary, Gym.Merging.MergePrefer)"/>)
    ///     when merging one key fails; the message names the key and <see cref="Exception.InnerException"/> holds the
    ///     cause. Keys merged before the failing one stay merged: the target is left half-merged.
    /// </summary>
    /// <remarks>
    ///     Obsolete with <see cref="Dict"/> (DR-56) and removed in 2.0. <see cref="Gym.Envs.Info.Update"/>, its
    ///     replacement, overwrites one level deep and has no failure of its own to report.
    /// </remarks>
    [Serializable]
    [Obsolete("Thrown only by the legacy Dict's deep merge; Gym.Envs.Info.Update needs none. Removed in 2.0 with Gym.Legacy.")]
    public class DictMergingException : Exception {
        /// <summary>An error without a message of its own (.NET's default text).</summary>
        public DictMergingException() { }

        /// <summary>An error naming the key whose merge failed.</summary>
        /// <param name="message">The message, naming the key.</param>
        public DictMergingException(string message) : base(message) { }

        /// <summary>An error naming the key whose merge failed, with the failure itself.</summary>
        /// <param name="message">The message, naming the key.</param>
        /// <param name="inner">The exception the key's merge threw.</param>
        public DictMergingException(string message, Exception inner) : base(message, inner) { }
    }
}
