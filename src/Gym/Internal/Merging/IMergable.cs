// Gym.Internal is 0.2's code, [Obsolete] since 1.0 and removed in 2.0 with Gym.Legacy (DR-56): it keeps no nullable
// context (DR-54), rather than 89 annotations for code on its way out.
#nullable disable
using System;
using System.Collections.Generic;

namespace Gym.Merging {
    /// <summary>
    ///     A string-keyed container the legacy deep merge (<see cref="Merger"/>) can read and write key by key: the legacy
    ///     <see cref="Gym.Collections.Dict"/> is the one implementation. <see cref="IsExpandable"/> decides whether a merge
    ///     may add keys or only overwrite existing ones.
    /// </summary>
    /// <remarks>
    ///     Obsolete with <see cref="Gym.Collections.Dict"/> (DR-56) and removed in 2.0; <see cref="Gym.Envs.Info"/> is the
    ///     2.0 dictionary, merged with <see cref="Gym.Envs.Info.Update"/>.
    /// </remarks>
    [Obsolete("The legacy Dict's deep merge: use Gym.Envs.Info.Update (Python's dict.update: the new value wins, one level deep). Removed in 2.0 with Gym.Legacy.")]
    public interface IMergable {
        /// <summary>
        ///     Performs a merge with object of unknown type: a dictionary, another <see cref="IMergable"/>, or a sequence of
        ///     string-keyed pairs. Anything else merges nothing, silently.
        /// </summary>
        /// <param name="right">object to take data from</param>
        /// <param name="preference">see the documentation on the enum values.</param>
        void Merge(object right, MergePrefer preference);

        /// <summary>The key-value pairs, in the implementation's order; the merge does not read it.</summary>
        IEnumerable<KeyValuePair<string, object>> Contents { get; }

        /// <summary>
        ///     Assign <paramref name="value"/> to given <paramref name="key"/>.
        /// </summary>
        /// <param name="key">The key to set.</param>
        /// <param name="value">The value, stored as given (not copied).</param>
        /// <exception cref="KeyNotFoundException">If <see cref="IsExpandable"/> is false and When the key is not present</exception>
        void Set(string key, object value);

        /// <summary>
        ///     Gets the value of <paramref name="key"/>.
        /// </summary>
        /// <param name="key">The key to read.</param>
        /// <returns>The value stored under <paramref name="key"/>.</returns>
        /// <exception cref="KeyNotFoundException">When <paramref name="key"/> is not present (the legacy <see cref="Gym.Collections.Dict"/> indexes directly).</exception>
        object Get(string key);

        /// <summary>
        ///     All current keys of this object.
        /// </summary>
        IEnumerable<string> Keys { get; }

        /// <summary>
        ///     Is it possible to add values of keys that do not present in <see cref="Keys"/>?
        /// </summary>
        bool IsExpandable { get; }

        /// <summary>
        ///     The number of keys; a merge from an <see cref="IMergable"/> with none returns at once.
        /// </summary>
        int Count { get; }
    }

    /// <summary>
    ///     An <see cref="IMergable"/> with a typed merge entry point, which the legacy <see cref="Gym.Collections.Dict"/>
    ///     implements for dictionaries, lists and other mergeables.
    /// </summary>
    /// <typeparam name="T">The type of value the merge takes.</typeparam>
    /// <remarks>Obsolete with <see cref="Gym.Collections.Dict"/> (DR-56) and removed in 2.0.</remarks>
    [Obsolete("The legacy Dict's deep merge: use Gym.Envs.Info.Update (Python's dict.update: the new value wins, one level deep). Removed in 2.0 with Gym.Legacy.")]
    public interface IMergable<in T> : IMergable {
        /// <summary>
        ///     Performs a merge with object of <typeparamref name="T"/> type.
        /// </summary>
        /// <param name="right">object to take data from</param>
        /// <param name="preference">see the documentation on the enum values.</param>
        void Merge(T right, MergePrefer preference);
    }
}
