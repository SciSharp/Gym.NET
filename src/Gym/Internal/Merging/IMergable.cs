// Gym.Internal is 0.2's code, under review for deletion (R5 of docs/plans/CSHARP_CODE_REVIEW.md): it keeps no nullable
// context until that decision (DR-54), rather than 89 annotations for code that may go.
#nullable disable
using System.Collections.Generic;

namespace Gym.Merging {
    public interface IMergable {
        /// <summary>
        ///     Performs a merge with object of unknown type.
        /// </summary>
        /// <param name="right">object to take data from</param>
        /// <param name="preference">see the documentation on the enum values.</param>
        void Merge(object right, MergePrefer preference);

        IEnumerable<KeyValuePair<string, object>> Contents { get; }

        /// <summary>
        ///     Assign <paramref name="value"/> to given <paramref name="key"/>.
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        /// <exception cref="KeyNotFoundException">If <see cref="IsExpandable"/> is false and When the key is not present</exception>
        void Set(string key, object value);

        /// <summary>
        ///     Gets the value of <paramref name="key"/>.
        /// </summary>
        /// <param name="key"></param>
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
        ///     Is it possible to add values of keys that do not present in <see cref="Keys"/>?
        /// </summary>
        int Count { get; }
    }

    public interface IMergable<in T> : IMergable {
        /// <summary>
        ///     Performs a merge with object of <typeparamref name="{T}"/> type.
        /// </summary>
        /// <param name="right">object to take data from</param>
        /// <param name="preference">see the documentation on the enum values.</param>
        void Merge(T right, MergePrefer preference);
    }
}