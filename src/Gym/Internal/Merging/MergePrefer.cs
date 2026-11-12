// Gym.Internal is 0.2's code, [Obsolete] since 1.0 and removed in 2.0 with Gym.Legacy (DR-56): it keeps no nullable
// context (DR-54), rather than 89 annotations for code on its way out.
#nullable disable
using System;

namespace Gym.Merging {
    /// <summary>
    ///     Which value wins when the legacy deep merge (<see cref="Merger"/>) meets the same key on both sides and cannot
    ///     merge the two values (neither is a dictionary or a growable list).
    /// </summary>
    /// <remarks>
    ///     Obsolete with the legacy <see cref="Gym.Collections.Dict"/> (DR-56) and removed in 2.0. Its replacement,
    ///     <see cref="Gym.Envs.Info.Update"/>, always lets the new value win (<see cref="New"/>), one level deep, as Python's
    ///     <c>dict.update</c> does. Footgun: two merge paths ignore it, see <see cref="Merger"/>.
    /// </remarks>
    [Obsolete("The legacy Dict's deep merge: use Gym.Envs.Info.Update (Python's dict.update: the new value wins, one level deep). Removed in 2.0 with Gym.Legacy.")]
    public enum MergePrefer {
        /// <summary>
        ///     Prefer the data in current dictionary over the data in the new dictionary merge is happening with.
        /// </summary>
        Old,

        /// <summary>
        ///     Prefer the data in new dictionary merge is happening with over the data in the current dictionary.
        /// </summary>
        New
    }
}
