// Gym.Internal is 0.2's code, under review for deletion (R5 of docs/plans/CSHARP_CODE_REVIEW.md): it keeps no nullable
// context until that decision (DR-54), rather than 89 annotations for code that may go.
#nullable disable
namespace Gym.Merging {
    /// <summary>
    ///     See the value documentations.
    /// </summary>
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