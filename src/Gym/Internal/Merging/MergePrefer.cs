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