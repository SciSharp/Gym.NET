using System;
using Gym.Collections;
using JetBrains.Annotations;
using NumSharp;

namespace Gym.Legacy {
    /// <summary>
    ///     The pre-0.26 transition, <c>(observation, reward, done, info)</c>, with a single <see cref="Done"/> flag.
    ///     <see cref="LegacyEnvAdapter"/> produces it from any <see cref="Gym.Envs.IEnv"/> through
    ///     <see cref="Gym.Utils.StepApiCompatibility"/>.
    /// </summary>
    /// <remarks>
    ///     Moved from <c>Gym.Observations</c> by the core-API work (P1); kept for one major version (port plan §16).
    ///     <see cref="Done"/> merges termination and truncation (<c>Information["TimeLimit.truncated"]</c> tells them apart);
    ///     the reward is float64, as in <see cref="Gym.Observations.StepResult{TObs}"/>.
    /// </remarks>
    [Obsolete("The 4-tuple step API is legacy: Env<TObs, TAct>.Step returns Gym.Observations.StepResult<TObs> (done is terminated || truncated). Gym.Legacy is removed in 3.0.")]
    public class Step : IEquatable<Step>, ICloneable {
        /// <summary>The observation after the action.</summary>
        public NDArray Observation { get; set; }

        /// <summary>The reward, float64 like every reward in Gym.NET.</summary>
        public double Reward { get; set; }

        /// <summary>Whether the episode ended, by termination or truncation (<c>terminated or truncated</c>).</summary>
        public bool Done { get; set; }

        /// <summary>The diagnostics; <c>"TimeLimit.truncated"</c> is set when the episode ended, true for a truncation.</summary>
        [CanBeNull] public Dict Information { get; set; }

        /// <summary>An empty transition, filled through the setters.</summary>
        public Step() { }

        /// <summary>A transition of the four values.</summary>
        /// <param name="observation">The observation after the action.</param>
        /// <param name="reward">The reward.</param>
        /// <param name="done">Whether the episode ended.</param>
        /// <param name="information">The diagnostics, or <see langword="null"/>.</param>
        public Step(NDArray observation, double reward, bool done, Dict information) {
            Observation = observation;
            Done = done;
            Information = information;
            Reward = reward;
        }

        #region Generated

        /// <summary>Deconstructs like Python's 4-tuple: <c>var (obs, reward, done, info) = step;</c>.</summary>
        /// <param name="observation">The observation.</param>
        /// <param name="reward">The reward.</param>
        /// <param name="done">Whether the episode ended.</param>
        /// <param name="information">The diagnostics.</param>
        public void Deconstruct(out NDArray observation, out double reward, out bool done, out Dict information) {
            observation = Observation;
            reward = Reward;
            done = Done;
            information = Information;
        }

        /// <summary>
        ///     Value equality over the four fields. Observations are equal when they have the same shape, the same dtype and
        ///     identical elements; they are never compared with <see cref="NDArray.Equals(object)"/>, which broadcasts (a 0-d
        ///     observation would equal every array holding its value) and throws for shapes that cannot broadcast.
        /// </summary>
        /// <param name="other">The step to compare with; may be <see langword="null"/>.</param>
        /// <returns><see langword="true"/> when observation, reward, done flag and information are all equal.</returns>
        public bool Equals(Step other) {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return ObservationEquals(Observation, other.Observation) && Reward.Equals(other.Reward) && Done == other.Done && Equals(Information, other.Information);
        }

        /// <summary>
        ///     Exact, non-broadcasting equality of two observations: the same shape, the same dtype, then
        ///     <see cref="np.array_equal(NDArray, NDArray)"/>. Two <see langword="null"/> observations are equal.
        /// </summary>
        /// <param name="left">The first observation; may be <see langword="null"/>.</param>
        /// <param name="right">The second observation; may be <see langword="null"/>.</param>
        /// <returns><see langword="true"/> when both are <see langword="null"/>, the same array, or equal element for element.</returns>
        /// <remarks>An observation containing NaN is equal only to itself (by reference), as NaN never equals NaN.</remarks>
        private static bool ObservationEquals(NDArray left, NDArray right) {
            if (ReferenceEquals(left, right)) return true;
            if (left is null || right is null) return false;
            return left.Shape == right.Shape && left.dtype == right.dtype && np.array_equal(left, right);
        }

        /// <summary>Determines whether the specified object is equal to the current object.</summary>
        /// <param name="obj">The object to compare with the current object. </param>
        /// <returns>
        /// <see langword="true" /> if the specified object  is equal to the current object; otherwise, <see langword="false" />.</returns>
        public override bool Equals(object obj) {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((Step) obj);
        }

        /// <summary>
        ///     Hashes the observation's shape (not its data), the reward, the done flag and the information. The data cannot
        ///     take part: <see cref="NDArray.GetHashCode"/> throws, because NumPy arrays are mutable and therefore unhashable.
        /// </summary>
        /// <returns>
        ///     A hash consistent with <see cref="Equals(Step)"/>: equal steps always have observations of the same shape. Steps
        ///     that differ only in observation values collide, which costs dictionary performance, never correctness.
        /// </returns>
        public override int GetHashCode() {
            unchecked {
                // `Observation != null` would be an element-wise comparison in NumSharp (an NDArray<bool>, not a bool).
                var hashCode = (Observation is not null ? Observation.Shape.GetHashCode() : 0);
                hashCode = (hashCode * 397) ^ Reward.GetHashCode();
                hashCode = (hashCode * 397) ^ Done.GetHashCode();
                hashCode = (hashCode * 397) ^ (Information != null ? Information.GetHashCode() : 0);
                return hashCode;
            }
        }

        /// <summary>Creates a new object that is a copy of the current instance.</summary>
        /// <returns>A new object that is a copy of this instance.</returns>
        public object Clone() {
            return new Step((NDArray) Observation.Clone(), Reward, Done, Information);
        }

        /// <summary>Returns a value that indicates whether the values of two <see cref="Step" /> objects are equal.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if the <paramref name="left" /> and <paramref name="right" /> parameters have the same value; otherwise, false.</returns>
        public static bool operator ==(Step left, Step right) {
            return Equals(left, right);
        }

        /// <summary>Returns a value that indicates whether two <see cref="Step" /> objects have different values.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if <paramref name="left" /> and <paramref name="right" /> are not equal; otherwise, false.</returns>
        public static bool operator !=(Step left, Step right) {
            return !Equals(left, right);
        }

        /// <summary>Returns a string that represents the current object.</summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString() {
            return $"{nameof(Reward)}: {Reward}, {nameof(Done)}: {Done}, {nameof(Information)}: {Information}, {nameof(Observation)}: {Observation}";
        }

        #endregion
    }
}