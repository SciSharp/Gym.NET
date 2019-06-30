using System;
using Gym.Collections;
using JetBrains.Annotations;
using NumSharp;

namespace Gym.Observations {
    public class Step : IEquatable<Step>, ICloneable {
        public NDArray Observation { get; set; }
        public float Reward { get; set; }
        public bool Done { get; set; }
        [CanBeNull] public Dict Information { get; set; }

        public Step() { }

        public Step(NDArray observation, float reward, bool done, Dict information) {
            Observation = observation;
            Done = done;
            Information = information;
            Reward = reward;
        }

        #region Generated

        public void Deconstruct(out NDArray observation, out float reward, out bool done, out Dict information) {
            observation = Observation;
            reward = Reward;
            done = Done;
            information = Information;
        }

        /// <summary>Indicates whether the current object is equal to another object of the same type.</summary>
        /// <param name="other">An object to compare with this object.</param>
        /// <returns>
        /// <see langword="true" /> if the current object is equal to the <paramref name="other" /> parameter; otherwise, <see langword="false" />.</returns>
        public bool Equals(Step other) {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return Equals(Observation, other.Observation) && Reward.Equals(other.Reward) && Done == other.Done && Equals(Information, other.Information);
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

        /// <summary>Serves as the default hash function. </summary>
        /// <returns>A hash code for the current object.</returns>
        public override int GetHashCode() {
            unchecked {
                var hashCode = (Observation != null ? Observation.GetHashCode() : 0);
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

        /// <summary>Returns a value that indicates whether the values of two <see cref="T:Gym.Observations.Step" /> objects are equal.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if the <paramref name="left" /> and <paramref name="right" /> parameters have the same value; otherwise, false.</returns>
        public static bool operator ==(Step left, Step right) {
            return Equals(left, right);
        }

        /// <summary>Returns a value that indicates whether two <see cref="T:Gym.Observations.Step" /> objects have different values.</summary>
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