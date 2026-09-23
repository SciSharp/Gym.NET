using System;
using NumSharp;

namespace Gym.Spaces {
    public class Discrete : Space {
        public int N { get; }
        public int Start { get; private set; }
        protected NumPyRandom RandomState;

        /// <summary>Initializes a new instance of the <see cref="T:System.Object" /> class.</summary>
        public Discrete(int n, Type dType = null, int seed = -1, int start=0, NumPyRandom random_state = null) : base(new Shape(n), (dType = dType ?? np.float32)) {
            N = n;
            Start = start;
            RandomState = seed != -1 ? np.random.RandomState(seed) : random_state ?? np.random;
        }

        /// <summary>
        ///     Draws a random action from <c>[Start, Start + N)</c>, optionally restricted by <paramref name="mask"/>.
        /// </summary>
        /// <param name="mask">
        ///     Optional validity mask where entries equal to 1 mark allowed actions; <see langword="null"/> samples
        ///     uniformly over all <see cref="N"/> actions.
        /// </param>
        /// <returns>
        ///     The sampled action offset by <see cref="Start"/>. When a mask allows no action, <see cref="Start"/>
        ///     is returned without drawing from the generator.
        /// </returns>
        /// <remarks>
        ///     Draws come from this space's generator, so seeding the space (<see cref="Seed"/>) makes the sequence
        ///     reproducible. With a mask, the action is drawn uniformly from the allowed indices, the same selection
        ///     as Gymnasium's <c>start + np_random.choice(np.where(mask)[0])</c>. The generator is still the legacy
        ///     <c>RandomState</c>, so draws aren't bit-identical to Gymnasium until the space is re-ported (plan §7).
        /// </remarks>
        public override NDArray Sample(NDArray mask = null) {
            // `is not null` rather than `!= null`: since NumSharp 0.70, NDArray overloads != element-wise and
            // returns an NDArray<bool>, so a null check through the operator no longer yields a bool.
            if (mask is not null)
            {
                NDArray bmask = (mask == 1); // Valid action mask with boolean selector
                if (np.any(bmask))
                {
                    // choice(NDArray) returns one uniformly drawn element of the allowed-index array. The former
                    // `(int)` cast sampled from [0, firstAllowedIndex) under NumSharp.Lite (plan defect B-04), and
                    // under NumSharp 0.70 it throws, because only 0-d arrays convert to scalars there.
                    return Start + RandomState.choice(np.nonzero(bmask)[0]);
                }
                return Start;
            }
            return Start + RandomState.randint(0, N, default);
        }

        public override bool Contains(object ndArray) {
            if (ndArray is int i) {
                return Contains(i);
            }

            throw new NotSupportedException(ndArray?.ToString() ?? nameof(ndArray));
        }

        public bool Contains(int x) {
            return x >= 0 && x < N;
        }

        public bool Contains(Enum x) {
            return Contains((int) (object) x);
        }

        public override void Seed(int seed) {
            RandomState = np.random.RandomState(seed);
        }

        #region Object Overrides

        /// <summary>Returns a string that represents the current object.</summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString() {
            return "Discrete" + Shape;
        }

        #endregion
    }
}