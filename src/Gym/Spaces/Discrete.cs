using System;
using NumSharp;

namespace Gym.Spaces {
    public class Discrete : Space {
        public int N { get; }
        public int Start { get; private set; }
        protected NumPyRandom RandomState;

        /// <summary>
        ///     The space of the <paramref name="n"/> integers <c>{start, ..., start + n - 1}</c>.
        /// </summary>
        /// <param name="n">The number of elements.</param>
        /// <param name="dType">The declared element type; <see langword="null"/> means float32. <see cref="Sample"/> returns int64 regardless.</param>
        /// <param name="seed">A seed for this space's own generator, or -1 to use <paramref name="random_state"/> instead.</param>
        /// <param name="start">The smallest element.</param>
        /// <param name="random_state">
        ///     The generator to draw from when <paramref name="seed"/> is -1, or <see langword="null"/> for the process-wide
        ///     <c>np.random</c>.
        /// </param>
        /// <remarks>
        ///     The shape stays <c>(n)</c> instead of Gymnasium's <c>()</c> (B-03, P2): the examples read
        ///     <c>ActionSpace.Shape.Size</c> as the number of actions.
        /// </remarks>
        public Discrete(int n, DType dType = null, int seed = -1, int start=0, NumPyRandom random_state = null) : base(new Shape(n), (dType = dType ?? np.float32)) {
            N = n;
            Start = start;
            RandomState = seed != -1 ? np.random.RandomState(seed) : random_state ?? np.random;
        }

        /// <summary>
        ///     Draws one element uniformly, or uniformly among the elements a mask allows. The result is a fresh 0-d int64
        ///     array (Gymnasium's <c>np.int64</c>) that the caller owns.
        /// </summary>
        /// <param name="mask">
        ///     <see langword="null"/> to draw from all <see cref="N"/> elements; otherwise a 1-D array of length <see cref="N"/>
        ///     in which <c>1</c> marks an allowed element. It is not validated yet (Gymnasium asserts int8, shape and 0/1 values).
        /// </param>
        /// <returns>
        ///     A 0-d int64 array: <see cref="Start"/> plus the drawn offset, or <see cref="Start"/> itself when the mask allows
        ///     nothing, as in Gymnasium.
        /// </returns>
        /// <remarks>
        ///     Mirrors the mask and uniform branches of <c>gymnasium/spaces/discrete.py:127</c> @ 9e04324, drawing with the
        ///     legacy generator (<c>randint</c>/<c>choice</c> instead of Gymnasium's <c>Generator.integers</c>/<c>choice</c>)
        ///     until F2; both draws are bit-identical to NumPy's <c>RandomState</c>. Not ported yet (P2): the probability mask,
        ///     the mask assertions, and sampling in the space's own dtype (Gymnasium's default is int64 as well).
        /// </remarks>
        public override NDArray Sample(NDArray mask = null) {
            // `mask != null` is element-wise in NumSharp (an NDArray<bool>, not a bool), which made every call throw (B-01).
            if (mask is not null)
            {
                using NDArray bmask = (mask == 1); // Valid action mask with boolean selector
                if (np.any(bmask))
                {
                    // Gymnasium draws uniformly among the valid actions: np_random.choice(np.where(mask == 1)[0]). The former
                    // (int) cast collapsed the whole index array into one int and sampled [0, firstValidIndex) (B-04). A
                    // Discrete mask is 1-D, so nonzero returns exactly one index array.
                    using NDArray validActions = np.nonzero(bmask)[0];
                    using NDArray chosen = RandomState.choice(validActions);
                    return NDArray.Scalar<long>(Start + (long)chosen);
                }
                return NDArray.Scalar<long>(Start);
            }
            // NumSharp's randint defaults to int32 where NumPy 2 defaults to int64. The values and the stream are the same
            // for either dtype at this range; int64 makes the result NumPy's dtype.
            using NDArray drawn = RandomState.randint(0, N, dtype: np.int64);
            return NDArray.Scalar<long>(Start + (long)drawn);
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