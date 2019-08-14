using System;
using NumSharp;

namespace Gym.Spaces {
    public class Discrete : Space {
        public int N { get; }
        protected NumPyRandom RandomState;

        /// <summary>Initializes a new instance of the <see cref="T:System.Object" /> class.</summary>
        public Discrete(int n, Type dType = null, int seed = -1) : base(new Shape(n), (dType = dType ?? np.float32)) {
            N = n;

            RandomState = seed != -1 ? np.random.RandomState(seed) : np.random;
        }

        public override NDArray Sample() {
            return RandomState.randint(0, N, default);
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