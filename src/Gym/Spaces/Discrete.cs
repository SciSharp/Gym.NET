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

        public override NDArray Sample(NDArray mask = null) {
            if (mask != null)
            {
                NDArray bmask = (mask == 1); // Valid action mask with boolean selector
                if (np.any(bmask))
                {
                    return Start + RandomState.choice((int)np.nonzero(bmask)[0]);
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