using System;
using NumSharp;

namespace Gym.Spaces {
    public abstract class Space {
        public Shape Shape { get; protected set; }
        public Type DType { get; protected set; }

        /// <summary>Initializes a new instance of the <see cref="T:System.Object" /> class.</summary>
        protected Space(Shape shape, Type dType) {
            Shape = shape;
            DType = dType;
        }

        public abstract NDArray Sample(NDArray mask = null);
        public abstract bool Contains(object ndArray);
        public abstract void Seed(int seed);
    }
}