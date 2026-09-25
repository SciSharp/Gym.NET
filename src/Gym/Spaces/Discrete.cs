using System;
using Gym.Exceptions;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     The integers <c>{Start, ..., Start + N - 1}</c>: a categorical action or observation, such as CartPole's two
    ///     pushes or LunarLander's four engine choices.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/spaces/discrete.py#L27">gymnasium/spaces/discrete.py</see>.</remarks>
    public class Discrete : Space {
        /// <summary>The number of elements, <see cref="long"/> like Gymnasium's <c>np.int64</c> <c>n</c>.</summary>
        public long N { get; }

        /// <summary>The smallest element, <see cref="long"/> like Gymnasium's <c>np.int64</c> <c>start</c>.</summary>
        public long Start { get; private set; }

        /// <summary>
        ///     The generator this space samples from: NumPy's <c>Generator(PCG64)</c>, Gymnasium's <c>np_random</c>. The space
        ///     owns it, unless one was passed in to be shared (LunarLander shares its own with its action space).
        /// </summary>
        /// <remarks>Named <c>RandomState</c> and typed as NumPy's legacy <c>RandomState</c> (MT19937) before Gym.NET adopted Gymnasium's seeding.</remarks>
        protected Generator NpRandom;

        /// <summary>
        ///     The space of the <paramref name="n"/> integers <c>{start, ..., start + n - 1}</c>.
        /// </summary>
        /// <param name="n">The number of elements.</param>
        /// <param name="dType">
        ///     The declared element type; <see langword="null"/> means int64, Gymnasium's default. <see cref="Sample"/> returns
        ///     int64 regardless.
        /// </param>
        /// <param name="seed">
        ///     A non-negative seed for this space's own generator, drawing what Gymnasium's <c>Discrete(n, seed=seed)</c> draws;
        ///     or -1 to use <paramref name="random_state"/> instead.
        /// </param>
        /// <param name="start">The smallest element.</param>
        /// <param name="random_state">
        ///     The generator to draw from when <paramref name="seed"/> is -1 (it may be shared, for example with the env), or
        ///     <see langword="null"/> for a generator of the space's own, seeded from OS entropy.
        /// </param>
        /// <exception cref="Error">Thrown when <paramref name="seed"/> is negative but not -1, with Gymnasium's message.</exception>
        /// <remarks>
        ///     The shape stays <c>(n)</c> instead of Gymnasium's <c>()</c> (B-03, P2): the examples read
        ///     <c>ActionSpace.Shape.Size</c> as the number of actions. The default dtype was float32 under NumSharp.Lite.
        /// </remarks>
        public Discrete(long n, DType dType = null, long seed = -1, long start=0, Generator random_state = null) : base(new Shape(n), (dType = dType ?? np.int64)) {
            N = n;
            Start = start;
            // A generator of the space's own unless one is shared on purpose: a shared generator couples two streams, and
            // generators are not synchronized, so two threads drawing from one corrupt its state.
            NpRandom = seed != -1 ? Seeding.NpRandom(seed).Rng : random_state ?? Seeding.NpRandom().Rng;
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
        ///     The mask and uniform branches make Gymnasium's generator calls (<c>choice(np.where(mask == 1)[0])</c> and
        ///     <c>integers(n, dtype=int64)</c>), so a space seeded like Gymnasium's samples the same values. Not supported yet
        ///     (P2): the probability mask, the mask assertions, and sampling in a dtype other than int64 (Gymnasium draws in
        ///     the space's dtype, int64 by default). Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/spaces/discrete.py#L127">gymnasium/spaces/discrete.py</see>.
        /// </remarks>
        public override NDArray Sample(NDArray mask = null) {
            // `mask != null` is element-wise in NumSharp (an NDArray<bool>, not a bool), which made every call throw (B-01).
            if (mask is not null)
            {
                using NDArray bmask = (mask == 1); // Valid action mask with boolean selector
                if (np.any(bmask))
                {
                    // Gymnasium draws uniformly among the valid actions: np_random.choice(np.where(mask == 1)[0]). A Discrete
                    // mask is 1-D, so nonzero returns exactly one index array.
                    using NDArray validActions = np.nonzero(bmask)[0];
                    using NDArray chosen = NpRandom.choice(validActions);
                    return NDArray.Scalar<long>(Start + SingleValue(chosen));
                }
                return NDArray.Scalar<long>(Start);
            }
            // Gymnasium: self.start + self.np_random.integers(self.n, dtype=self.dtype.type), i.e. [0, n).
            using NDArray drawn = NpRandom.integers(N, dtype: np.int64);
            return NDArray.Scalar<long>(Start + SingleValue(drawn));
        }

        /// <summary>
        ///     The one integer a size-less draw returns. NumSharp's <c>Generator.choice(array)</c> returns it as a one-element
        ///     1-D array where NumPy returns a scalar, while <c>Generator.integers</c> returns a 0-d array; both hold the same
        ///     value NumPy draws.
        /// </summary>
        /// <param name="draw">A 0-d or one-element array; the caller keeps ownership.</param>
        /// <returns>The drawn value.</returns>
        /// <exception cref="InvalidOperationException">Thrown when <paramref name="draw"/> does not hold exactly one element.</exception>
        private static long SingleValue(NDArray draw) {
            if (draw.size != 1)
                throw new InvalidOperationException($"Expected a single drawn value, got {draw.size}.");
            return Convert.ToInt64(draw.GetAtIndex(0));
        }

        /// <summary>
        ///     Whether <paramref name="ndArray"/> denotes an element of this space: any integer (<see cref="TryToInt64"/>'s
        ///     inputs) in <c>[Start, Start + N)</c>. Anything that is not an integer returns <see langword="false"/>, as in
        ///     Gymnasium; under NumSharp.Lite it threw <see cref="NotSupportedException"/> for everything but a boxed
        ///     <see cref="int"/>.
        /// </summary>
        /// <param name="ndArray">The candidate: a boxed integer of any width, a <see cref="bool"/>, an enum member, or a 0-d integer array.</param>
        /// <returns><see langword="true"/> when the candidate converts to an integer inside the space.</returns>
        /// <remarks>
        ///     Accepts what Gymnasium accepts, except that Gymnasium also requires <c>np.can_cast(x.dtype, self.dtype)</c>,
        ///     which rejects, for example, a uint64 value in an int64 space (P2). Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/spaces/discrete.py#L198">gymnasium/spaces/discrete.py</see>.
        /// </remarks>
        public override bool Contains(object ndArray) {
            try {
                return TryToInt64(ndArray, out long x) && Contains(x);
            } catch (OverflowException) {
                // Only an unsigned 64-bit value above long.MaxValue gets here. It cannot be an element of an int64 space;
                // Gymnasium returns False for an integer outside the dtype's range as well.
                return false;
            }
        }

        /// <summary>Whether <paramref name="x"/> lies in <c>[Start, Start + N)</c>.</summary>
        /// <param name="x">The candidate element.</param>
        /// <returns><see langword="true"/> when <paramref name="x"/> is an element of this space.</returns>
        /// <remarks>
        ///     Honours <see cref="Start"/> (B-02: <c>Discrete(3, start: 5)</c> accepted 0 and rejected 5 and 7). The upper bound
        ///     is computed in <see cref="Int128"/>, because Gymnasium compares unbounded Python ints and
        ///     <c>Start + N</c> can exceed <see cref="long.MaxValue"/>.
        /// </remarks>
        public bool Contains(long x) {
            return x >= Start && (Int128) x < (Int128) Start + N;
        }

        /// <summary>Whether the value of the enum member <paramref name="x"/> is an element of this space.</summary>
        /// <param name="x">The candidate, for example a <c>LunarLanderDiscreteActions</c> member.</param>
        /// <returns><see langword="true"/> when the member's underlying value lies in <c>[Start, Start + N)</c>.</returns>
        /// <remarks>
        ///     Works for every underlying type. The former <c>(int)(object)x</c> unboxed exactly and threw
        ///     <see cref="InvalidCastException"/> for enums not backed by <see cref="int"/>.
        /// </remarks>
        public bool Contains(Enum x) {
            return Contains((object) x);
        }

        /// <summary>
        ///     Converts an action handed to <c>Env.Step(object)</c> into the discrete value it denotes. Envs must call this
        ///     instead of casting: <c>(int)action</c> unboxes exactly and throws <see cref="InvalidCastException"/> as soon as
        ///     the action is a <see langword="long"/>, or the 0-d int64 <see cref="NDArray"/> that <see cref="Sample"/> returns.
        /// </summary>
        /// <param name="action">A boxed integer of any width, a <see cref="bool"/>, an enum member, or a 0-d integer <see cref="NDArray"/>.</param>
        /// <returns>The action as <see langword="long"/>, the .NET spelling of Gymnasium's <c>np.int64</c>.</returns>
        /// <exception cref="ArgumentException">
        ///     Thrown when <paramref name="action"/> is <see langword="null"/>, not an integer, or an array that is not 0-d.
        /// </exception>
        /// <exception cref="OverflowException">
        ///     Thrown when an unsigned 64-bit action exceeds <see cref="long.MaxValue"/> (see <see cref="TryToInt64"/>).
        /// </exception>
        public static long ToInt64(object action)
        {
            if (TryToInt64(action, out long value))
                return value;
            throw new ArgumentException(
                $"{action ?? "null"} ({action?.GetType().Name ?? "null"}) is not a discrete action: expected an integer, an enum or a 0-d integer NDArray.",
                nameof(action));
        }

        /// <summary>
        ///     Non-throwing form of <see cref="ToInt64"/> for membership tests, so that <see cref="Contains(object)"/> decides
        ///     what an input that is not an integer means instead of surfacing a conversion error.
        /// </summary>
        /// <param name="action">The candidate action.</param>
        /// <param name="value">The converted value when the method returns <see langword="true"/>; otherwise 0.</param>
        /// <returns><see langword="true"/> when <paramref name="action"/> denotes an integer.</returns>
        /// <exception cref="OverflowException">
        ///     Thrown when an unsigned 64-bit action (a <see langword="ulong"/>, a 0-d uint64 array or an enum backed by
        ///     <see langword="ulong"/>) exceeds <see cref="long.MaxValue"/>.
        /// </exception>
        /// <remarks>
        ///     Accepts what Gymnasium's <c>Discrete.contains</c> accepts: Python ints, which include <c>bool</c> (a subclass
        ///     of <c>int</c>, so <c>True</c> is 1), and 0-d arrays of an integer dtype (NumPy scalars). A 0-d <em>boolean</em>
        ///     array is rejected, because <c>np.bool_</c> is not an <c>np.integer</c>.
        /// </remarks>
        public static bool TryToInt64(object action, out long value)
        {
            switch (action)
            {
                case long l: value = l; return true;
                case int i: value = i; return true;
                case short s: value = s; return true;
                case sbyte sb: value = sb; return true;
                case byte b: value = b; return true;
                case ushort us: value = us; return true;
                case uint ui: value = ui; return true;
                case ulong ul: value = checked((long)ul); return true;
                case bool flag: value = flag ? 1 : 0; return true;
                case Enum e: value = Convert.ToInt64(e); return true;
                // Gymnasium's Discrete.contains accepts 0-d integer arrays (np.int64 scalars) as well as Python ints.
                case NDArray nd when nd.ndim == 0 && nd.dtype.kind is 'i' or 'u':
                    // NumSharp's (long) is a C-style cast: a uint64 above long.MaxValue wraps silently (ulong.MaxValue
                    // becomes -1, measured), so that dtype takes a checked conversion.
                    value = nd.typecode == NPTypeCode.UInt64 ? checked((long)(ulong)nd) : (long)nd;
                    return true;
                default:
                    value = 0;
                    return false;
            }
        }

        /// <summary>
        ///     Gives this space a fresh generator, <c>Generator(PCG64(SeedSequence(seed)))</c>, so it samples what Gymnasium's
        ///     space samples after <c>space.seed(seed)</c>. A generator the space shared (for example with its env) is left
        ///     untouched, so reseeding the space never moves the env's stream.
        /// </summary>
        /// <param name="seed">Any non-negative seed.</param>
        /// <exception cref="Error">Thrown when <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        /// <remarks>
        ///     Gymnasium's <c>seed</c> also accepts no seed and returns the one it used; that shape of the API arrives with P2.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/spaces/space.py#L112">gymnasium/spaces/space.py</see>.
        /// </remarks>
        public override void Seed(System.Numerics.BigInteger seed) {
            NpRandom = Seeding.NpRandom(seed).Rng;
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