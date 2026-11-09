#nullable enable
using System;
using System.Linq;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Wrappers {
    /// <summary>
    ///     Applies a function to every action before the env steps with it, as Gymnasium's <c>TransformAction</c> does.
    ///     Give it the space of the actions it takes when they differ from the env's.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TWAct">The action type the wrapper takes.</typeparam>
    /// <typeparam name="TAct">The env's action type.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Without an action space the wrapper forwards the env's, which only works when <typeparamref name="TWAct"/> is the
    ///     env's action type: reading it otherwise raises <see cref="TypeError"/>. The base of Gymnasium's action wrappers,
    ///     which set <see cref="Func"/> and <see cref="SavedKwargs"/> in their own constructors.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_action.py#L26">gymnasium/wrappers/transform_action.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class TransformAction<TObs, TWAct, TAct> : ActionWrapper<TObs, TWAct, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env.</param>
        /// <param name="func">The transformation; <see langword="null"/> fails at the first step, as Python's <c>None</c> does when called.</param>
        /// <param name="actionSpace">The wrapper's action space, owned by the wrapper from now on; <see langword="null"/> forwards the env's.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public TransformAction(Env<TObs, TAct> env, Func<TWAct, TAct> func, Space<TWAct>? actionSpace) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["func"] = func, ["action_space"] = actionSpace });
            if (actionSpace is not null)
                ActionSpace = actionSpace;
            Func = func;
        }

        /// <summary>
        ///     Wraps <paramref name="env"/> for a subclass that computes its function and space in its own constructor and
        ///     assigns <see cref="Func"/>, <see cref="Env{TObs, TAct}.ActionSpace"/> and <see cref="SavedKwargs"/> there.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        protected TransformAction(Env<TObs, TAct> env) : base(env) {
            SavedKwargs = new Info();
            Func = null!;
        }

        /// <summary>The recorded constructor arguments (Python's <c>_saved_kwargs</c>): a subclass's own, as Python records the first call's.</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; protected set; }

        /// <summary>The transformation (<c>func</c>).</summary>
        [PyName("func")]
        public Func<TWAct, TAct> Func { get; protected set; }

        /// <summary>Applies <see cref="Func"/> (<c>action(action)</c>).</summary>
        /// <param name="action">The wrapper's action.</param>
        /// <returns>The env's action.</returns>
        /// <exception cref="TypeError">Thrown when the function is <see langword="null"/>: <c>'NoneType' object is not callable</c>.</exception>
        public override TAct Action(TWAct action) =>
            Func is null ? throw new TypeError("'NoneType' object is not callable") : Func(action);
    }

    /// <summary>
    ///     Clips every action into the env's <c>Box</c> bounds before the env sees it, as Gymnasium's <c>ClipAction</c> does;
    ///     the wrapper's own action space is unbounded (<c>Box(-inf, inf)</c> of the same shape and dtype), since any action
    ///     is accepted.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The env's action type: arrays (<see cref="NDArray"/>, or <see langword="object"/> holding them).</typeparam>
    /// <remarks>
    ///     <para>
    ///     The clip is NumPy's <c>np.clip(action, low, high)</c> with the env's bounds, so its dtype follows NumPy's promotion
    ///     of the action with the bounds. Each step's action is a new array.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_action.py#L76">gymnasium/wrappers/transform_action.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class ClipAction<TObs, TAct> : TransformAction<TObs, NDArray, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its action space must be a <c>Box</c>.</param>
        /// <exception cref="TypeError">Thrown when the action space is no <c>Box</c> (Gymnasium's message), or <paramref name="env"/> is <see langword="null"/>.</exception>
        public ClipAction(Env<TObs, TAct> env) : base(env) {
            if (ObjectSpace.Unview(env.ActionSpace) is not Box box)
                throw new TypeError($"ClipAction requires a Box action space, got {PyBuiltins.TypeRepr(ObjectSpace.Unview(env.ActionSpace))}");
            SavedKwargs = RecordConstructorArgs.Record(new Info());
            ActionSpace = new Box(double.NegativeInfinity, double.PositiveInfinity, box.Shape, box.DType);
            Func = action => (TAct)(object)np.clip(action, box.Low, box.High);
        }
    }

    /// <summary>
    ///     Takes actions in <c>[min_action, max_action]</c> and rescales them affinely onto the env's <c>Box</c> bounds, as
    ///     Gymnasium's <c>RescaleAction</c> does (the inverse of <see cref="WrapperFunctions.RescaleBox"/>'s map).
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The env's action type: arrays (<see cref="NDArray"/>, or <see langword="object"/> holding them).</typeparam>
    /// <remarks>
    ///     <para>
    ///     Each step's action is <c>(action - intercept) / gradient</c>, a new array in NumPy's promotion of the action with
    ///     the env's dtype. A component whose new bounds coincide has no inverse and is refused at construction.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_action.py#L126">gymnasium/wrappers/transform_action.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class RescaleAction<TObs, TAct> : TransformAction<TObs, NDArray, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its action space must be a <c>Box</c>.</param>
        /// <param name="minAction">The wrapper's lower bound: an array of the space's shape, or a number (<see cref="WrapperFunctions.RescaleBox"/>).</param>
        /// <param name="maxAction">The wrapper's upper bound, in the same forms.</param>
        /// <exception cref="TypeError">Thrown when the action space is no <c>Box</c> or a bound has the wrong type (Gymnasium's messages), or <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="NumSharp.ValueError">Thrown when the bounds do not fit the space (see <see cref="WrapperFunctions.RescaleBox"/>).</exception>
        /// <exception cref="InvalidBound">Thrown when a component's new bounds are equal, with Gymnasium's message.</exception>
        public RescaleAction(Env<TObs, TAct> env, object minAction, object maxAction) : base(env) {
            if (ObjectSpace.Unview(env.ActionSpace) is not Box box)
                throw new TypeError($"RescaleAction requires a Box action space, got {PyBuiltins.TypeRepr(ObjectSpace.Unview(env.ActionSpace))}");
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["min_action"] = minAction, ["max_action"] = maxAction });

            (Box rescaled, _, Func<NDArray, NDArray> backward) = WrapperFunctions.RescaleBox(box, minAction, maxAction);
            // The wrapper applies the inverse of the rescaling, and a component that is rescaled onto a single point has no
            // inverse.
            using (NDArray collapsed = rescaled.Low == rescaled.High) {
                if (np.any(collapsed)) {
                    rescaled.Dispose();
                    throw new InvalidBound($"Min action ({PyBuiltins.Str(minAction)}) must be strictly smaller than max action ({PyBuiltins.Str(maxAction)}), the rescaling has no inverse where they are equal");
                }
            }

            ActionSpace = rescaled;
            Func = action => (TAct)(object)backward(action);
        }
    }

    /// <summary>
    ///     Takes discrete actions and steps the env with the center of the matching bin of its finite <c>Box</c>, as
    ///     Gymnasium's <c>DiscretizeAction</c> does: one <c>Discrete</c> number (row-major over the box's elements) or a
    ///     <c>MultiDiscrete</c> vector of one index per element.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The env's action type: arrays (<see cref="NDArray"/>, or <see langword="object"/> holding them).</typeparam>
    /// <remarks>
    ///     <para>
    ///     The centers are <c>0.5 * (edges[:-1] + edges[1:])</c> of NumPy's <c>np.linspace(low[i], high[i], bins[i] + 1)</c>,
    ///     in the box's dtype; an index beyond its element's bins is clamped to the nearest bin.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_action.py#L187">gymnasium/wrappers/transform_action.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class DiscretizeAction<TObs, TAct> : TransformAction<TObs, NDArray, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its action space must be a finite <c>Box</c>.</param>
        /// <param name="bins">The bins per element: one number for every element, or a sequence of one per element.</param>
        /// <param name="multidiscrete">Whether actions are <c>MultiDiscrete</c> vectors instead of one <c>Discrete</c> number.</param>
        /// <exception cref="TypeError">Thrown when the action space is no <c>Box</c> (Gymnasium's message), or <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="NumSharp.ValueError">Thrown when a bound is infinite, or the bins do not match the elements, with Gymnasium's messages.</exception>
        public DiscretizeAction(Env<TObs, TAct> env, object bins, bool multidiscrete = false) : base(env) {
            if (ObjectSpace.Unview(env.ActionSpace) is not Box box)
                throw new TypeError("DiscretizeAction is only compatible with Box continuous actions.");

            Low = np.ravel(box.Low);
            High = np.ravel(box.High);
            NDims = (int)Low.size;
            ObservationWrapperFunctions.RequireFinite(Low, High, "action", () => {
                Low.Dispose();
                High.Dispose();
            });

            Multidiscrete = multidiscrete;
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["bins"] = bins, ["multidiscrete"] = multidiscrete });
            Bins = ObservationWrapperFunctions.BinsPerDimension(bins, NDims, "action");

            BinCenters = new NDArray[NDims];
            for (int i = 0; i < NDims; i++) {
                using NDArray start = Low[i];
                using NDArray stop = High[i];
                using NDArray edges = WrapperFunctions.Linspace(start, stop, Bins[i] + 1);
                using NDArray lower = edges[":-1"];
                using NDArray upper = edges["1:"];
                using NDArray sum = lower + upper;
                // 0.5 is a weak Python float, so the centers keep the edges' dtype.
                BinCenters[i] = 0.5 * sum;
            }

            if (multidiscrete)
                ActionSpace = new MultiDiscrete(Bins);
            else
                ActionSpace = new Discrete(Bins.Aggregate(1L, (product, b) => product * b));

            _envShape = box.Shape;
            _envDType = box.DType!;
            Func = action => (TAct)(object)Continuous(action);
        }

        /// <summary>The env's action shape, which a continuous action is reshaped to.</summary>
        private readonly Shape _envShape;

        /// <summary>The env's action dtype, which a continuous action is built in.</summary>
        private readonly DType _envDType;

        /// <summary>The box's lower bounds, flattened (<c>low</c>).</summary>
        public NDArray Low { get; }

        /// <summary>The box's upper bounds, flattened (<c>high</c>).</summary>
        public NDArray High { get; }

        /// <summary>The number of elements discretized (<c>n_dims</c>).</summary>
        [PyName("n_dims")]
        public int NDims { get; }

        /// <summary>Whether actions are <c>MultiDiscrete</c> (<c>multidiscrete</c>).</summary>
        public bool Multidiscrete { get; }

        /// <summary>The bins of each element (<c>bins</c>).</summary>
        public long[] Bins { get; }

        /// <summary>The bin centers of each element (<c>bin_centers</c>), in the box's dtype.</summary>
        public NDArray[] BinCenters { get; }

        /// <summary>
        ///     The discrete action whose bin centers are nearest a continuous action (<c>revert_action</c>): per element, the
        ///     center at the smallest distance (<c>np.argmin(np.abs(centers - action[i]))</c>).
        /// </summary>
        /// <param name="action">A continuous action of the env's shape.</param>
        /// <returns>A new int64 array: <c>(n_dims,)</c> for <c>MultiDiscrete</c>, 0-d (the flat index) for <c>Discrete</c>.</returns>
        public NDArray RevertAction(NDArray action) {
            using NDArray flat = np.ravel(action);
            var indices = new long[NDims];
            for (int i = 0; i < NDims; i++) {
                using NDArray element = flat[i];
                using NDArray difference = BinCenters[i] - element;
                using NDArray distance = np.abs(difference);
                using NDArray nearest = np.argmin(distance);
                indices[i] = Convert.ToInt64(nearest.GetAtIndex(0));
            }

            if (Multidiscrete)
                return np.array(indices);
            // np.ravel_multi_index(indices, self.bins): row-major, the last element varying fastest.
            long flatIndex = 0;
            for (int i = 0; i < NDims; i++)
                flatIndex = flatIndex * Bins[i] + indices[i];
            return NDArray.Scalar(flatIndex);
        }

        /// <summary>The continuous action of a discrete one (<c>action(act)</c>): each element's bin center, clamped into its bins.</summary>
        /// <param name="action">A <c>Discrete</c> number (a 0-d array) or a <c>MultiDiscrete</c> vector.</param>
        /// <returns>A new array of the env's shape and dtype.</returns>
        private NDArray Continuous(NDArray action) {
            long[] indices = Multidiscrete ? ObservationWrapperFunctions.Longs(action) : UnflattenIndex(ObservationWrapperFunctions.Long(action));
            var centers = new NDArray[indices.Length];
            try {
                for (int i = 0; i < indices.Length; i++)
                    centers[i] = BinCenters[i][Math.Min(Math.Max(indices[i], 0), Bins[i] - 1)].copy();
                using NDArray stacked = np.stack(centers);
                using NDArray typed = stacked.astype(_envDType);
                return np.reshape(typed, _envShape).copy();
            } finally {
                foreach (NDArray? center in centers)
                    center?.Dispose();
            }
        }

        /// <summary>The per-element indices of a flat index (<c>_unflatten_index</c>), last element varying fastest.</summary>
        /// <param name="flatIndex">The flat index.</param>
        /// <returns>One index per element.</returns>
        private long[] UnflattenIndex(long flatIndex) {
            var indices = new long[NDims];
            for (int i = NDims - 1; i >= 0; i--) {
                // Python's % and // (floor), as np.int64 arithmetic gives them.
                indices[i] = PyMath.Mod(flatIndex, Bins[i]);
                flatIndex = PyMath.FloorDiv(flatIndex, Bins[i]);
            }

            return indices;
        }

        /// <summary>Releases the bounds and centers, then disposes the wrapper (and the env beneath) as every wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            if (disposing) {
                // Low and High are views of the env's space bounds: released before the env disposes its space.
                Low.Dispose();
                High.Dispose();
                foreach (NDArray center in BinCenters)
                    center?.Dispose();
            }

            base.Dispose(disposing);
        }
    }
}
