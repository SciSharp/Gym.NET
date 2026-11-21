#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using Gym.Vector;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Wrappers {
    public static partial class WrapperFunctions {
        /// <summary>
        ///     A valid element of <paramref name="space"/> as close to zero as the space allows (Gymnasium's
        ///     <c>create_zero_array</c>): unlike <see cref="VectorUtils.CreateEmptyArray"/>, a <c>Box</c> that excludes zero gets
        ///     its nearest bound, so the result is always a member of the space. What the observation wrappers fill with
        ///     before an episode has data (<c>DelayObservation</c>, <c>FrameStackObservation</c>'s padding).
        /// </summary>
        /// <param name="space">The space; an <see cref="ObjectSpace"/> view is looked through.</param>
        /// <returns>
        ///     A new element the caller owns: an array for <c>Box</c>, <c>MultiDiscrete</c> and <c>MultiBinary</c>, a 0-d array
        ///     of the space's dtype holding <c>start</c> for <c>Discrete</c>, an <see langword="object"/>?[] (Python's tuple)
        ///     for <c>Tuple</c>, an <see cref="Info"/> for <c>Dict</c>, the first character repeated <c>min_length</c> times for
        ///     <c>Text</c>, a <see cref="GraphInstance"/> with one zero node (and one zero edge from node 0 to node 0) for
        ///     <c>Graph</c>, the stacked feature space's zero or an empty tuple for <c>Sequence</c>, and <c>(0, zero of the
        ///     first subspace)</c> for <c>OneOf</c>.
        /// </returns>
        /// <exception cref="TypeError">Thrown when <paramref name="space"/> is <see langword="null"/>, with Gymnasium's message.</exception>
        /// <exception cref="CustomSpaceError">Thrown for a space class Gymnasium registers no zero for, with Gymnasium's message.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/utils.py#L75">gymnasium/wrappers/utils.py</see>.</remarks>
        public static object? CreateZeroArray(Space space) {
            switch (ObjectSpace.Unview(space)) {
                case null:
                    throw new TypeError("The space provided to `create_zero_array` is not a gymnasium Space instance, type: <class 'NoneType'>, None");
                case Box box: {
                    // zeros, then low where low > 0 and high where high < 0: the bound nearest zero when zero is outside.
                    using NDArray zeros = np.zeros(box.Shape, box.DType!);
                    using NDArray lowPositive = box.Low > 0;
                    using NDArray raised = np.where(lowPositive, box.Low, zeros);
                    using NDArray highNegative = box.High < 0;
                    return np.where(highNegative, box.High, raised);
                }
                case Discrete discrete:
                    // Gymnasium's start is a scalar of the space's dtype (self.dtype.type(start)).
                    return NDArray.Scalar(discrete.Start).astype(discrete.DType);
                case MultiDiscrete multiDiscrete:
                    return multiDiscrete.Start.astype(multiDiscrete.DType!, copy: true);
                case MultiBinary multiBinary:
                    return np.zeros(multiBinary.Shape, multiBinary.DType!);
                case Tuple tuple:
                    return tuple.Spaces.Select(CreateZeroArray).ToArray();
                case Dict dict: {
                    var zero = new Info();
                    foreach (KeyValuePair<string, Space> pair in dict)
                        zero[pair.Key] = CreateZeroArray(pair.Value);
                    return zero;
                }
                case Sequence sequence:
                    return sequence.Stack ? CreateZeroArray(sequence.StackedFeatureSpace!) : System.Array.Empty<object?>();
                case Text text:
                    // "".join(space.characters[0] for _ in range(space.min_length)): the first code point, repeated.
                    return string.Concat(Enumerable.Repeat(Text.CodePoints(text.Characters).First(), (int)text.MinLength));
                case Graph graph:
                    return GraphZero(graph);
                case OneOf oneOf:
                    // A Python tuple (0, zero): the index is a Python int.
                    return new object?[] { 0L, CreateZeroArray(oneOf.Spaces[0]) };
                case var other:
                    throw new CustomSpaceError($"Space of type `{PyBuiltins.TypeRepr(other)}` doesn't have an registered `create_zero_array` function. Register `{PyBuiltins.TypeRepr(other)}` for `create_zero_array` to support it.");
            }
        }

        /// <summary>
        ///     Rescales and shifts a <c>Box</c> onto new bounds (Gymnasium's <c>rescale_box</c>): the new box, the forward map
        ///     (original values to rescaled ones) and the backward map. Unbounded components stay unbounded and are shifted, not
        ///     scaled. What <c>RescaleObservation</c> (forward) and <c>RescaleAction</c> (backward) apply.
        /// </summary>
        /// <param name="box">The space; it must be a <c>Box</c> (an <see cref="ObjectSpace"/> view of one is looked through).</param>
        /// <param name="newMin">
        ///     The new lower bound: an array of the box's shape, or a number broadcast to it. A .NET integer is a Python
        ///     <c>int</c> (int64), a <see cref="double"/> a Python <c>float</c> (float64), a <see cref="float"/> or a 0-d array a
        ///     NumPy scalar of its own dtype, as <c>np.full</c> would take them.
        /// </param>
        /// <param name="newMax">The new upper bound, in the same forms.</param>
        /// <returns>
        ///     The rescaled box (the caller owns it) and the two maps. Each map captures the gradient and intercept arrays, in
        ///     the box's dtype, and returns a new array the caller owns; its dtype is NumPy's promotion of the argument with
        ///     the box's dtype (float32 for float32 values of a float32 box).
        /// </returns>
        /// <exception cref="TypeError">Thrown when <paramref name="box"/> is no <c>Box</c>, or a bound is neither an array nor an integer or float, with Gymnasium's messages.</exception>
        /// <exception cref="ValueError">
        ///     Thrown, with Gymnasium's messages, when a bound's shape is not the box's, an unbounded component's new bound is
        ///     not the same infinity, a new minimum exceeds its maximum, or the box's own low exceeds its high.
        /// </exception>
        /// <remarks>
        ///     <para>
        ///     Gymnasium computes <c>high - low</c> in <c>np.float128</c> where NumPy has it and in float64 otherwise; Windows
        ///     NumPy, the reference platform, has no float128, so this is the float64 path everywhere (port plan §1.3).
        ///     </para>
        ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/utils.py#L160">gymnasium/wrappers/utils.py</see>.</para>
        /// </remarks>
        public static (Box Box, Func<NDArray, NDArray> Forward, Func<NDArray, NDArray> Backward) RescaleBox(Space box, object newMin, object newMax) {
            if (ObjectSpace.Unview(box) is not Box original)
                throw new TypeError($"Expected box to be a Box space, got {PyBuiltins.TypeRepr(ObjectSpace.Unview(box))}");

            using NDArray minimum = BoundArray(newMin, "new_min", original);
            using NDArray maximum = BoundArray(newMax, "new_max", original);

            using (NDArray minimumEqual = minimum == original.Low)
            using (NDArray minimumInfinite = np.isinf(minimum))
            using (NDArray lowInfinite = np.isinf(original.Low))
            using (NDArray eitherInfinite = minimumInfinite | lowInfinite)
            using (NDArray checkedMinimum = minimumEqual[eitherInfinite]) {
                if (!np.all(checkedMinimum))
                    throw new ValueError("For unbounded components, the target bounds must match the original infinity bounds.");
            }

            using (NDArray maximumEqual = maximum == original.High)
            using (NDArray maximumInfinite = np.isinf(maximum))
            using (NDArray highInfinite = np.isinf(original.High))
            using (NDArray eitherInfinite = maximumInfinite | highInfinite)
            using (NDArray checkedMaximum = maximumEqual[eitherInfinite]) {
                if (!np.all(checkedMaximum))
                    throw new ValueError("For unbounded components, the target bounds must match the original infinity bounds.");
            }

            using (NDArray ordered = minimum <= maximum) {
                if (!np.all(ordered))
                    throw new ValueError($"Expected new_min to be less than or equal to new_max, got {PyBuiltins.Str(minimum)} and {PyBuiltins.Str(maximum)}");
            }

            using (NDArray boxOrdered = original.Low <= original.High) {
                if (!np.all(boxOrdered))
                    throw new ValueError($"Expected box.low to be less than or equal to box.high, got {PyBuiltins.Str(original.Low)} and {PyBuiltins.Str(original.High)}");
            }

            // Imagine the x-axis between the old Box and the y-axis being the new Box. Windows NumPy has no float128, so the
            // difference is float64, as on the reference platform.
            using NDArray minimumFinite = np.isfinite(minimum);
            using NDArray maximumFinite = np.isfinite(maximum);
            using NDArray bothFinite = minimumFinite & maximumFinite;

            using NDArray finiteHigh = original.High[bothFinite];
            using NDArray finiteLow = original.Low[bothFinite];
            using NDArray highAsDouble = finiteHigh.astype(np.float64);
            using NDArray lowAsDouble = finiteLow.astype(np.float64);
            using NDArray highLowDifference = highAsDouble - lowAsDouble;

            DType dtype = original.DType!;
            NDArray gradient = np.ones_like(minimum, dtype);
            NDArray intercept = np.zeros_like(minimum, dtype);
            try {
                using (NDArray finiteMaximum = maximum[bothFinite])
                using (NDArray finiteMinimum = minimum[bothFinite])
                using (NDArray span = finiteMaximum - finiteMinimum)
                using (NDArray slope = span / highLowDifference) {
                    // Assigned into the box-dtype array, so the float64 quotient is rounded once to that dtype, as NumPy does.
                    gradient[bothFinite] = slope;
                }

                // In cases where both are finite, the lower operation takes precedence.
                using (NDArray maximumAtFinite = maximum[maximumFinite])
                using (NDArray highAtFinite = original.High[maximumFinite])
                using (NDArray shift = maximumAtFinite - highAtFinite) {
                    intercept[maximumFinite] = shift;
                }

                using (NDArray gradientAtFinite = gradient[minimumFinite])
                using (NDArray lowAtFinite = original.Low[minimumFinite])
                using (NDArray negatedLow = -lowAtFinite)
                using (NDArray scaled = gradientAtFinite * negatedLow)
                using (NDArray minimumAtFinite = minimum[minimumFinite])
                using (NDArray shifted = scaled + minimumAtFinite) {
                    intercept[minimumFinite] = shifted;
                }

                var rescaled = new Box(minimum, maximum, original.Shape, dtype);

                // The maps own gradient and intercept for as long as they live; they are small, and a map may be called long
                // after this method returned, so nothing here disposes them.
                NDArray Forward(NDArray observation) {
                    using NDArray scaledObservation = gradient * observation;
                    return scaledObservation + intercept;
                }

                NDArray Backward(NDArray observation) {
                    using NDArray unshifted = observation - intercept;
                    return unshifted / gradient;
                }

                return (rescaled, Forward, Backward);
            } catch {
                // The maps never reached the caller: release what they would have captured.
                gradient.Dispose();
                intercept.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     A bound of <see cref="RescaleBox"/> as the array Gymnasium makes of it: an array as it is, a number through
        ///     <c>np.full(box.shape, value)</c>; refused when it is neither, or its shape is not the box's.
        /// </summary>
        /// <param name="value">The bound.</param>
        /// <param name="name"><c>new_min</c> or <c>new_max</c>, for the messages.</param>
        /// <param name="box">The box being rescaled.</param>
        /// <returns>A new array the caller owns (a copy for an array bound, so the caller's array is never disposed).</returns>
        /// <exception cref="TypeError">Thrown for a value that is neither an array nor an integer or float.</exception>
        /// <exception cref="ValueError">Thrown when the array's shape is not the box's.</exception>
        private static NDArray BoundArray(object value, string name, Box box) {
            NDArray array;
            switch (value) {
                case NDArray given when given.ndim > 0:
                    array = given.copy();
                    break;
                case NDArray scalar:
                    // A 0-d array stands for a NumPy scalar: np.full keeps its dtype.
                    array = np.full(box.Shape, scalar, scalar.dtype);
                    break;
                case sbyte or byte or short or ushort or int or uint or long or ulong:
                    // A Python int: np.full makes int64.
                    array = np.full(box.Shape, Convert.ToInt64(value), np.int64);
                    break;
                case double number:
                    array = np.full(box.Shape, number, np.float64);
                    break;
                case float number:
                    array = np.full(box.Shape, number, np.float32);
                    break;
                default:
                    throw new TypeError($"Expected {name} to be an integer, float, or numpy array, got {PyBuiltins.TypeRepr(value)}");
            }

            if (!array.Shape.Equals(box.Shape)) {
                string expected = PyBuiltins.Repr(ShapeTuple(box.Shape));
                string actual = PyBuiltins.Repr(ShapeTuple(array.Shape));
                array.Dispose();
                throw new ValueError($"Expected {name}.shape to be {expected}, got {actual}");
            }

            return array;
        }

        /// <summary>A shape as the Python tuple NumPy reports it, <c>(3,)</c>.</summary>
        /// <param name="shape">The shape.</param>
        /// <returns>Its dimensions as an <see langword="object"/>?[] of <see cref="long"/>s, which <see cref="PyBuiltins.Repr"/> prints as a tuple.</returns>
        internal static object?[] ShapeTuple(Shape shape) => shape.dimensions.Select(d => (object?)d).ToArray();

        /// <summary>
        ///     <c>create_zero_array</c> of a <c>Graph</c>: one zero node from the node space batched once, and, when the graph
        ///     has edges, one zero edge linking node 0 to itself.
        /// </summary>
        /// <param name="graph">The graph space.</param>
        /// <returns>A new graph instance the caller owns.</returns>
        private static GraphInstance GraphZero(Graph graph) {
            // A composite node or edge space batches to a composite, so its zero is a tuple or dict, not an array.
            using Space nodeBatch = VectorUtils.BatchSpace(graph.NodeSpace, 1);
            object? nodes = CreateZeroArray(nodeBatch);
            if (graph.EdgeSpace is null)
                return new GraphInstance(nodes, null, null);
            using Space edgeBatch = VectorUtils.BatchSpace(graph.EdgeSpace, 1);
            object? edges = CreateZeroArray(edgeBatch);
            return new GraphInstance(nodes, edges, np.zeros(new Shape(1L, 2L), np.int64));
        }
    }
}
