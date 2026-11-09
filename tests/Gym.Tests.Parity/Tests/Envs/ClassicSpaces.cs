using System;
using System.Collections.Generic;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Envs {
    /// <summary>
    ///     The space comparisons the classic-control env cells share: a Gym.NET env's <see cref="Box"/> or
    ///     <see cref="Discrete"/> against the same space of Gymnasium's env made by id, attribute by attribute, and seeded
    ///     samples with the generator state behind them.
    /// </summary>
    internal static class ClassicSpaces {
        /// <summary>
        ///     A <see cref="Box"/> against the unwrapped Gymnasium env's space: <c>high</c>, <c>low</c>, <c>bounded_above</c>,
        ///     <c>bounded_below</c>, <c>dtype</c> and <c>shape</c>, each byte for byte.
        /// </summary>
        /// <param name="what">The space, for reports (<c>Pendulum-v1 observation_space</c>).</param>
        /// <param name="id">The registered id Gymnasium's env is made from.</param>
        /// <param name="spaceRow">The facade row that reads the space off the env.</param>
        /// <param name="space">Gym.NET's space.</param>
        /// <param name="kwargs">Constructor keyword arguments in Python spelling, or <see langword="null"/>.</param>
        /// <exception cref="AssertFailedException">Thrown at the first attribute that differs.</exception>
        internal static void AssertBoxSame(string what, string id, Func<PyObject, PyObject> spaceRow, Box space, IReadOnlyDictionary<string, object?>? kwargs = null) {
            using PyRef python = PyRef.Of(() => CartPoleLiveParityTests.MadeSpace(id, spaceRow, kwargs));
            ByteContract.AssertSame($"{what}.high", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.high(python.Object)), Outcome.Returned(ValueFacts.Of(space.High)));
            ByteContract.AssertSame($"{what}.low", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.low(python.Object)), Outcome.Returned(ValueFacts.Of(space.Low)));
            ByteContract.AssertSame($"{what}.bounded_above", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.bounded_above(python.Object)), Outcome.Returned(ValueFacts.Of(space.BoundedAbove)));
            ByteContract.AssertSame($"{what}.bounded_below", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.bounded_below(python.Object)), Outcome.Returned(ValueFacts.Of(space.BoundedBelow)));
            ByteContract.AssertSame($"{what}.dtype", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.dtype(python.Object)), Outcome.Returned(ValueFacts.Of(space.DType)));
            ByteContract.AssertSame($"{what}.shape", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.shape(python.Object)), Outcome.Returned(Spaces.DiscreteLiveParityTests.ShapeFacts(space.Shape)));
        }

        /// <summary>
        ///     An env's action space against Gymnasium's: a <see cref="Discrete"/> by <c>n</c>, <c>start</c> and dtype, a
        ///     <see cref="Box"/> by <see cref="AssertBoxSame"/>; then 20 samples after <c>seed(5)</c> and the generator state.
        /// </summary>
        /// <param name="id">The registered id Gymnasium's env is made from.</param>
        /// <param name="space">Gym.NET's action space.</param>
        /// <param name="kwargs">Constructor keyword arguments in Python spelling, or <see langword="null"/>.</param>
        /// <exception cref="AssertFailedException">Thrown at the first attribute or sample that differs.</exception>
        /// <exception cref="NotSupportedException">Thrown for a space of another type.</exception>
        internal static void AssertActionSpaceSame(string id, Space space, IReadOnlyDictionary<string, object?>? kwargs = null) {
            string what = $"{id} action_space";
            switch (space) {
                case Discrete discrete: {
                    using PyRef discreteSpace = PyRef.Of(() => CartPoleLiveParityTests.MadeSpace(id, Python.gymnasium.Env.action_space, kwargs));
                    ByteContract.AssertSame($"{what}.n", Outcome.CapturePython(() => Python.gymnasium.spaces.discrete.Discrete.n(discreteSpace.Object)), Outcome.Returned(ValueFacts.Of(discrete.N)));
                    ByteContract.AssertSame($"{what}.start", Outcome.CapturePython(() => Python.gymnasium.spaces.discrete.Discrete.start(discreteSpace.Object)), Outcome.Returned(ValueFacts.Of(discrete.Start)));
                    ByteContract.AssertSame($"{what}.dtype", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.dtype(discreteSpace.Object)), Outcome.Returned(ValueFacts.Of(discrete.DType)));
                    break;
                }
                case Box box:
                    AssertBoxSame(what, id, Python.gymnasium.Env.action_space, box, kwargs);
                    break;
                default:
                    throw new NotSupportedException($"{id}'s action space is a {space.GetType().Name}, which the classic-control cells do not compare.");
            }

            using PyRef python = PyRef.Of(() => CartPoleLiveParityTests.MadeSpace(id, Python.gymnasium.Env.action_space, kwargs));
            Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.seed(python.Object, 5)).Expect("action_space.seed(5)");
            space.Seed(5);
            for (int i = 1; i <= 20; i++)
                ByteContract.AssertSame($"{what}.sample() #{i} after seed(5)", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(python.Object)), Outcome.Capture(() => space.Sample()));
            ByteContract.AssertSame($"{what} generator state", RngFacts.OfPythonSpace(python.Object), Outcome.Returned(GeneratorProbe.StateOf(space)));
        }
    }
}
