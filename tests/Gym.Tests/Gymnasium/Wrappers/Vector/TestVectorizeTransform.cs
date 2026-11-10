#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Spaces;
using Gym.Vector;
using Gym.Wrappers;
using Gym.Wrappers.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers.Vector {
    /// <summary>
    ///     Gymnasium's tests of the vectorizing wrappers (<c>test_vectorize_transform.py</c>): single-env transforms whose
    ///     function changes the space's type, applied per sub-env.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_vectorize_transform.py">tests/wrappers/vector/test_vectorize_transform.py</see>.</remarks>
    [TestClass]
    public class TestVectorizeTransform {
        /// <summary>
        ///     <c>VectorizeTransformAction</c> of <c>TransformAction</c> whose actions are dicts of a <c>Box</c>: the batched
        ///     dict samples are split per sub-env, mapped to the env's box, and stacked again.
        /// </summary>
        /// <remarks>
        ///     The dict space is given through an <see cref="ObjectSpace"/> view, as the untyped stand-in env takes every space:
        ///     <see cref="TransformAction{TObs, TWAct, TAct}"/> is closed over <see langword="object"/> for the generic
        ///     <c>typeof(TransformAction&lt;,,&gt;)</c>.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_vectorize_transform.py#L11">tests/wrappers/vector/test_vectorize_transform.py (test_vectorize_box_to_dict_action)</see>.
        /// </remarks>
        [TestMethod]
        public void TestVectorizeBoxToDictAction() {
            static object? Func(object? x) => ((IReadOnlyDictionary<string, object?>)x!)["key"];

            var envs = new SyncVectorEnv(Enumerable.Range(0, 2).Select(_ => (Func<IEnv>)(() => new GenericTestEnv())));
            using var wrapped = new VectorizeTransformAction<object?, object?, object?>(envs, typeof(TransformAction<,,>), new Info {
                ["func"] = (Func<object?, object?>)Func,
                ["action_space"] = new ObjectSpace(new Dict(new Dictionary<string, Space> { ["key"] = new Box(0.0, 1.0, new Shape(1), np.float32) })),
            });

            wrapped.Reset();
            wrapped.Step(wrapped.ActionSpace.Sample());
            wrapped.Close();
        }

        /// <summary>
        ///     <c>make_vec("CartPole-v1", 2, ASYNC)</c> with a <c>TransformObservation</c> on each sub-env that splits the state
        ///     into a dict: the batched observations of the reset and of a step are in the batched dict space.
        /// </summary>
        /// <remarks>
        ///     The asynchronous vector env comes with P6.2; until then <c>make_vec</c> refuses the mode.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_vectorize_transform.py#L31">tests/wrappers/vector/test_vectorize_transform.py (test_vectorize_dict_to_box_obs)</see>.
        /// </remarks>
        [TestMethod]
        [OpenBugs("P6.2")]
        public void TestVectorizeDictToBoxObs() {
            static IEnv Wrapper(IEnv env) =>
                new TransformObservation<Info, NDArray, NDArray>((Env<NDArray, NDArray>)env, x => new Info { ["key1"] = x["0:1"], ["key2"] = x["1:"] },
                    new Dict(new Dictionary<string, Space> {
                        ["key1"] = new Box(double.NegativeInfinity, double.PositiveInfinity, new Shape(1)),
                        ["key2"] = new Box(double.NegativeInfinity, double.PositiveInfinity, new Shape(3)),
                    }));

            using IVectorEnv envs = MakeVec("CartPole-v1", numEnvs: 2, vectorizationMode: "async", wrappers: new Func<IEnv, IEnv>[] { Wrapper });
            VectorResetResult<object?> reset = envs.Reset();
            Assert.IsTrue(envs.ObservationSpace.Contains(reset.Observations));
            VectorStepResult<object?> step = envs.Step(envs.ActionSpace.Sample());
            Assert.IsTrue(envs.ObservationSpace.Contains(step.Observations));
            envs.Close();
        }
    }
}
