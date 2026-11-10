#nullable enable
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Vector {
    /// <summary>Gymnasium's vector wrapper tests (<c>test_vector_wrapper.py</c>), run against <see cref="VectorWrapper{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_wrapper.py">tests/vector/test_vector_wrapper.py</see>.</remarks>
    [TestClass]
    public class TestVectorWrapper {
        /// <summary>Test vector environment wrapper inheritance: a subclass's <c>reset</c> override runs.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_wrapper.py#L34">tests/vector/test_vector_wrapper.py (test_vector_env_wrapper_inheritance)</see>.</remarks>
        [TestMethod]
        public void TestVectorEnvWrapperInheritance() {
            using VectorEnv<NDArray, NDArray> env = MakeVec<NDArray, NDArray>("FrozenLake-v1", vectorizationMode: "sync");
            var wrapped = new DummyVectorWrapper(env);
            wrapped.Reset();
            Assert.AreEqual(1, wrapped.Counter);

            env.Close();
        }

        /// <summary>Test if <c>set_attr</c> and <c>call</c> of a vector wrapper reach the vector env it wraps.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_wrapper.py#L44">tests/vector/test_vector_wrapper.py (test_vector_env_wrapper_attributes)</see>.</remarks>
        [TestMethod]
        public void TestVectorEnvWrapperAttributes() {
            using var env = (SyncVectorEnv<NDArray, NDArray>)MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 3, vectorizationMode: "sync");
            var wrapped = new DummyVectorWrapper(MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 3, vectorizationMode: "sync"));
            var inner = (SyncVectorEnv<NDArray, NDArray>)wrapped.Env;

            CollectionAssert.AreEqual(env.Call("gravity").ToArray(), inner.Call("gravity").ToArray());
            env.SetAttr("gravity", new List<object?> { 20.0, 20.0, 20.0 });
            inner.SetAttr("gravity", new List<object?> { 20.0, 20.0, 20.0 });
            CollectionAssert.AreEqual(env.GetAttr("gravity").ToArray(), inner.GetAttr("gravity").ToArray());

            env.Close();
            wrapped.Close();
        }

        /// <summary>Test if <c>metadata</c> of a vector wrapper forwards to the vector env it wraps, and stops when the env's is replaced.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_wrapper.py#L59">tests/vector/test_vector_wrapper.py (test_vector_env_metadata)</see>.</remarks>
        [TestMethod]
        public void TestVectorEnvMetadata() {
            using VectorEnv<NDArray, NDArray> env = MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 3, vectorizationMode: "sync");
            var wrapped = new DummyVectorWrapper(MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 3, vectorizationMode: "sync"));

            Assert.AreEqual(env.Metadata, wrapped.Metadata);
            env.Metadata = new Metadata { [Metadata.RenderModesKey] = new List<string> { "rgb_array" } };
            Assert.AreNotEqual(env.Metadata, wrapped.Metadata);

            env.Close();
            wrapped.Close();
        }

        /// <summary>
        ///     Gymnasium's <c>DummyVectorWrapper</c>: a vector wrapper counting its resets, to show a subclass's override runs.
        /// </summary>
        private sealed class DummyVectorWrapper : VectorWrapper<NDArray, NDArray> {
            /// <summary>Wraps <paramref name="env"/> with the counter at zero.</summary>
            /// <param name="env">The vector env.</param>
            public DummyVectorWrapper(VectorEnv<NDArray, NDArray> env) : base(env) { }

            /// <summary>How many times <c>reset</c> ran (<c>counter</c>).</summary>
            public int Counter { get; private set; }

            /// <summary>Counts the reset, then resets the env.</summary>
            /// <param name="seed">The seed, passed on.</param>
            /// <param name="options">The options, passed on.</param>
            /// <returns>The env's reset.</returns>
            public override VectorResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
                Counter++;
                return base.Reset(seed, options);
            }
        }
    }
}
