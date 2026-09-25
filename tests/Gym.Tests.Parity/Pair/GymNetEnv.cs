using System;
using Gym.Envs;
using Gym.Observations;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Oracle;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Parity.Pair {
    /// <summary>
    ///     Puts a Gym.NET env (possibly a stack of wrappers) into a <see cref="GymPair"/> through <see cref="IEnv"/>, the
    ///     contract Gymnasium's envs follow too: <c>Reset(seed)</c> returns <c>(obs, info)</c>, <c>Step</c> the 5-tuple with a
    ///     float64 reward. Nothing is adapted, so <see cref="Adaptations"/> is empty and every value compared is exactly what
    ///     Gym.NET returns.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Attributes are read and written the way the Python side does it, on the innermost env and by Python name
    ///     (<c>env.unwrapped.state</c>), through Gym.NET's attribute protocol (<see cref="PyAttributes"/>), so the cells
    ///     exercise it too.
    ///     </para>
    ///     <para>
    ///     Every observation is a fresh array since the core-API work (B-13 is fixed, and every env returns a new array per
    ///     call), so this adapter describes and then disposes it, keeping NumSharp's leak gate meaningful.
    ///     </para>
    /// </remarks>
    public sealed class GymNetEnv : IDotNetEnv {
        /// <summary>The env, owned.</summary>
        private readonly IEnv _env;

        /// <summary>Wraps an env; the adapter owns and disposes it.</summary>
        /// <param name="env">The Gym.NET env, bare or wrapped.</param>
        public GymNetEnv(IEnv env) {
            _env = env;
            Name = $"Gym.NET {PyAttributes.PyTypeName(env.GetType())}";
        }

        /// <summary>The wrapped env, for tests that also inspect its spaces.</summary>
        public IEnv Env => _env;

        /// <inheritdoc />
        public string Name { get; }

        /// <inheritdoc />
        public string[] Adaptations { get; } = Array.Empty<string>();

        /// <inheritdoc />
        public CallResult Reset(long seed) {
            try {
                (object? observation, Info info) = _env.Reset(seed);
                try {
                    return CallResult.Of(new Transition(ValueFacts.Of(observation), null, null, null, ValueFacts.Of(info)));
                } finally {
                    (observation as NDArray)?.Dispose();
                }
            } catch (Exception exception) when (exception is not UnitTestAssertException) {
                return CallResult.Threw(RaisedError.FromDotNet(exception));
            }
        }

        /// <inheritdoc />
        public CallResult Step(object action) {
            try {
                (object? observation, double reward, bool terminated, bool truncated, Info info) = _env.Step(action);
                try {
                    return CallResult.Of(new Transition(ValueFacts.Of(observation), ValueFacts.Of(reward), ValueFacts.FromBool(terminated), ValueFacts.FromBool(truncated), ValueFacts.Of(info)));
                } finally {
                    (observation as NDArray)?.Dispose();
                }
            } catch (Exception exception) when (exception is not UnitTestAssertException) {
                return CallResult.Threw(RaisedError.FromDotNet(exception));
            }
        }

        /// <inheritdoc />
        /// <remarks>
        ///     Reads the innermost env's <c>_np_random</c> without creating it (reading <see cref="IEnv.NpRandom"/> would draw
        ///     a fresh entropy generator into an env that has not been seeded, changing what it is being asked about).
        /// </remarks>
        public bool TryGetRngState(out ValueFacts? state, out string reason) {
            if (PyAttributes.GetAttr(_env.Unwrapped, "_np_random") is not Generator generator) {
                state = null;
                reason = $"{Name}'s unwrapped env has no generator yet (not seeded or drawn from)";
                return false;
            }

            state = RngFacts.Of(generator);
            reason = "";
            return true;
        }

        /// <inheritdoc />
        /// <remarks>
        ///     Assigns <c>env.unwrapped.&lt;name&gt;</c> through <see cref="PyAttributes.SetAttr"/>. CartPole's <c>state</c> is a
        ///     float64 <see cref="T:double[]"/>, which receives a copy of the values; an <see cref="NDArray"/> attribute
        ///     receives a new float64 array.
        /// </remarks>
        /// <exception cref="NotSupportedException">Thrown when the env has no such attribute.</exception>
        public void SetAttribute(string name, double[] values) {
            IEnv unwrapped = _env.Unwrapped;
            if (!PyAttributes.HasAttr(unwrapped, name))
                throw new NotSupportedException($"{Name} has no attribute '{name}' on its unwrapped env.");
            object? current = PyAttributes.GetAttr(unwrapped, name);
            if (current is NDArray)
                PyAttributes.SetAttr(unwrapped, name, np.array(values));
            else
                PyAttributes.SetAttr(unwrapped, name, (double[])values.Clone());
        }

        /// <inheritdoc />
        /// <remarks>
        ///     A <see cref="T:double[]"/> attribute is described as the float64 array Gymnasium holds (through NumSharp, so it
        ///     compares with NumPy's <c>(4,) &lt;f8</c>); other values are described as they are.
        /// </remarks>
        public ValueFacts? GetAttribute(string name) {
            IEnv unwrapped = _env.Unwrapped;
            if (!PyAttributes.HasAttr(unwrapped, name))
                return null;
            object? value = PyAttributes.GetAttr(unwrapped, name);
            if (value is double[] doubles) {
                using NDArray array = np.array(doubles);
                return ValueFacts.Of(array);
            }

            return ValueFacts.Of(value);
        }

        /// <inheritdoc />
        /// <exception cref="NotSupportedException">Thrown when the env renders something other than an <c>rgb_array</c> frame.</exception>
        public Frame Render() {
            switch (_env.Render()) {
                case RenderResult.Rgb rgb: {
                    using NDArray frame = rgb.Frame;
                    return new Frame((int)frame.shape[0], (int)frame.shape[1], frame.ToArray<byte>());
                }
                case var other:
                    throw new NotSupportedException($"{Name}.Render() returned {other}, not an rgb_array frame; build the env with render_mode \"rgb_array\".");
            }
        }

        /// <inheritdoc />
        public void SeedActionSpace(long seed) => _env.ActionSpace.Seed(seed);

        /// <inheritdoc />
        public object SampleAction() => _env.ActionSpace.Sample();

        /// <summary>Closes the env.</summary>
        public void Dispose() => _env.Dispose();
    }
}
