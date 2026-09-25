using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using NumSharp;
using Python.Runtime;

namespace Gym.Tests.Parity.Oracle {
    /// <summary>
    ///     A live Gymnasium env, the Python half of a <see cref="GymPair"/>: an instance of <c>gym_oracle.OracleEnv</c>, created
    ///     through <c>gymnasium.make</c>.
    /// </summary>
    /// <remarks>Every member takes the GIL itself. Dispose it: that closes the env (pygame surfaces included) under the GIL.</remarks>
    public sealed class PyGymEnv : IDisposable {
        /// <summary>The <c>OracleEnv</c> instance.</summary>
        private readonly PyHandle _env;

        /// <summary>Wraps an <c>OracleEnv</c>.</summary>
        /// <param name="env">The handle, owned.</param>
        /// <param name="description">How the env was made, for reports.</param>
        private PyGymEnv(PyHandle env, string description) {
            _env = env;
            Description = description;
        }

        /// <summary>How the env was made, in Python spelling: <c>gymnasium.make('CartPole-v1').unwrapped</c>.</summary>
        public string Description { get; }

        /// <summary>Creates an env through <c>gymnasium.make</c>, exactly as a Gymnasium user would.</summary>
        /// <param name="envId">A registered id, for example <c>CartPole-v1</c>.</param>
        /// <param name="kwargs">Constructor keyword arguments in Python spelling (<c>continuous</c>, <c>enable_wind</c>); values are converted by <c>PyGym.ToPython</c>.</param>
        /// <param name="viaMake">
        ///     <see langword="true"/> to step the made env with its wrapper stack (<c>TimeLimit</c>, <c>OrderEnforcing</c>,
        ///     <c>PassiveEnvChecker</c>); <see langword="false"/> to step its <c>unwrapped</c> env, the right counterpart for a
        ///     Gym.NET env that has no wrappers yet.
        /// </param>
        /// <param name="renderMode"><c>rgb_array</c> to enable frame comparisons, or <see langword="null"/>.</param>
        /// <returns>The env; the caller disposes it.</returns>
        /// <exception cref="PythonException">Thrown when <c>gymnasium.make</c> raises.</exception>
        public static PyGymEnv Make(string envId, IReadOnlyDictionary<string, object?>? kwargs = null, bool viaMake = false, string? renderMode = null) {
            kwargs ??= new Dictionary<string, object?>();
            string description = $"gymnasium.make('{envId}'{string.Concat(kwargs.Select(k => $", {k.Key}={Format(k.Value)}"))}{(renderMode is null ? "" : $", render_mode='{renderMode}'")}){(viaMake ? "" : ".unwrapped")}";
            using (Py.GIL()) {
                using var id = new PyString(envId);
                using var arguments = new PyDict();
                foreach (var (name, value) in kwargs) {
                    using PyObject converted = PyGym.ToPython(value);
                    arguments.SetItem(name, converted);
                }

                using PyObject make = viaMake.ToPython();
                using PyObject mode = PyGym.ToPython(renderMode);
                using PyObject type = GymPythonSession.Oracle.GetAttr("OracleEnv");
                return new PyGymEnv(new PyHandle(type.Invoke(id, arguments, make, mode), description), description);
            }
        }

        /// <summary>Resets with a seed: <c>env.reset(seed=seed)</c>.</summary>
        /// <param name="seed">The seed.</param>
        /// <returns>The described <c>(observation, info)</c>, or what the reset raised.</returns>
        public CallResult Reset(long seed) => CallResult.FromOutcome(_env.Invoke("reset", seed), reset: true);

        /// <summary>Steps: <c>env.step(action)</c>, with the action's Python representation.</summary>
        /// <param name="action">The action; it stays owned by the caller.</param>
        /// <returns>The described 5-tuple, or what the step raised.</returns>
        public CallResult Step(ParityAction action) {
            using (Py.GIL()) {
                using PyObject step = _env.Object.GetAttr("step");
                using PyObject invoke = GymPythonSession.Oracle.GetAttr("invoke");
                // The action's own object is passed (still owned by the action), not a conversion of it, so a policy's
                // float64 array reaches Gymnasium exactly as the policy made it.
                using PyObject result = invoke.Invoke(step, action.Python());
                return CallResult.FromOutcome(PyGym.ReadOutcome(result), reset: false);
            }
        }

        /// <summary>The action a named <c>gym_oracle</c> policy picks for the last observation.</summary>
        /// <param name="policy">The policy's name, a key of <c>gym_oracle.POLICIES</c>.</param>
        /// <returns>The action; the caller disposes it.</returns>
        /// <exception cref="PythonException">Thrown when the policy raises.</exception>
        public ParityAction Policy(string policy) {
            using (Py.GIL()) {
                using var name = new PyString(policy);
                return ParityAction.FromPython(_env.Object.InvokeMethod("policy", name));
            }
        }

        /// <summary>The env generator's state, <c>env.unwrapped.np_random.bit_generator.state</c>.</summary>
        /// <returns>The described state dictionary.</returns>
        /// <exception cref="Microsoft.VisualStudio.TestTools.UnitTesting.AssertFailedException">Thrown when reading it raises.</exception>
        public ValueFacts RngState() => _env.Invoke("rng_state").Expect($"{Description}.np_random.bit_generator.state");

        /// <summary>Renders: <c>env.render()</c>, which for <c>render_mode='rgb_array'</c> is an <c>(H, W, 3)</c> uint8 frame.</summary>
        /// <returns>The frame.</returns>
        /// <exception cref="Microsoft.VisualStudio.TestTools.UnitTesting.AssertFailedException">Thrown when rendering raises.</exception>
        /// <exception cref="ArgumentException">Thrown when the env returns something other than an RGB frame (no render mode).</exception>
        public Frame Render() => Frame.FromFacts(_env.Invoke("render").Expect($"{Description}.render()"));

        /// <summary>Injects state: <c>env.unwrapped.&lt;name&gt; = np.array(values)</c> (float64, from a NumSharp copy).</summary>
        /// <param name="name">The attribute, for example <c>state</c>.</param>
        /// <param name="values">The float64 values.</param>
        /// <exception cref="Microsoft.VisualStudio.TestTools.UnitTesting.AssertFailedException">Thrown when the assignment raises.</exception>
        public void SetAttribute(string name, double[] values) {
            using NDArray array = np.array(values);
            _env.Invoke("set_attribute", name, array).Expect($"{Description}.{name} = ...");
        }

        /// <summary>Reads an attribute of the unwrapped env.</summary>
        /// <param name="name">The attribute.</param>
        /// <returns>The outcome; an <c>AttributeError</c> when the env has no such attribute.</returns>
        public Outcome GetAttribute(string name) => _env.Invoke("get_attribute", name);

        /// <summary>Seeds the action space: <c>env.action_space.seed(seed)</c>.</summary>
        /// <param name="seed">The seed.</param>
        /// <exception cref="Microsoft.VisualStudio.TestTools.UnitTesting.AssertFailedException">Thrown when seeding raises.</exception>
        public void SeedActionSpace(long seed) => _env.Invoke("seed_action_space", seed).Expect($"{Description}.action_space.seed({seed})");

        /// <summary>Samples the action space: <c>env.action_space.sample()</c>, kept raw so that it can be fed to <c>step</c>. Must run under the GIL.</summary>
        /// <returns>The sample; the caller owns it.</returns>
        internal PyObject SampleActionRaw() => _env.Object.InvokeMethod("sample_action");

        /// <summary>Closes the env and releases it, under the GIL. A failing <c>close</c> is ignored: the comparison is over by then.</summary>
        public void Dispose() {
            try {
                _env.Invoke("close");
            } finally {
                _env.Dispose();
            }
        }

        /// <inheritdoc />
        public override string ToString() => Description;

        /// <summary>Formats a keyword argument's value the way Python prints it, for <see cref="Description"/>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>Its Python spelling.</returns>
        private static string Format(object? value) {
            return value switch {
                null => "None",
                bool flag => flag ? "True" : "False",
                string text => $"'{text}'",
                IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
                _ => value.ToString() ?? "",
            };
        }
    }
}
