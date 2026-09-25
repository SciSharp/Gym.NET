using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Pair;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Oracle {
    /// <summary>
    ///     A live Gymnasium env, the Python half of a <see cref="GymPair"/>, created through <c>gymnasium.make</c> and driven
    ///     through the <see cref="Python"/> tree.
    /// </summary>
    /// <remarks>
    ///     Owns three Python references (the made env, its unwrapped env, and the last observation) and releases them
    ///     under the GIL on <see cref="Dispose"/>, after closing the env. Every member takes the GIL itself.
    /// </remarks>
    public sealed class PyGymEnv : IDisposable {
        /// <summary>The <c>gymnasium.make</c> result, with its wrapper stack.</summary>
        private PyObject? _made;

        /// <summary><c>made.unwrapped</c>, the bare env.</summary>
        private PyObject? _unwrapped;

        /// <summary>Whether reset and step go to the made env (wrappers included) rather than to the unwrapped one.</summary>
        private readonly bool _viaMake;

        /// <summary>Gymnasium's last observation, which policies act on; <see langword="null"/> before the first reset.</summary>
        private PyObject? _lastObservation;

        /// <summary>Takes ownership of the made env and its unwrapped env.</summary>
        /// <param name="made">The made env.</param>
        /// <param name="unwrapped">Its unwrapped env (a separate reference).</param>
        /// <param name="viaMake">Whether reset and step go to <paramref name="made"/>.</param>
        /// <param name="description">How the env was made, for reports.</param>
        private PyGymEnv(PyObject made, PyObject unwrapped, bool viaMake, string description) {
            _made = made;
            _unwrapped = unwrapped;
            _viaMake = viaMake;
            Description = description;
        }

        /// <summary>How the env was made, in Python spelling: <c>gymnasium.make('CartPole-v1').unwrapped</c>.</summary>
        public string Description { get; }

        /// <summary>The env reset and step go to: the made env or the unwrapped one. Borrowed; never dispose it.</summary>
        /// <exception cref="ObjectDisposedException">Thrown after <see cref="Dispose"/>.</exception>
        private PyObject Env => (_viaMake ? _made : _unwrapped) ?? throw new ObjectDisposedException(Description);

        /// <summary>The unwrapped env, where attributes like <c>state</c> and <c>np_random</c> live. Borrowed; never dispose it.</summary>
        /// <exception cref="ObjectDisposedException">Thrown after <see cref="Dispose"/>.</exception>
        private PyObject Unwrapped => _unwrapped ?? throw new ObjectDisposedException(Description);

        /// <summary>Creates an env through <c>gymnasium.make</c>, exactly as a Gymnasium user would.</summary>
        /// <param name="envId">A registered id, for example <c>CartPole-v1</c>.</param>
        /// <param name="kwargs">Constructor keyword arguments in Python spelling (<c>continuous</c>, <c>sutton_barto_reward</c>).</param>
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
            return PythonRuntime.Run(() => {
                PyObject made = Python.gymnasium.make(envId, renderMode, kwargs);
                try {
                    return new PyGymEnv(made, Python.gymnasium.Env.unwrapped(made), viaMake, description);
                } catch {
                    made.Dispose();
                    throw;
                }
            });
        }

        /// <summary>Resets with a seed: <c>env.reset(seed=seed, options=options)</c>.</summary>
        /// <param name="seed">The seed.</param>
        /// <param name="options">Reset options in Python spelling (<c>low</c>, <c>high</c>), or <see langword="null"/>.</param>
        /// <returns>The described <c>(observation, info)</c>, or what the reset raised.</returns>
        public CallResult Reset(long seed, IReadOnlyDictionary<string, object?>? options = null) {
            return Call(() => Python.gymnasium.Env.reset(Env, seed, options), reset: true);
        }

        /// <summary>Steps: <c>env.step(action)</c>, with the action's Python representation.</summary>
        /// <param name="action">The action; it stays owned by the caller.</param>
        /// <returns>The described 5-tuple, or what the step raised.</returns>
        public CallResult Step(ParityAction action) {
            // The action's own object is passed (still owned by the action), not a conversion of it, so a policy's float64
            // array reaches Gymnasium exactly as the policy made it.
            return PythonRuntime.Run(() => Call(() => Python.gymnasium.Env.step(Env, action.PythonValue()), reset: false));
        }

        /// <summary>
        ///     The action a named policy picks for Gymnasium's last observation: <c>cartpole_balance</c> (a proportional-derivative
        ///     controller) or <c>lunar_lander_heuristic</c> (Gymnasium's own <c>lunar_lander.heuristic</c>).
        /// </summary>
        /// <param name="policy">The policy's name.</param>
        /// <returns>The action; the caller disposes it.</returns>
        /// <exception cref="ArgumentException">Thrown for an unknown policy.</exception>
        /// <exception cref="InvalidOperationException">Thrown before the first reset.</exception>
        /// <exception cref="PythonException">Thrown when Gymnasium's heuristic raises.</exception>
        public ParityAction Policy(string policy) {
            return PythonRuntime.Run(() => {
                PyObject observation = _lastObservation ?? throw new InvalidOperationException("A policy needs an observation: reset first.");
                switch (policy) {
                    case "cartpole_balance": {
                        using NDArray values = observation.ToNDArray();
                        // NumSharp's typed getters read the buffer as the requested type without converting, so the float32
                        // observation is widened (exactly, as Python's float() does) before GetDouble reads it.
                        using NDArray widened = values.astype(np.float64);
                        return ParityAction.Discrete(CartPoleBalance(widened));
                    }
                    case "lunar_lander_heuristic":
                        return ParityAction.FromPython(Python.gymnasium.envs.box2d.lunar_lander.heuristic(Unwrapped, observation));
                    default:
                        throw new ArgumentException($"Unknown policy '{policy}'.", nameof(policy));
                }
            });
        }

        /// <summary>The env generator's state, <c>env.unwrapped.np_random.bit_generator.state</c>.</summary>
        /// <returns>The described state dictionary.</returns>
        /// <exception cref="PythonException">Thrown when reading it raises.</exception>
        public ValueFacts RngState() {
            return PythonRuntime.Run(() => {
                using PyObject generator = Python.gymnasium.Env.np_random(Unwrapped);
                using PyObject bitGenerator = Python.numpy.random.Generator.bit_generator(generator);
                using PyObject state = Python.numpy.random.BitGenerator.state(bitGenerator);
                return ValueFacts.FromPython(state);
            });
        }

        /// <summary>Renders: <c>env.render()</c>, which for <c>render_mode='rgb_array'</c> is an <c>(H, W, 3)</c> uint8 frame.</summary>
        /// <returns>The frame.</returns>
        /// <exception cref="PythonException">Thrown when rendering raises.</exception>
        /// <exception cref="ArgumentException">Thrown when the env returns something other than an RGB frame (no render mode).</exception>
        public Frame Render() {
            return PythonRuntime.Run(() => {
                using PyObject frame = Python.gymnasium.Env.render(Env);
                return Frame.FromFacts(ValueFacts.FromPython(frame));
            });
        }

        /// <summary>
        ///     Injects state: <c>env.unwrapped.&lt;name&gt; = np.array(values)</c>, a float64 NumPy copy made by NumSharp's interop.
        ///     The attribute is chosen at run time, so it is set by name rather than through a facade row.
        /// </summary>
        /// <param name="name">The attribute, for example <c>state</c>.</param>
        /// <param name="values">The float64 values.</param>
        /// <exception cref="PythonException">Thrown when the assignment raises.</exception>
        public void SetAttribute(string name, double[] values) {
            PythonRuntime.Run(() => {
                using NDArray array = np.array(values);
                using PyObject value = array.ToNumpyCopy();
                Unwrapped.SetAttr(name, value);
            });
        }

        /// <summary>Reads an attribute of the unwrapped env, chosen at run time.</summary>
        /// <param name="name">The attribute.</param>
        /// <returns>The outcome; an <c>AttributeError</c> when the env has no such attribute.</returns>
        public Outcome GetAttribute(string name) => Outcome.CapturePython(() => Unwrapped.GetAttr(name));

        /// <summary>Seeds the action space: <c>env.action_space.seed(seed)</c>.</summary>
        /// <param name="seed">The seed.</param>
        /// <exception cref="PythonException">Thrown when seeding raises.</exception>
        public void SeedActionSpace(long seed) {
            PythonRuntime.Run(() => {
                using PyObject space = Python.gymnasium.Env.action_space(Env);
                using PyObject _ = Python.gymnasium.spaces.space.Space.seed(space, seed);
            });
        }

        /// <summary>Samples the action space: <c>env.action_space.sample()</c>, kept raw so it can be fed to <c>step</c>. Must run under the GIL.</summary>
        /// <returns>The sample; the caller owns it.</returns>
        internal PyObject SampleActionRaw() {
            using PyObject space = Python.gymnasium.Env.action_space(Env);
            return Python.gymnasium.spaces.space.Space.sample(space);
        }

        /// <summary>Closes the env and releases the Python references, under the GIL. A failing <c>close</c> is ignored: the comparison is over by then.</summary>
        public void Dispose() {
            if (_made is null)
                return;
            PythonRuntime.Run(() => {
                try {
                    Python.gymnasium.Env.close(_made);
                } catch (PythonException) {
                    // Closing is best effort: a failure here cannot change a comparison that already finished.
                }

                _lastObservation?.Dispose();
                _unwrapped?.Dispose();
                _made.Dispose();
                _lastObservation = null;
                _unwrapped = null;
                _made = null;
            });
        }

        /// <inheritdoc />
        public override string ToString() => Description;

        /// <summary>
        ///     The <c>cartpole_balance</c> policy: push towards the side the pole falls to, damped by the angular velocity,
        ///     with the cart's position and velocity as a weak restoring term. It keeps CartPole-v1 upright for all 500 steps on
        ///     seeds 0, 1, 7, 42, 1000 and 1234 (measured against Gymnasium on 2026-09-25).
        /// </summary>
        /// <param name="observation">Gymnasium's float32 observation, crossed to NumSharp.</param>
        /// <returns>1 (push right) or 0 (push left).</returns>
        /// <remarks>
        ///     The arithmetic is Python's for <c>0.1 * x + 0.5 * x_dot + 10.0 * theta + 2.0 * theta_dot &gt; 0.0</c> on
        ///     <c>float(v)</c> of each element: float64, left to right. The decision therefore matches what a Python policy
        ///     would pick, so both halves of a pair receive the action Gymnasium's own observation calls for.
        /// </remarks>
        private static long CartPoleBalance(NDArray observation) {
            double x = observation.GetDouble(0), xDot = observation.GetDouble(1), theta = observation.GetDouble(2), thetaDot = observation.GetDouble(3);
            return 0.1 * x + 0.5 * xDot + 10.0 * theta + 2.0 * thetaDot > 0.0 ? 1 : 0;
        }

        /// <summary>
        ///     Runs a reset or step call, keeps the observation for policies, and describes the result. Must run under the GIL.
        /// </summary>
        /// <param name="call">The facade call; it returns a new reference.</param>
        /// <param name="reset">Whether the result is <c>(observation, info)</c> rather than a 5-tuple.</param>
        /// <returns>The described result, or the exception it raised.</returns>
        private CallResult Call(Func<PyObject> call, bool reset) {
            return PythonRuntime.Run(() => {
                PyObject result;
                try {
                    result = call();
                } catch (PythonException exception) {
                    return CallResult.Threw(new RaisedError(ValueFacts.QualifiedName(exception.Type), exception.Message));
                }

                using (result) {
                    _lastObservation?.Dispose();
                    _lastObservation = result[0];
                    ValueFacts described = ValueFacts.FromPython(result);
                    return CallResult.Of(reset ? Transition.FromReset(described) : Transition.FromStep(described));
                }
            });
        }

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
