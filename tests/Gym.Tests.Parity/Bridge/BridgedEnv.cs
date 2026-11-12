using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Observations;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Utils;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Bridge {
    /// <summary>
    ///     The .NET half of one Gymnasium-side env of the reverse bridge (the interop design's §8): the Gym.NET env that a
    ///     <c>GymNetShim</c> instance (<see cref="ReverseBridge"/>) delegates every call to, with the conversions each call
    ///     needs. Python calls its public methods by their C# names (<c>self._env.Step(action)</c>); tests never call them.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>GIL:</b> pythonnet releases the GIL around a CLR method Python calls, so each member takes it back
    ///     (<see cref="PythonRuntime.Run(Action)"/>) for its conversions only; the Gym.NET call itself runs without it, as the
    ///     plain .NET code it is.
    ///     </para>
    ///     <para>
    ///     <b>Ownership:</b> the env belongs to this object and is disposed with it, by the <see cref="ReverseBridge"/> that
    ///     built it. Results cross as zero-copy NumPy views (<see cref="PyConvert.ToPythonShared"/>), each holding its own
    ///     reference to the memory, so the .NET result is released as soon as it has crossed, and memory Gym.NET returned twice
    ///     is memory Python sees twice. A returned <see cref="PyObject"/> is handed to Python, which takes its own reference;
    ///     the one this object created is released at its next call or at disposal (a one-slot handoff, as NumSharp's codec
    ///     does), never left to the finalizer.
    ///     </para>
    ///     <para>
    ///     <b>Errors and warnings:</b> every call goes through <see cref="BridgeCalls"/>, so what the env throws is raised in
    ///     Python as the type it stands for, and what it warns is warned in Python.
    ///     </para>
    ///     <para>
    ///     <b>Generator:</b> Gymnasium's tools read <c>env.unwrapped._np_random</c> directly (<c>deepcopy</c> it, compare its
    ///     <c>bit_generator.state</c>), so after every call the shim copies this env's PCG64 state into a NumPy generator of
    ///     its own (<see cref="NpRandomState"/>). The mirror is read-only by contract: a draw from it does not move the .NET
    ///     generator.
    ///     </para>
    /// </remarks>
    public sealed class BridgedEnv : IDisposable {
        /// <summary>The Gym.NET env every call reaches.</summary>
        private readonly IEnv _env;

        /// <summary>The env's class, the entry point of the .NET spec <see cref="SetSpec"/> mirrors.</summary>
        private readonly Type _envType;

        /// <summary>The last <see cref="PyObject"/> returned to Python, released at the next call (the handoff slot).</summary>
        private PyObject? _handedOff;

        /// <summary>The generator <see cref="NpRandomState"/> last saw, to tell a replaced generator from an advanced one.</summary>
        private Generator? _generator;

        /// <summary>Counts generator replacements, so the shim makes a new NumPy generator exactly when the env made a new one.</summary>
        private long _generation;

        /// <summary>Set by <see cref="Dispose"/>.</summary>
        private bool _disposed;

        /// <summary>The Python type <see cref="Step"/> returns the reward as.</summary>
        private readonly BridgeReward _reward;

        /// <summary>How <see cref="Reset"/> and <see cref="Step"/> return the observation.</summary>
        private readonly BridgeObservation _observation;

        /// <summary>Takes ownership of <paramref name="env"/>.</summary>
        /// <param name="env">The env; disposed with this object.</param>
        /// <param name="envType">The env's class, named in the spec <see cref="SetSpec"/> mirrors.</param>
        /// <param name="reward">The Python type of the reward <see cref="Step"/> returns: the type Gymnasium's env of that class returns.</param>
        /// <param name="observation">How the observations cross: as Gymnasium's env of that class returns them.</param>
        internal BridgedEnv(IEnv env, Type envType, BridgeReward reward = BridgeReward.Float, BridgeObservation observation = BridgeObservation.Shared) {
            _env = env;
            _envType = envType;
            _reward = reward;
            _observation = observation;
        }

        /// <summary>The Gym.NET env, for a test that inspects it after Gymnasium's tools ran.</summary>
        internal IEnv Env => _env;

        /// <summary>
        ///     <c>env.reset(seed=seed, options=options)</c>: resets the .NET env and returns <c>(obs, info)</c>, the observation as
        ///     a zero-copy view (or as Python ints, <see cref="BridgeObservation.PythonInt"/>).
        /// </summary>
        /// <param name="seed">A Python <c>int</c> or <c>None</c> (also a CLR <see langword="null"/>); disposed here.</param>
        /// <param name="options">A <c>dict</c> or <c>None</c>; disposed here.</param>
        /// <returns>The tuple, handed to Python.</returns>
        /// <exception cref="PythonException">
        ///     Raised in Python as what the env threw (a negative seed's <c>gymnasium.error.Error</c>, an option's
        ///     <c>ValueError</c>), or as <c>gymnasium.error.Error</c> for a seed that is not a Python <c>int</c>, with
        ///     Gymnasium's message.
        /// </exception>
        public PyObject Reset(PyObject? seed, PyObject? options) => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            (BigInteger? clrSeed, Info? clrOptions) = PythonRuntime.Run(() => {
                using (seed)
                using (options)
                    return (SeedOf(seed), OptionsOf(options));
            });
            try {
                ResetResult<object?> result = _env.Reset(clrSeed, clrOptions);
                try {
                    return PythonRuntime.Run(() => {
                        using PyObject observation = CrossObservation(result.Observation);
                        using PyObject info = CrossInfo(result.Info);
                        return HandOff(new PyTuple(new[] { observation, info }));
                    });
                } finally {
                    // The views hold their own references to the memory, so the .NET result goes now.
                    Outcome.Release(result.Observation);
                    Outcome.Release(result.Info);
                }
            } finally {
                Outcome.Release(clrOptions);
            }
        });

        /// <summary>
        ///     <c>env.step(action)</c>: converts the action (<see cref="PyConvert.FromPython"/>: a NumPy scalar becomes a 0-d
        ///     array, an <c>int</c> a <see langword="long"/>), steps the .NET env, and returns
        ///     <c>(obs, reward, terminated, truncated, info)</c> with <c>bool</c> flags, and the observation and the reward as the
        ///     bridge was told Gymnasium's env returns them (<see cref="BridgeObservation"/>; <see cref="BridgeReward"/>: a
        ///     <c>float</c>, an <c>np.float64</c> or an <c>int</c>).
        /// </summary>
        /// <param name="action">The action; disposed here.</param>
        /// <returns>The tuple, handed to Python.</returns>
        /// <exception cref="PythonException">Raised in Python as what the env threw (an invalid action's <c>AssertionError</c>), or as <c>TypeError</c> for an action with no Gym.NET form.</exception>
        public PyObject Step(PyObject? action) => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            object? clrAction = PythonRuntime.Run(() => {
                using (action)
                    return PyConvert.FromPython(action);
            });
            try {
                StepResult<object?> result = _env.Step(clrAction);
                try {
                    return PythonRuntime.Run(() => {
                        using PyObject observation = CrossObservation(result.Observation);
                        // Gym.NET's reward is a float64 double; its Python type is the one Gymnasium's env of this class
                        // returns, which data_equivalence compares (Pendulum's -costs is an np.float64, CartPole's 1.0 a float,
                        // the toy_text tables' -1 an int).
                        using PyObject reward = _reward switch {
                            BridgeReward.NumpyFloat64 => Python.numpy.float64(result.Reward),
                            BridgeReward.Int => new PyInt(checked((long)result.Reward)),
                            // LunarLander's terminal -100/+100 are Python int literals.
                            BridgeReward.NumpyFloat64OrTerminalInt => result.Terminated && Math.Abs(result.Reward) == 100
                                ? new PyInt((long)result.Reward)
                                : Python.numpy.float64(result.Reward),
                            // BipedalWalker's fall is the int literal -100; its other rewards take the float32 actions'
                            // precision. The double is a float32 value widened, so the narrowing is exact.
                            BridgeReward.NumpyFloat32OrFallInt => result.Terminated && result.Reward == -100
                                ? new PyInt(-100)
                                : Python.numpy.float32((float)result.Reward),
                            // CarRacing's score differences are Python floats; leaving the playfield assigns the int -100.
                            BridgeReward.FloatOrOffFieldInt => result.Terminated && result.Reward == -100
                                ? new PyInt(-100)
                                : new PyFloat(result.Reward),
                            _ => new PyFloat(result.Reward),
                        };
                        using PyObject terminated = result.Terminated.ToPython();
                        using PyObject truncated = result.Truncated.ToPython();
                        using PyObject info = CrossInfo(result.Info);
                        return HandOff(new PyTuple(new[] { observation, reward, terminated, truncated, info }));
                    });
                } finally {
                    Outcome.Release(result.Observation);
                    Outcome.Release(result.Info);
                }
            } finally {
                // The converted action is this call's own; the env does not keep it.
                Outcome.Release(clrAction);
            }
        });

        /// <summary>
        ///     The info as Gymnasium's env returns it: a <c>dict</c> in the info's order whose 0-d arrays, at any depth of nested
        ///     dictionaries, cross as the NumPy scalars of their dtypes (<c>np.float64(0.5)</c>, not <c>array(0.5)</c>), and
        ///     everything else as <see cref="PyConvert.ToPythonShared"/> crosses it. Must run under the GIL.
        /// </summary>
        /// <remarks>
        ///     Gym.NET holds a NumPy scalar info value as a 0-d array, NumSharp having no scalar types (DR-08: the MuJoCo envs'
        ///     <c>np.float64</c> reward terms and positions, their <c>np.float32</c> control costs), and
        ///     <c>check_environments_match</c>'s <c>data_equivalence</c> tells <c>np.float64</c> from a 0-d <c>ndarray</c> by
        ///     type, so a view would fail the comparison on a value with the same bits. No built-in env returns a genuine 0-d
        ///     array in an info (the toy_text envs' <c>prob</c> is a Python <c>int</c> or <c>float</c>, Taxi's
        ///     <c>action_mask</c> a 1-d array), so the rule holds for every bridged env, not only the MuJoCo ones.
        /// </remarks>
        /// <param name="info">The .NET info, or a value inside it; stays owned by the caller.</param>
        /// <returns>A new owned reference.</returns>
        /// <exception cref="ArgumentException">Thrown for a value with no defined conversion (<see cref="PyConvert.ToPython"/>'s refusal).</exception>
        /// <exception cref="PythonException">Thrown when NumPy refuses the scalar read.</exception>
        private static PyObject CrossInfo(object? info) {
            switch (info) {
                case NDArray { ndim: 0 } scalar: {
                    // a[()] of a zero-copy view: the scalar copies the value out, so nothing keeps the .NET memory alive.
                    using PyObject view = scalar.ToNumpy();
                    return Python.numpy.ndarray.item_scalar(view);
                }
                case IEnumerable<KeyValuePair<string, object?>> entries: {
                    var dictionary = new PyDict();
                    try {
                        foreach ((string key, object? value) in entries) {
                            using PyObject converted = CrossInfo(value);
                            // The dict takes its own reference to the value.
                            dictionary[key] = converted;
                        }
                    } catch {
                        dictionary.Dispose();
                        throw;
                    }

                    return dictionary;
                }
                default:
                    // Arrays of any other rank stay zero-copy views, so Gymnasium's data-reuse checks still see shared memory.
                    return PyConvert.ToPythonShared(info);
            }
        }

        /// <summary>
        ///     The observation as Gymnasium's env of this class returns it: a zero-copy view (<see cref="BridgeObservation.Shared"/>),
        ///     or with every 0-d integer array, alone or in a tuple, as a Python <c>int</c> (<see cref="BridgeObservation.PythonInt"/>).
        ///     Must run under the GIL.
        /// </summary>
        /// <param name="observation">The .NET observation; still owned by the caller.</param>
        /// <returns>A new owned reference.</returns>
        private PyObject CrossObservation(object? observation) {
            if (_observation == BridgeObservation.Shared)
                return PyConvert.ToPythonShared(observation);
            return AsPythonInts(observation);

            static PyObject AsPythonInts(object? value) {
                switch (value) {
                    case NDArray { ndim: 0 } scalar when scalar.dtype.kind is 'i' or 'u':
                        // int(s): the toy_text envs return Python ints, which data_equivalence tells from NumPy integers.
                        return new PyInt(Convert.ToInt64(scalar.GetAtIndex(0), System.Globalization.CultureInfo.InvariantCulture));
                    case object?[] tuple: {
                        var items = new PyObject[tuple.Length];
                        try {
                            for (int i = 0; i < items.Length; i++)
                                items[i] = AsPythonInts(tuple[i]);
                            return new PyTuple(items);
                        } finally {
                            // PyTuple takes its own references to the items.
                            foreach (PyObject? item in items)
                                item?.Dispose();
                        }
                    }
                    default:
                        return PyConvert.ToPythonShared(value);
                }
            }
        }

        /// <summary>
        ///     <c>env.render()</c>: an <c>rgb_array</c> frame as a zero-copy <c>(H, W, 3)</c> uint8 view, <c>None</c> for human
        ///     mode or no render mode, text for <c>ansi</c>, a list for collected frames, <c>(rgb, depth)</c> for
        ///     <c>rgbd_tuple</c>.
        /// </summary>
        /// <returns>The frame, handed to Python.</returns>
        /// <exception cref="PythonException">Raised in Python as what the env threw while rendering.</exception>
        /// <remarks>
        ///     The bridge owns the render result and releases it once it crossed, which is right for the envs it bridges
        ///     (unwrapped ones, which hand each frame over); a Gym.NET <c>RenderCollection</c> with <c>pop_frames=False</c>,
        ///     whose frames stay its own, is not bridged: Gymnasium's <c>make</c> adds its own wrappers over the shim.
        /// </remarks>
        public PyObject Render() => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            RenderResult frame = _env.Render();
            try {
                return PythonRuntime.Run(() => HandOff(RenderToPython(frame)));
            } finally {
                frame.Dispose();
            }
        });

        /// <summary><c>env.close()</c>: closes the .NET env (twice is allowed, as in Gymnasium); disposal stays with the bridge.</summary>
        /// <exception cref="PythonException">Raised in Python as what the env threw while closing.</exception>
        public void Close() => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            _env.Close();
        });

        /// <summary>The env's <c>metadata</c> as a new <c>dict</c> (<c>render_modes</c> a <c>list</c>, <c>render_fps</c> an <c>int</c>).</summary>
        /// <returns>The dict, handed to Python.</returns>
        /// <exception cref="PythonException">Raised in Python when the metadata holds a value with no Python form.</exception>
        public PyObject Metadata() => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            return PythonRuntime.Run(() => HandOff(PyConvert.ToPython(_env.Metadata)));
        });

        /// <summary>The env's <c>render_mode</c>: a <c>str</c>, or <c>None</c>.</summary>
        /// <returns>The value, handed to Python.</returns>
        /// <exception cref="PythonException">Raised in Python when the bridge is disposed.</exception>
        public PyObject RenderMode() => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            return PythonRuntime.Run(() => HandOff(PyConvert.ToPython(_env.RenderMode)));
        });

        /// <summary>The env's <c>observation_space</c>, mirrored as the Gymnasium space it equals (<see cref="ReverseBridge.Space"/>).</summary>
        /// <returns>The space, handed to Python; it samples from its own Python generator, not the .NET space's.</returns>
        /// <exception cref="PythonException">Raised in Python for a space the bridge does not mirror (as <c>NotSupportedException</c>, which has no Python counterpart).</exception>
        public PyObject ObservationSpace() => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            return PythonRuntime.Run(() => HandOff(ReverseBridge.Space(_env.ObservationSpace)));
        });

        /// <summary>The env's <c>action_space</c>, mirrored as the Gymnasium space it equals.</summary>
        /// <returns>The space, handed to Python; Gymnasium's tools sample actions from it, so they come from a Python generator.</returns>
        /// <exception cref="PythonException">Raised in Python for a space the bridge does not mirror.</exception>
        public PyObject ActionSpace() => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            return PythonRuntime.Run(() => HandOff(ReverseBridge.Space(_env.ActionSpace)));
        });

        /// <summary>
        ///     The env's generator as the shim mirrors it: <c>None</c> while the env has none (Gymnasium's
        ///     <c>_np_random is None</c>), else <c>(generation, seed, state)</c>, where <c>generation</c> changes exactly when the
        ///     env replaced its generator (a seeded reset), <c>seed</c> is its <c>_np_random_seed</c>, and <c>state</c> is the
        ///     PCG64 state in the shape NumPy's <c>bit_generator.state</c> setter takes.
        /// </summary>
        /// <returns>The mirror, handed to Python. Reading it never creates or advances the .NET generator.</returns>
        /// <exception cref="NotSupportedException">
        ///     Thrown for a bit generator other than PCG64, which no Gym.NET env uses; it has no Python counterpart, so Python
        ///     sees the CLR exception.
        /// </exception>
        public PyObject NpRandomState() => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            IEnv unwrapped = _env.Unwrapped;
            // The raw attribute, as Gymnasium's _np_random: the property would create a generator where the env has none.
            var generator = PyAttributes.GetAttr(unwrapped, "_np_random") as Generator;
            if (!ReferenceEquals(generator, _generator)) {
                _generator = generator;
                _generation++;
            }

            if (generator is null)
                return PythonRuntime.Run(() => HandOff(PyConvert.ToPython(null)));
            if (generator.bit_generator is not PCG64 pcg)
                throw new NotSupportedException($"The bridge mirrors PCG64 generators only; this env uses {generator.bit_generator.GetType().Name}.");
            PCG64.State state = pcg.state;
            object? seed = PyAttributes.GetAttr(unwrapped, "_np_random_seed");
            var mirror = new object?[] {
                _generation,
                seed,
                // NumPy's PCG64.state layout, keys in its order. NumSharp keeps the 128-bit words as UInt128, which has no Python
                // conversion of its own; a BigInteger crosses as the Python int NumPy's setter takes.
                new Info {
                    ["bit_generator"] = "PCG64",
                    ["state"] = new Info { ["state"] = (BigInteger)state.state, ["inc"] = (BigInteger)state.inc },
                    ["has_uint32"] = (long)state.has_uint32,
                    ["uinteger"] = (long)state.uinteger,
                },
            };
            return PythonRuntime.Run(() => HandOff(PyConvert.ToPythonShared(mirror)));
        });

        /// <summary>
        ///     Gymnasium's <c>np_random</c> getter on the .NET side: creates the env's generator from OS entropy when it has none
        ///     (the shim then mirrors it), so a Python read of <c>env.np_random</c> never makes a Python-only generator.
        /// </summary>
        /// <exception cref="PythonException">Raised in Python when the bridge is disposed.</exception>
        public void CreateNpRandom() => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            // Reading the property is what creates the generator.
            _ = _env.NpRandom;
        });

        /// <summary>The env's <c>np_random_seed</c>, creating its generator first when it has none, as Gymnasium's property does.</summary>
        /// <returns>The seed as a Python <c>int</c>, handed to Python.</returns>
        /// <exception cref="PythonException">Raised in Python when the bridge is disposed.</exception>
        public PyObject NpRandomSeed() => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            BigInteger seed = _env.NpRandomSeed;
            return PythonRuntime.Run(() => HandOff(PyConvert.ToPython(seed)));
        });

        /// <summary>
        ///     Mirrors the spec Gymnasium assigned the shim (<c>make</c> sets <c>env.unwrapped.spec</c>) onto the .NET env, so
        ///     the env sees the spec a Gymnasium env would: its <c>render()</c> without a render mode warns with the id, and its
        ///     <c>str</c> names it. The entry point becomes the env's own class; the fields convert by
        ///     <see cref="PyConvert.FromPython"/>.
        /// </summary>
        /// <param name="id">The spec's id.</param>
        /// <param name="rewardThreshold">A number or <c>None</c>; disposed here.</param>
        /// <param name="nondeterministic">Whether the env is nondeterministic.</param>
        /// <param name="maxEpisodeSteps">An <c>int</c> or <c>None</c>; disposed here.</param>
        /// <param name="orderEnforce">Whether <c>make</c> adds <c>OrderEnforcing</c>.</param>
        /// <param name="disableEnvChecker">Whether <c>make</c> leaves out the passive checker.</param>
        /// <param name="kwargs">The constructor kwargs, a <c>dict</c>; disposed here.</param>
        /// <exception cref="PythonException">Raised in Python as <c>gymnasium.error.Error</c> for an id .NET's spec refuses, or <c>TypeError</c> for a field with no Gym.NET form.</exception>
        public void SetSpec(string id, PyObject? rewardThreshold, bool nondeterministic, PyObject? maxEpisodeSteps, bool orderEnforce, bool disableEnvChecker,
            PyObject? kwargs) => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            (object? threshold, object? limit, object? arguments) = PythonRuntime.Run(() => {
                using (rewardThreshold)
                using (maxEpisodeSteps)
                using (kwargs)
                    return (PyConvert.FromPython(rewardThreshold), PyConvert.FromPython(maxEpisodeSteps), PyConvert.FromPython(kwargs));
            });
            _env.Spec = new EnvSpec(id, _envType, threshold is null ? null : Convert.ToDouble(threshold, System.Globalization.CultureInfo.InvariantCulture), nondeterministic,
                limit is null ? null : Convert.ToInt64(limit, System.Globalization.CultureInfo.InvariantCulture), orderEnforce, disableEnvChecker, arguments as Info);
        });

        /// <summary>Mirrors <c>env.spec = None</c> onto the .NET env.</summary>
        /// <exception cref="PythonException">Raised in Python when the bridge is disposed.</exception>
        public void ClearSpec() => BridgeCalls.Invoke(() => {
            ThrowIfDisposed();
            _env.Spec = null;
        });

        /// <summary>Releases the handed-off object under the GIL and disposes the env. Safe to call twice.</summary>
        public void Dispose() {
            if (_disposed)
                return;
            _disposed = true;
            if (_handedOff != null) {
                PythonRuntime.Run(() => {
                    _handedOff.Dispose();
                    _handedOff = null;
                });
            }

            _env.Dispose();
        }

        /// <summary>
        ///     Keeps <paramref name="value"/> as the object Python is being handed and releases the previous one, which Python
        ///     received (with its own reference) on an earlier call. Must run under the GIL.
        /// </summary>
        /// <param name="value">The new reference about to be returned.</param>
        /// <returns><paramref name="value"/>.</returns>
        private PyObject HandOff(PyObject value) {
            _handedOff?.Dispose();
            _handedOff = value;
            return value;
        }

        /// <summary>Refuses calls after <see cref="Dispose"/>: the env they would reach is gone.</summary>
        /// <exception cref="ObjectDisposedException">Thrown after disposal (it reaches Python as the CLR exception).</exception>
        private void ThrowIfDisposed() {
            if (_disposed)
                throw new ObjectDisposedException(nameof(BridgedEnv), "The reverse bridge that built this env was disposed.");
        }

        /// <summary>
        ///     A reset seed as Gym.NET takes it: <c>None</c> keeps the generator, a Python <c>int</c> (any size, <c>bool</c>
        ///     included, as Python's <c>isinstance(seed, int)</c> accepts) reseeds. Must run under the GIL.
        /// </summary>
        /// <param name="seed">The Python value, or <see langword="null"/>.</param>
        /// <returns>The seed, or <see langword="null"/>.</returns>
        /// <exception cref="Gym.Exceptions.Error">
        ///     Thrown for anything else with Gymnasium's own message (<c>Seed must be a python integer, actual type: &lt;class
        ///     'float'&gt;</c>): Gymnasium's <c>seeding.np_random</c> refuses it there, and Gym.NET's typed API cannot even take it.
        /// </exception>
        private static BigInteger? SeedOf(PyObject? seed) {
            if (seed is null || seed.IsNone())
                return null;
            using PyObject intType = PythonRuntime.Builtins.GetAttr("int");
            if (!seed.IsInstance(intType))
                throw new Gym.Exceptions.Error($"Seed must be a python integer, actual type: <class '{ValueFacts.PythonTypeName(seed)}'>");
            using var integer = new PyInt(seed);
            return integer.ToBigInteger();
        }

        /// <summary>Reset options as Gym.NET takes them: <c>None</c>, or a <c>dict</c> converted by <see cref="PyConvert.FromPython"/>. Must run under the GIL.</summary>
        /// <param name="options">The Python value, or <see langword="null"/>.</param>
        /// <returns>The options, or <see langword="null"/>; the caller owns the arrays in them.</returns>
        /// <exception cref="TypeError">Thrown for options that are not a <c>dict</c> (or hold a value with no Gym.NET form).</exception>
        private static Info? OptionsOf(PyObject? options) {
            object? converted = PyConvert.FromPython(options);
            if (converted is null || converted is Info)
                return (Info?)converted;
            Outcome.Release(converted);
            throw new TypeError($"Gym.NET's reset options are a dict or None, not {ValueFacts.PythonTypeName(options!)}");
        }

        /// <summary>A render result in Python's form, its arrays as zero-copy views. Must run under the GIL.</summary>
        /// <param name="result">The result; stays owned by the caller.</param>
        /// <returns>A new owned reference.</returns>
        /// <exception cref="NotSupportedException">Thrown for a <see cref="RenderResult"/> case this bridge does not know.</exception>
        private static PyObject RenderToPython(RenderResult result) {
            switch (result) {
                case RenderResult.NoneResult:
                    return PyConvert.ToPython(null);
                case RenderResult.Rgb rgb:
                    return rgb.Frame.ToNumpy();
                case RenderResult.Depth depth:
                    return depth.Frame.ToNumpy();
                case RenderResult.RgbDepth pair: {
                    using PyObject rgbFrame = pair.RgbFrame.ToNumpy();
                    using PyObject depthFrame = pair.DepthFrame.ToNumpy();
                    return new PyTuple(new[] { rgbFrame, depthFrame });
                }
                case RenderResult.Ansi ansi:
                    return new PyString(ansi.Text);
                case RenderResult.Frames frames: {
                    var list = new PyList();
                    try {
                        foreach (RenderResult item in frames.Items) {
                            using PyObject converted = RenderToPython(item);
                            list.Append(converted);
                        }
                    } catch {
                        list.Dispose();
                        throw;
                    }

                    return list;
                }
                default:
                    throw new NotSupportedException($"The bridge has no Python form for {result.GetType().Name}.");
            }
        }
    }

    /// <summary>
    ///     The constructor of one Gymnasium-side env class (<see cref="ReverseBridge.ShimClass"/>): Python's
    ///     <c>GymNetShim.__init__(**kwargs)</c> calls <see cref="Create"/>, which builds the .NET env from the same kwargs, so
    ///     <c>make</c>'s keyword arguments, <c>render_mode</c> included, reach the Gym.NET constructor by Python name.
    /// </summary>
    public sealed class BridgedEnvFactory {
        /// <summary>The env class the shim class stands for.</summary>
        private readonly Type _envType;

        /// <summary>Builds the .NET env from the kwargs (by default <see cref="PyKwargs.Construct"/> on <see cref="_envType"/>).</summary>
        private readonly Func<Info, IEnv> _create;

        /// <summary>Hands each built env to its <see cref="ReverseBridge"/>, which owns and disposes it.</summary>
        private readonly Action<BridgedEnv> _register;

        /// <summary>The Python type of the rewards the built envs return.</summary>
        private readonly BridgeReward _reward;

        /// <summary>How the built envs return their observations.</summary>
        private readonly BridgeObservation _observation;

        /// <summary>A factory for <paramref name="envType"/>.</summary>
        /// <param name="envType">The env class.</param>
        /// <param name="create">How to build the env from the kwargs.</param>
        /// <param name="register">Receives every built env.</param>
        /// <param name="reward">The Python type of the rewards the built envs return.</param>
        /// <param name="observation">How the built envs return their observations.</param>
        internal BridgedEnvFactory(Type envType, Func<Info, IEnv> create, Action<BridgedEnv> register, BridgeReward reward = BridgeReward.Float,
            BridgeObservation observation = BridgeObservation.Shared) {
            _envType = envType;
            _create = create;
            _register = register;
            _reward = reward;
            _observation = observation;
        }

        /// <summary>Builds a .NET env from the shim constructor's kwargs and wraps it for Python.</summary>
        /// <param name="kwargs">The kwargs, a <c>dict</c>; disposed here.</param>
        /// <returns>The bridged env; Python holds it as a CLR object, the <see cref="ReverseBridge"/> disposes it.</returns>
        /// <exception cref="PythonException">
        ///     Raised in Python as what construction threw: Python's <c>TypeError</c> for a keyword the constructor does not take,
        ///     with the text Gymnasium's <c>make</c> then wraps (<c>CartPoleEnv.__init__() got an unexpected keyword argument 'x'</c>).
        /// </exception>
        public BridgedEnv Create(PyObject? kwargs) => BridgeCalls.Invoke(() => {
            Info arguments = PythonRuntime.Run(() => {
                using (kwargs)
                    return PyConvert.FromPython(kwargs) as Info ?? new Info();
            });
            // The env may keep its arguments (an array kwarg becomes its state), so they are not released here.
            IEnv env = _create(arguments);
            var bridged = new BridgedEnv(env, _envType, _reward, _observation);
            _register(bridged);
            return bridged;
        });
    }

    /// <summary>
    ///     The Python type a bridged env returns its reward as. Gym.NET's reward is always a float64 <see langword="double"/>;
    ///     Gymnasium's envs return a Python <c>float</c> (CartPole, MountainCar, Acrobot, Blackjack), a NumPy scalar
    ///     (Pendulum's <c>-costs</c> is an <c>np.float64</c>) or a Python <c>int</c> (the toy_text tables), and
    ///     <c>check_environments_match</c>'s <c>data_equivalence</c> compares the types, so the bridge returns the type
    ///     Gymnasium's env of the same class does.
    /// </summary>
    public enum BridgeReward {
        /// <summary>A Python <c>float</c>.</summary>
        Float,

        /// <summary>A NumPy <c>np.float64</c>.</summary>
        NumpyFloat64,

        /// <summary>A Python <c>int</c>: the toy_text tables' rewards (FrozenLake's schedule, CliffWalking's and Taxi's -1, -10, -100, 20).</summary>
        Int,

        /// <summary>
        ///     A NumPy <c>np.float64</c> (LunarLander's shaping differences), except a terminal -100 or +100, which LunarLander
        ///     assigns as Python <c>int</c> literals.
        /// </summary>
        NumpyFloat64OrTerminalInt,

        /// <summary>
        ///     A NumPy <c>np.float32</c>: BipedalWalker's reward under the float32 actions Gymnasium's tools sample (a Python float
        ///     shaping difference less float32 energy terms is a float32 under NEP 50), except a fall's -100, a Python
        ///     <c>int</c> literal; the far end's finish keeps its float32.
        /// </summary>
        NumpyFloat32OrFallInt,

        /// <summary>
        ///     A Python <c>float</c>: CarRacing's change of score, except the int -100 its step assigns when the car leaves the
        ///     playfield.
        /// </summary>
        FloatOrOffFieldInt,
    }

    /// <summary>
    ///     How a bridged env returns its observations. Gym.NET's discrete observation is a 0-d int64 array (DR-08), where
    ///     Gymnasium's toy_text envs return <c>int(s)</c> (Blackjack a tuple of ints), and <c>check_environments_match</c>'s
    ///     <c>data_equivalence</c> tells a Python <c>int</c> from a NumPy integer, so the bridge returns what Gymnasium's env of
    ///     the same class does.
    /// </summary>
    public enum BridgeObservation {
        /// <summary>Zero-copy NumPy views of the arrays (<see cref="PyConvert.ToPythonShared"/>): the array envs.</summary>
        Shared,

        /// <summary>Every 0-d integer array, alone or in a tuple, as a Python <c>int</c>; anything else as <see cref="Shared"/>.</summary>
        PythonInt,
    }

    /// <summary>
    ///     How every bridge call meets Python's semantics: Gym.NET's warnings are warned in Python, and Gym.NET's exceptions
    ///     are raised in Python as the types they stand for (<see cref="ExceptionMap.PythonTypeFor"/>), with their messages.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Warnings: the call runs inside a <see cref="Warnings.Catch"/> block, and each record is re-issued with Python's
    ///     <c>warnings.warn</c> before the call returns to Python, in order; Python's filters and <c>catch_warnings</c> blocks
    ///     then see exactly what a Gymnasium env's <c>logger.warn</c> would have issued (the same <c>WARN: ...</c> text and
    ///     category), attributed to the shim method that crossed. Warnings issued before a failure are re-issued before the
    ///     failure is raised.
    ///     </para>
    ///     <para>
    ///     Errors: pythonnet restores a <see cref="PythonException"/> thrown from a CLR method as the Python exception it
    ///     carries, so a translated error reaches Python as a real <c>AssertionError</c> or <c>gymnasium.error.Error</c>, which
    ///     Gymnasium's tools catch by type (<c>check_env</c> turns a reset's <c>TypeError</c> into its own message). An
    ///     exception with no Python counterpart passes through as the CLR exception, which pythonnet raises as a CLR type.
    ///     </para>
    /// </remarks>
    internal static class BridgeCalls {
        /// <summary>Runs a bridge call with Python's warning and error semantics.</summary>
        /// <typeparam name="T">The call's result.</typeparam>
        /// <param name="body">The call.</param>
        /// <returns>What <paramref name="body"/> returned.</returns>
        /// <exception cref="PythonException">Thrown with the Python form of what <paramref name="body"/> threw, or of a warning a Python filter turned into an error.</exception>
        /// <exception cref="Exception">Thrown as it is when <paramref name="body"/> threw an exception with no Python counterpart.</exception>
        internal static T Invoke<T>(Func<T> body) {
            WarningCatcher caught = Warnings.Catch();
            try {
                T result = body();
                Forward(caught);
                return result;
            } catch (Exception exception) when (exception is not PythonException) {
                Forward(caught);
                if (Translate(exception) is { } raised)
                    throw raised;
                throw;
            } finally {
                caught.Dispose();
            }
        }

        /// <summary>Runs a bridge call that returns nothing, with Python's warning and error semantics.</summary>
        /// <param name="body">The call.</param>
        /// <exception cref="PythonException">Thrown with the Python form of what <paramref name="body"/> threw, or of a warning a Python filter turned into an error.</exception>
        /// <exception cref="Exception">Thrown as it is when <paramref name="body"/> threw an exception with no Python counterpart.</exception>
        internal static void Invoke(Action body) {
            Invoke<object?>(() => {
                body();
                return null;
            });
        }

        /// <summary>Re-issues the caught Gym.NET warnings in Python, in order. Takes the GIL itself.</summary>
        /// <param name="caught">The catcher of the call.</param>
        /// <exception cref="PythonException">Thrown when a Python filter turns a warning into an error.</exception>
        private static void Forward(WarningCatcher caught) {
            IReadOnlyList<WarningMessage> records = caught.Records;
            if (records.Count == 0)
                return;
            PythonRuntime.Run(() => {
                foreach (WarningMessage record in records) {
                    using PyObject category = record.Category == WarningCategory.DeprecationWarning
                        ? Python.builtins.DeprecationWarning()
                        : Python.builtins.UserWarning();
                    Python.warnings.warn(record.Message, category);
                }
            });
        }

        /// <summary>
        ///     The Python exception <paramref name="exception"/> stands for: an instance of the mapped type built from the .NET
        ///     message, wrapped so pythonnet restores it, with the .NET exception attached as the inner one for diagnostics.
        ///     Takes the GIL itself.
        /// </summary>
        /// <param name="exception">The .NET exception.</param>
        /// <returns>The Python exception, or <see langword="null"/> when no Python type stands for it (or Gymnasium lacks the mapped <c>gymnasium.error</c> class).</returns>
        private static PythonException? Translate(Exception exception) {
            string? name = ExceptionMap.PythonTypeFor(exception);
            if (name is null)
                return null;
            return PythonRuntime.Run(() => {
                int dot = name.LastIndexOf('.');
                // A runtime-chosen type (the skill's raw call site): builtins for built-in names, the named module otherwise.
                PyObject owner = dot < 0 ? PythonRuntime.Builtins : PythonRuntime.Import(name.Substring(0, dot));
                string typeName = name.Substring(dot + 1);
                if (!owner.HasAttr(typeName))
                    return null;
                using PyObject type = owner.GetAttr(typeName);
                using var message = new PyString(exception.Message);
                // The exception keeps the instance (and its own reference to the type) until pythonnet restores it.
                PyObject instance = type.Invoke(message);
                return new PythonException(new PyType(type), instance, null, exception);
            });
        }
    }
}
