using System;
using System.Collections.Generic;
using Gym.Tests.Parity.Interop;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

// The tree's type names are Python's module and class names (numpy, gymnasium, spaces, box, ...), lower-case on purpose so
// call sites read like the Python they drive; CS8981's "may become reserved" note is future-proofing this file accepts, as
// NumSharp.Core does for np, linalg and fft.
#pragma warning disable CS8981

namespace Gym.Tests.Parity {
    /// <summary>
    ///     The <b>pythonic</b> surface of the parity suite (the pythonic skill, as NumSharp and OptunaSharp spell it): a root
    ///     <c>static class Python</c> whose nested classes form a tree reaching each FIXED Python call the suite makes, so the
    ///     C# reads like the Python it drives: <c>Python.gymnasium.make("CartPole-v1")</c>,
    ///     <c>Python.gymnasium.Env.step(env, action)</c>, <c>Python.gymnasium.spaces.Discrete(4, seed: 0)</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It is always spelled <c>Python.*</c>: inside <c>Gym.Tests.Parity</c> the class shadows pythonnet's
    ///     <c>Python.Runtime</c> namespace, so pythonnet types are reached through the file-level <c>using Python.Runtime;</c>,
    ///     never through a <c>Python.Runtime.X</c> qualifier.
    ///     </para>
    ///     <para>
    ///     A Python object's members are nested classes that take the object, mirroring Python's unbound spelling
    ///     (<c>Space.sample(space)</c>). Where a class is both called (its constructor) and used through its members, and C#
    ///     cannot give a method and a nested class the same name, the constructor is reached through the package that
    ///     re-exports the class and the members through the module that defines it, both real Python paths:
    ///     <c>gymnasium.spaces.Discrete(4)</c> and <c>gymnasium.spaces.discrete.Discrete.n(space)</c>;
    ///     <c>pygame.Vector2(x, y)</c> and <c>pygame.math.Vector2.rotate_rad(v, a)</c>.
    ///     </para>
    ///     <para>
    ///     Every member calls the C-API and must run under the GIL (inside <see cref="PythonRuntime.Run(Action)"/>). A
    ///     returned <see cref="PyObject"/> is a new reference owned by the caller: dispose it under the GIL. Arguments stay
    ///     owned by their callers. Module handles come from <see cref="PythonRuntime.Import"/>'s cache and are never disposed
    ///     here. Arrays cross through NumSharp's interop (<c>ToNumpyCopy</c>), never element by element. Calls chosen at run
    ///     time (type reflection, generic <c>dict.items()</c> walks) stay in <see cref="Oracle.ValueFacts"/>, the suite's
    ///     value marshaling, as the skill prescribes.
    ///     </para>
    /// </remarks>
    internal static class Python {
        /// <summary>
        ///     The snippet runner (the skill's flavor B): executes a multi-statement Python body in a throwaway scope and
        ///     returns one variable, for code that must read 1:1 like Python rather than as a chain of facade calls.
        /// </summary>
        /// <param name="setup">The statements.</param>
        /// <param name="resultVar">The variable (or expression) to return once they ran.</param>
        /// <param name="names">Values bound in the scope before the body runs; they stay owned by the caller.</param>
        /// <returns>A new owned reference; dispose it under the GIL.</returns>
        /// <exception cref="PythonException">Thrown when the body raises.</exception>
        internal static PyObject evalStmts(string setup, string resultVar, params (string Name, PyObject Value)[] names) {
            using PyModule scope = Py.CreateScope();
            foreach ((string name, PyObject value) in names)
                scope.Set(name, value);
            scope.Exec(setup);
            return scope.Eval(resultVar);
        }

        /// <summary><c>import sys</c>.</summary>
        internal static class sys {
            /// <summary>The cached module; never dispose it.</summary>
            private static PyObject Module => PythonRuntime.Import("sys");

            /// <summary><c>sys.dont_write_bytecode</c>: whether imports skip writing <c>__pycache__</c>.</summary>
            /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
            internal static bool dont_write_bytecode {
                get {
                    using PyObject value = Module.GetAttr("dont_write_bytecode");
                    return value.IsTrue();
                }
            }

            /// <summary><c>sys.flags</c>.</summary>
            internal static class flags {
                /// <summary><c>sys.flags.hash_randomization</c>: 0 when <c>PYTHONHASHSEED=0</c> reached the interpreter's start.</summary>
                /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                internal static long hash_randomization {
                    get {
                        using PyObject flagsObject = Module.GetAttr("flags");
                        using PyObject value = flagsObject.GetAttr("hash_randomization");
                        return value.As<long>();
                    }
                }
            }
        }

        /// <summary><c>import os</c>.</summary>
        internal static class os {
            /// <summary><c>os.environ</c>.</summary>
            internal static class environ {
                /// <summary>
                ///     <c>os.environ.get(key)</c>: the variable as Python sees it, which is the C runtime's copy of the
                ///     environment, not the Win32 block <see cref="Environment.SetEnvironmentVariable(string, string)"/> writes.
                /// </summary>
                /// <param name="key">The variable's name.</param>
                /// <returns>A new owned reference: the value as a <c>str</c>, or <c>None</c> when unset.</returns>
                /// <exception cref="PythonException">Thrown when the lookup fails.</exception>
                internal static PyObject get(string key) {
                    using PyObject environment = PythonRuntime.Import("os").GetAttr("environ");
                    using PyObject getter = environment.GetAttr("get");
                    using var name = new PyString(key);
                    return getter.Invoke(name);
                }
            }
        }

        /// <summary><c>import platform</c>.</summary>
        internal static class platform {
            /// <summary>
            ///     <c>platform.python_version()</c> as a CLR string (the skill's <c>_t</c> typed variant: the value only ever
            ///     leaves Python, for the stack report), for example <c>3.12.12</c>.
            /// </summary>
            /// <returns>The version string.</returns>
            /// <exception cref="PythonException">Thrown when the call fails.</exception>
            internal static string python_version_t() {
                using PyObject function = PythonRuntime.Import("platform").GetAttr("python_version");
                using PyObject version = function.Invoke();
                return version.As<string>();
            }
        }

        /// <summary><c>import gc</c>.</summary>
        internal static class gc {
            /// <summary><c>gc.collect()</c>: a full Python collection, part of the leak gate's release turn.</summary>
            /// <exception cref="PythonException">Thrown when the call fails.</exception>
            internal static void collect() {
                using PyObject function = PythonRuntime.Import("gc").GetAttr("collect");
                using PyObject _ = function.Invoke();
            }
        }

        /// <summary>Python's built-in functions.</summary>
        internal static class builtins {
            /// <summary><c>int(x)</c>: Python's conversion to an integer, raising <c>ValueError</c> for text that is not one.</summary>
            /// <param name="x">The value; stays owned by the caller.</param>
            /// <returns>A new owned reference.</returns>
            /// <exception cref="PythonException">Thrown when <c>int()</c> raises.</exception>
            internal static PyObject @int(PyObject x) {
                using PyObject function = PythonRuntime.Builtins.GetAttr("int");
                return function.Invoke(x);
            }

            /// <summary><c>repr(x)</c>: what Python prints for the object, which Gym.NET's <c>ToString()</c> reproduces.</summary>
            /// <param name="x">The value; stays owned by the caller.</param>
            /// <returns>A new owned reference to the <c>str</c>.</returns>
            /// <exception cref="PythonException">Thrown when <c>repr()</c> raises.</exception>
            internal static PyObject repr(PyObject x) {
                using PyObject function = PythonRuntime.Builtins.GetAttr("repr");
                return function.Invoke(x);
            }
        }

        /// <summary><c>import numpy as np</c>: the few NumPy calls the suite makes directly (arrays cross through NumSharp's interop).</summary>
        internal static class numpy {
            /// <summary>The cached module; never dispose it.</summary>
            private static PyObject Module => PythonRuntime.Import("numpy");

            /// <summary><c>np.__version__</c>.</summary>
            /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
            internal static string __version__ {
                get {
                    using PyObject value = Module.GetAttr("__version__");
                    return value.As<string>();
                }
            }

            /// <summary><c>np.__file__</c>: where NumPy was imported from, which tells a venv's copy from the base installation's.</summary>
            /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
            internal static string __file__ {
                get {
                    using PyObject value = Module.GetAttr("__file__");
                    return value.As<string>();
                }
            }

            /// <summary><c>np.dtype(name)</c>, for dtype arguments (<c>spaces.Box(..., dtype=np.dtype("float32"))</c> equals <c>dtype=np.float32</c>).</summary>
            /// <param name="name">A NumPy dtype name, for example <c>float32</c>.</param>
            /// <returns>A new owned reference.</returns>
            /// <exception cref="PythonException">Thrown for an unknown name.</exception>
            internal static PyObject dtype(string name) {
                using PyObject function = Module.GetAttr("dtype");
                using var text = new PyString(name);
                return function.Invoke(text);
            }

            /// <summary><c>np.int8(value)</c>.</summary>
            /// <param name="value">The value.</param>
            /// <returns>A new owned reference to the NumPy scalar.</returns>
            /// <exception cref="PythonException">Thrown when the call fails.</exception>
            internal static PyObject int8(sbyte value) => Scalar("int8", new PyInt(value));

            /// <summary><c>np.int32(value)</c>.</summary>
            /// <param name="value">The value.</param>
            /// <returns>A new owned reference to the NumPy scalar.</returns>
            /// <exception cref="PythonException">Thrown when the call fails.</exception>
            internal static PyObject int32(int value) => Scalar("int32", new PyInt(value));

            /// <summary><c>np.int64(value)</c>.</summary>
            /// <param name="value">The value.</param>
            /// <returns>A new owned reference to the NumPy scalar.</returns>
            /// <exception cref="PythonException">Thrown when the call fails.</exception>
            internal static PyObject int64(long value) => Scalar("int64", new PyInt(value));

            /// <summary><c>np.uint8(value)</c>.</summary>
            /// <param name="value">The value.</param>
            /// <returns>A new owned reference to the NumPy scalar.</returns>
            /// <exception cref="PythonException">Thrown when the call fails.</exception>
            internal static PyObject uint8(byte value) => Scalar("uint8", new PyInt(value));

            /// <summary><c>np.uint64(value)</c>.</summary>
            /// <param name="value">The value.</param>
            /// <returns>A new owned reference to the NumPy scalar.</returns>
            /// <exception cref="PythonException">Thrown when the call fails.</exception>
            internal static PyObject uint64(ulong value) => Scalar("uint64", new PyInt(value));

            /// <summary><c>np.float32(value)</c>, from the float32 value itself (no rounding on the way).</summary>
            /// <param name="value">The value.</param>
            /// <returns>A new owned reference to the NumPy scalar.</returns>
            /// <exception cref="PythonException">Thrown when the call fails.</exception>
            internal static PyObject float32(float value) => Scalar("float32", new PyFloat(value));

            /// <summary><c>np.float64(value)</c>.</summary>
            /// <param name="value">The value.</param>
            /// <returns>A new owned reference to the NumPy scalar.</returns>
            /// <exception cref="PythonException">Thrown when the call fails.</exception>
            internal static PyObject float64(double value) => Scalar("float64", new PyFloat(value));

            /// <summary><c>np.atleast_1d(x)</c>.</summary>
            /// <param name="x">The value; stays owned by the caller.</param>
            /// <returns>A new owned reference.</returns>
            /// <exception cref="PythonException">Thrown when the call fails.</exception>
            internal static PyObject atleast_1d(PyObject x) {
                using PyObject function = Module.GetAttr("atleast_1d");
                return function.Invoke(x);
            }

            /// <summary>Calls a NumPy scalar type on a Python number and releases the number.</summary>
            /// <param name="type">The scalar type's name.</param>
            /// <param name="value">The Python number; disposed here.</param>
            /// <returns>A new owned reference to the scalar.</returns>
            private static PyObject Scalar(string type, PyObject value) {
                using (value) {
                    using PyObject constructor = Module.GetAttr(type);
                    return constructor.Invoke(value);
                }
            }

            /// <summary>
            ///     <c>np.ndarray</c> instance members: NumPy's own reading of an array, which the session cells hold against
            ///     NumSharp's import (<see cref="Oracle.ValueFacts.FromPython"/>) so the suite's byte reader is verified, not assumed.
            /// </summary>
            internal static class ndarray {
                /// <summary><c>a.dtype</c>.</summary>
                /// <param name="a">The array.</param>
                /// <returns>A new owned reference to the <c>numpy.dtype</c>.</returns>
                /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                internal static PyObject dtype(PyObject a) => a.GetAttr("dtype");

                /// <summary><c>a.shape</c>.</summary>
                /// <param name="a">The array.</param>
                /// <returns>A new owned reference to the shape tuple.</returns>
                /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                internal static PyObject shape(PyObject a) => a.GetAttr("shape");

                /// <summary>
                ///     <c>a.tobytes()</c> as a CLR array (the <c>_t</c> typed variant): the elements in C order, whatever the
                ///     array's strides, converted by pythonnet rather than NumSharp. pythonnet's conversion is slow for large
                ///     buffers (about 84 ms for a 600 × 400 × 3 frame), so keep it to the session's small probes.
                /// </summary>
                /// <param name="a">The array.</param>
                /// <returns>The bytes.</returns>
                /// <exception cref="PythonException">Thrown when the call fails.</exception>
                internal static byte[] tobytes_t(PyObject a) {
                    using PyObject method = a.GetAttr("tobytes");
                    using PyObject bytes = method.Invoke();
                    return bytes.As<byte[]>();
                }
            }

            /// <summary><c>np.random</c>.</summary>
            internal static class random {
                /// <summary><c>np.random.Generator</c> instance members (unbound: they take the generator).</summary>
                internal static class Generator {
                    /// <summary><c>g.uniform(low, high, size)</c>.</summary>
                    /// <param name="g">The generator.</param>
                    /// <param name="low">The lower bound.</param>
                    /// <param name="high">The upper bound.</param>
                    /// <param name="size">The shape, or <see langword="null"/> for a scalar draw.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when the call fails.</exception>
                    internal static PyObject uniform(PyObject g, double low, double high, long[]? size = null) {
                        using var lo = new PyFloat(low);
                        using var hi = new PyFloat(high);
                        return Call(g, "uniform", new PyObject[] { lo, hi }, ("size", size));
                    }

                    /// <summary><c>g.integers(low, high, size, dtype)</c>.</summary>
                    /// <param name="g">The generator.</param>
                    /// <param name="low">The lower bound, or the exclusive upper bound when <paramref name="high"/> is <see langword="null"/>.</param>
                    /// <param name="high">The exclusive upper bound, or <see langword="null"/>.</param>
                    /// <param name="size">The shape, or <see langword="null"/> for a scalar draw.</param>
                    /// <param name="dtype">The dtype, or <see langword="null"/> for NumPy's default (int64).</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when the call fails.</exception>
                    internal static PyObject integers(PyObject g, long low, long? high = null, long[]? size = null, PyObject? dtype = null) {
                        using var lo = new PyInt(low);
                        using PyObject? hi = high is { } h ? new PyInt(h) : null;
                        var positional = hi is null ? new PyObject[] { lo } : new PyObject[] { lo, hi };
                        return Call(g, "integers", positional, ("size", size), ("dtype", dtype));
                    }

                    /// <summary><c>g.choice(a, size, p=p)</c> over an array.</summary>
                    /// <param name="g">The generator.</param>
                    /// <param name="a">The population; stays owned by the caller.</param>
                    /// <param name="size">The shape, or <see langword="null"/> for one draw.</param>
                    /// <param name="p">The probabilities, or <see langword="null"/> for uniform draws; stays owned by the caller.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when the call fails.</exception>
                    internal static PyObject choice(PyObject g, PyObject a, long[]? size = null, PyObject? p = null) {
                        return Call(g, "choice", new[] { a }, ("size", size), ("p", p));
                    }

                    /// <summary><c>g.choice(n, size, p=p)</c> over <c>range(n)</c>.</summary>
                    /// <param name="g">The generator.</param>
                    /// <param name="a">The population size.</param>
                    /// <param name="size">The shape, or <see langword="null"/> for one draw.</param>
                    /// <param name="p">The probabilities, or <see langword="null"/>; stays owned by the caller.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when the call fails.</exception>
                    internal static PyObject choice(PyObject g, long a, long[]? size = null, PyObject? p = null) {
                        using var n = new PyInt(a);
                        return Call(g, "choice", new PyObject[] { n }, ("size", size), ("p", p));
                    }

                    /// <summary><c>g.normal(size=size)</c>: standard normal draws.</summary>
                    /// <param name="g">The generator.</param>
                    /// <param name="size">The shape.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when the call fails.</exception>
                    internal static PyObject normal(PyObject g, long[] size) => Call(g, "normal", Array.Empty<PyObject>(), ("size", size));

                    /// <summary><c>g.exponential(size=size)</c>.</summary>
                    /// <param name="g">The generator.</param>
                    /// <param name="size">The shape.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when the call fails.</exception>
                    internal static PyObject exponential(PyObject g, long[] size) => Call(g, "exponential", Array.Empty<PyObject>(), ("size", size));

                    /// <summary><c>g.random(size)</c>: uniform draws in [0, 1).</summary>
                    /// <param name="g">The generator.</param>
                    /// <param name="size">The shape.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when the call fails.</exception>
                    internal static PyObject random(PyObject g, long[] size) => Call(g, "random", Array.Empty<PyObject>(), ("size", size));

                    /// <summary><c>g.standard_normal(size)</c>.</summary>
                    /// <param name="g">The generator.</param>
                    /// <param name="size">The shape.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when the call fails.</exception>
                    internal static PyObject standard_normal(PyObject g, long[] size) => Call(g, "standard_normal", Array.Empty<PyObject>(), ("size", size));

                    /// <summary><c>g.bit_generator</c>.</summary>
                    /// <param name="g">The generator.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                    internal static PyObject bit_generator(PyObject g) => g.GetAttr("bit_generator");

                    /// <summary>
                    ///     Calls a generator method with positional arguments and the keyword arguments that are set: a
                    ///     <see langword="long"/> array becomes a shape tuple, a <see cref="PyObject"/> is passed as is, and
                    ///     <see langword="null"/> is left out so NumPy applies its own default.
                    /// </summary>
                    /// <param name="g">The generator.</param>
                    /// <param name="method">The method's name.</param>
                    /// <param name="positional">The positional arguments; they stay owned by the caller.</param>
                    /// <param name="keywords">The keyword arguments.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="ArgumentException">Thrown for a keyword value of another type.</exception>
                    private static PyObject Call(PyObject g, string method, PyObject[] positional, params (string Name, object? Value)[] keywords) {
                        using PyObject bound = g.GetAttr(method);
                        using var arguments = new PyDict();
                        foreach ((string name, object? value) in keywords) {
                            switch (value) {
                                case null:
                                    break;
                                case long[] shape: {
                                    using PyObject tuple = Oracle.PyConvert.ToTuple(shape);
                                    arguments[name] = tuple;
                                    break;
                                }
                                case PyObject python:
                                    arguments[name] = python;
                                    break;
                                default:
                                    throw new ArgumentException($"Keyword '{name}' has an unsupported value type {value.GetType().Name}.", nameof(keywords));
                            }
                        }

                        return bound.Invoke(positional, arguments);
                    }
                }

                /// <summary><c>np.random.BitGenerator</c> instance members.</summary>
                internal static class BitGenerator {
                    /// <summary><c>bit_generator.state</c>: for PCG64, <c>{'bit_generator', 'state': {'state', 'inc'}, 'has_uint32', 'uinteger'}</c>.</summary>
                    /// <param name="bitGenerator">The bit generator.</param>
                    /// <returns>A new owned reference to the state dictionary.</returns>
                    /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                    internal static PyObject state(PyObject bitGenerator) => bitGenerator.GetAttr("state");
                }
            }
        }

        /// <summary><c>import gymnasium</c>, from <c>refs/Gymnasium</c>.</summary>
        internal static class gymnasium {
            /// <summary>The cached module; never dispose it.</summary>
            private static PyObject Module => PythonRuntime.Import("gymnasium");

            /// <summary><c>gymnasium.__version__</c>.</summary>
            /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
            internal static string __version__ {
                get {
                    using PyObject value = Module.GetAttr("__version__");
                    return value.As<string>();
                }
            }

            /// <summary><c>gymnasium.__file__</c>: must lie in <c>refs/Gymnasium</c>.</summary>
            /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
            internal static string __file__ {
                get {
                    using PyObject value = Module.GetAttr("__file__");
                    return value.As<string>();
                }
            }

            /// <summary><c>gymnasium.make(id, render_mode=render_mode, **kwargs)</c>.</summary>
            /// <param name="id">A registered id, for example <c>CartPole-v1</c>.</param>
            /// <param name="render_mode"><c>rgb_array</c>, <c>human</c>, or <see langword="null"/>.</param>
            /// <param name="kwargs">Constructor keyword arguments in Python spelling; values converted by <see cref="Oracle.PyConvert.ToPython"/>.</param>
            /// <returns>A new owned reference to the made env (with its wrapper stack).</returns>
            /// <exception cref="PythonException">Thrown when <c>make</c> raises.</exception>
            internal static PyObject make(string id, string? render_mode = null, IReadOnlyDictionary<string, object?>? kwargs = null) {
                using PyObject function = Module.GetAttr("make");
                using var name = new PyString(id);
                using var arguments = Oracle.PyConvert.ToKwargs(kwargs);
                using PyObject mode = Oracle.PyConvert.ToPython(render_mode);
                arguments["render_mode"] = mode;
                return function.Invoke(new PyObject[] { name }, arguments);
            }

            /// <summary><c>gymnasium.Env</c> instance members (unbound: they take the env).</summary>
            internal static class Env {
                /// <summary><c>env.reset(seed=seed, options=options)</c>.</summary>
                /// <param name="env">The env.</param>
                /// <param name="seed">The seed, or <see langword="null"/> to keep the generator.</param>
                /// <param name="options">Reset options in Python spelling (<c>low</c>, <c>high</c>), or <see langword="null"/>.</param>
                /// <returns>A new owned reference to <c>(observation, info)</c>.</returns>
                /// <exception cref="PythonException">Thrown when the reset raises.</exception>
                internal static PyObject reset(PyObject env, long? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
                    using PyObject method = env.GetAttr("reset");
                    using var arguments = new PyDict();
                    using PyObject seedValue = Oracle.PyConvert.ToPython(seed);
                    arguments["seed"] = seedValue;
                    using PyObject optionsValue = options is null ? Oracle.PyConvert.ToPython(null) : (PyObject)Oracle.PyConvert.ToKwargs(options);
                    arguments["options"] = optionsValue;
                    return method.Invoke(Array.Empty<PyObject>(), arguments);
                }

                /// <summary><c>env.step(action)</c>.</summary>
                /// <param name="env">The env.</param>
                /// <param name="action">The action; stays owned by the caller.</param>
                /// <returns>A new owned reference to <c>(observation, reward, terminated, truncated, info)</c>.</returns>
                /// <exception cref="PythonException">Thrown when the step raises.</exception>
                internal static PyObject step(PyObject env, PyObject action) {
                    using PyObject method = env.GetAttr("step");
                    return method.Invoke(action);
                }

                /// <summary><c>env.render()</c>.</summary>
                /// <param name="env">The env.</param>
                /// <returns>A new owned reference: an <c>(H, W, 3)</c> uint8 array for <c>render_mode="rgb_array"</c>.</returns>
                /// <exception cref="PythonException">Thrown when rendering raises.</exception>
                internal static PyObject render(PyObject env) {
                    using PyObject method = env.GetAttr("render");
                    return method.Invoke();
                }

                /// <summary><c>env.close()</c>.</summary>
                /// <param name="env">The env.</param>
                /// <exception cref="PythonException">Thrown when closing raises.</exception>
                internal static void close(PyObject env) {
                    using PyObject method = env.GetAttr("close");
                    using PyObject _ = method.Invoke();
                }

                /// <summary><c>env.unwrapped</c>.</summary>
                /// <param name="env">The env.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                internal static PyObject unwrapped(PyObject env) => env.GetAttr("unwrapped");

                /// <summary><c>env.action_space</c>.</summary>
                /// <param name="env">The env.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                internal static PyObject action_space(PyObject env) => env.GetAttr("action_space");

                /// <summary><c>env.observation_space</c>.</summary>
                /// <param name="env">The env.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                internal static PyObject observation_space(PyObject env) => env.GetAttr("observation_space");

                /// <summary><c>env.metadata</c>: <c>render_modes</c>, <c>render_fps</c>, ... in Python spelling.</summary>
                /// <param name="env">The env.</param>
                /// <returns>A new owned reference to the dict.</returns>
                /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                internal static PyObject metadata(PyObject env) => env.GetAttr("metadata");

                /// <summary><c>env.np_random</c>. Reading it creates an entropy-seeded generator when the env has none yet.</summary>
                /// <param name="env">The env.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                internal static PyObject np_random(PyObject env) => env.GetAttr("np_random");
            }

            /// <summary><c>gymnasium.spaces</c>.</summary>
            internal static class spaces {
                /// <summary>The cached module; never dispose it.</summary>
                private static PyObject Module => PythonRuntime.Import("gymnasium.spaces");

                /// <summary><c>spaces.Discrete(n, seed=seed, start=start)</c>.</summary>
                /// <param name="n">The number of elements.</param>
                /// <param name="seed">The space's seed, or <see langword="null"/>.</param>
                /// <param name="start">The smallest element.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises (for example <c>n &lt;= 0</c>).</exception>
                internal static PyObject Discrete(long n, long? seed = null, long start = 0) {
                    using PyObject constructor = Module.GetAttr("Discrete");
                    using var count = new PyInt(n);
                    using var arguments = new PyDict();
                    using PyObject seedValue = Oracle.PyConvert.ToPython(seed);
                    using var startValue = new PyInt(start);
                    arguments["seed"] = seedValue;
                    arguments["start"] = startValue;
                    return constructor.Invoke(new PyObject[] { count }, arguments);
                }

                /// <summary><c>spaces.Box(low, high, shape, dtype, seed=seed)</c> with scalar bounds.</summary>
                /// <param name="low">
                ///     The lower bound of every element, converted by <see cref="Oracle.PyConvert.ToPython"/>: a
                ///     <see langword="double"/> becomes a Python <c>float</c> (<c>-inf</c> for unbounded) and a
                ///     <see langword="long"/> a Python <c>int</c>, so <c>Box(0, 5, (8,), np.int32)</c> receives exactly the
                ///     literals a Gymnasium user writes (Box classifies int and float bounds by type).
                /// </param>
                /// <param name="high">The upper bound of every element, converted the same way.</param>
                /// <param name="shape">The shape.</param>
                /// <param name="dtype">The dtype, from <see cref="numpy.dtype"/>; stays owned by the caller.</param>
                /// <param name="seed">The space's seed, or <see langword="null"/>.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
                /// <exception cref="ArgumentException">Thrown for a bound of a type <see cref="Oracle.PyConvert.ToPython"/> does not convert.</exception>
                internal static PyObject Box(object low, object high, long[] shape, PyObject dtype, long? seed = null) {
                    using PyObject lo = Oracle.PyConvert.ToPython(low);
                    using PyObject hi = Oracle.PyConvert.ToPython(high);
                    using PyObject dims = Oracle.PyConvert.ToTuple(shape);
                    return BoxCall(new PyObject[] { lo, hi, dims, dtype }, seed);
                }

                /// <summary><c>spaces.Box(low, high, dtype=dtype, seed=seed)</c> with array bounds (NumPy arrays, from NumSharp's interop).</summary>
                /// <param name="low">The lower bounds; stays owned by the caller.</param>
                /// <param name="high">The upper bounds; stays owned by the caller.</param>
                /// <param name="dtype">The dtype; stays owned by the caller.</param>
                /// <param name="seed">The space's seed, or <see langword="null"/>.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
                internal static PyObject Box(PyObject low, PyObject high, PyObject dtype, long? seed = null) {
                    return BoxCall(new[] { low, high }, seed, dtype);
                }

                /// <summary>Invokes <c>spaces.Box</c> with the given positional arguments and the keyword ones.</summary>
                /// <param name="positional">The positional arguments; they stay owned by the caller.</param>
                /// <param name="seed">The seed keyword.</param>
                /// <param name="dtype">The dtype keyword, when it is not positional.</param>
                /// <returns>A new owned reference.</returns>
                private static PyObject BoxCall(PyObject[] positional, long? seed, PyObject? dtype = null) {
                    using PyObject constructor = Module.GetAttr("Box");
                    using var arguments = new PyDict();
                    using PyObject seedValue = Oracle.PyConvert.ToPython(seed);
                    arguments["seed"] = seedValue;
                    if (dtype != null)
                        arguments["dtype"] = dtype;
                    return constructor.Invoke(positional, arguments);
                }

                /// <summary><c>gymnasium.spaces.space</c>, where the <c>Space</c> base class is defined.</summary>
                internal static class space {
                    /// <summary><c>Space</c> protocol members, called on any space (Python dispatches them to the subclass).</summary>
                    internal static class Space {
                        /// <summary><c>space.sample()</c>, or <c>space.sample(mask=mask)</c>.</summary>
                        /// <param name="space">The space.</param>
                        /// <param name="mask">The mask, or <see langword="null"/>; stays owned by the caller.</param>
                        /// <returns>A new owned reference.</returns>
                        /// <exception cref="PythonException">Thrown when sampling raises.</exception>
                        internal static PyObject sample(PyObject space, PyObject? mask = null) {
                            using PyObject method = space.GetAttr("sample");
                            if (mask is null)
                                return method.Invoke();
                            using var arguments = new PyDict();
                            arguments["mask"] = mask;
                            return method.Invoke(Array.Empty<PyObject>(), arguments);
                        }

                        /// <summary><c>space.contains(x)</c>.</summary>
                        /// <param name="space">The space.</param>
                        /// <param name="x">The candidate; stays owned by the caller.</param>
                        /// <returns>A new owned reference to the boolean.</returns>
                        /// <exception cref="PythonException">Thrown when the check raises.</exception>
                        internal static PyObject contains(PyObject space, PyObject x) {
                            using PyObject method = space.GetAttr("contains");
                            return method.Invoke(x);
                        }

                        /// <summary><c>space.seed(seed)</c>.</summary>
                        /// <param name="space">The space.</param>
                        /// <param name="seed">The seed.</param>
                        /// <returns>A new owned reference to the seeds used.</returns>
                        /// <exception cref="PythonException">Thrown when seeding raises.</exception>
                        internal static PyObject seed(PyObject space, long seed) {
                            using PyObject method = space.GetAttr("seed");
                            using var value = new PyInt(seed);
                            return method.Invoke(value);
                        }

                        /// <summary><c>space.np_random</c>.</summary>
                        /// <param name="space">The space.</param>
                        /// <returns>A new owned reference.</returns>
                        /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                        internal static PyObject np_random(PyObject space) => space.GetAttr("np_random");

                        /// <summary><c>space.shape</c>.</summary>
                        /// <param name="space">The space.</param>
                        /// <returns>A new owned reference to the shape tuple.</returns>
                        /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                        internal static PyObject shape(PyObject space) => space.GetAttr("shape");

                        /// <summary>
                        ///     <c>space.dtype</c>, a <c>numpy.dtype</c> (or <c>None</c> for spaces without one).
                        ///     <see cref="Oracle.ValueFacts.FromPython"/> describes a dtype by its array-interface string
                        ///     (<c>&lt;f4</c>), which is what NumSharp's <c>DType.str</c> spells.
                        /// </summary>
                        /// <param name="space">The space.</param>
                        /// <returns>A new owned reference.</returns>
                        /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                        internal static PyObject dtype(PyObject space) => space.GetAttr("dtype");
                    }
                }

                /// <summary><c>gymnasium.spaces.discrete</c>, where <c>Discrete</c> is defined.</summary>
                internal static class discrete {
                    /// <summary><c>Discrete</c> instance attributes.</summary>
                    internal static class Discrete {
                        /// <summary><c>space.n</c> (an <c>np.int64</c>).</summary>
                        /// <param name="space">The space.</param>
                        /// <returns>A new owned reference.</returns>
                        /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                        internal static PyObject n(PyObject space) => space.GetAttr("n");

                        /// <summary><c>space.start</c> (an <c>np.int64</c>).</summary>
                        /// <param name="space">The space.</param>
                        /// <returns>A new owned reference.</returns>
                        /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                        internal static PyObject start(PyObject space) => space.GetAttr("start");
                    }
                }

                /// <summary><c>gymnasium.spaces.box</c>, where <c>Box</c> is defined.</summary>
                internal static class box {
                    /// <summary><c>Box</c> instance attributes and methods.</summary>
                    internal static class Box {
                        /// <summary><c>space.low</c>.</summary>
                        /// <param name="space">The box.</param>
                        /// <returns>A new owned reference.</returns>
                        /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                        internal static PyObject low(PyObject space) => space.GetAttr("low");

                        /// <summary><c>space.high</c>.</summary>
                        /// <param name="space">The box.</param>
                        /// <returns>A new owned reference.</returns>
                        /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                        internal static PyObject high(PyObject space) => space.GetAttr("high");

                        /// <summary><c>space.bounded_below</c>.</summary>
                        /// <param name="space">The box.</param>
                        /// <returns>A new owned reference.</returns>
                        /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                        internal static PyObject bounded_below(PyObject space) => space.GetAttr("bounded_below");

                        /// <summary><c>space.bounded_above</c>.</summary>
                        /// <param name="space">The box.</param>
                        /// <returns>A new owned reference.</returns>
                        /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                        internal static PyObject bounded_above(PyObject space) => space.GetAttr("bounded_above");

                        /// <summary><c>space.is_bounded(manner)</c>.</summary>
                        /// <param name="space">The box.</param>
                        /// <param name="manner"><c>both</c>, <c>below</c> or <c>above</c>.</param>
                        /// <returns>A new owned reference to the boolean.</returns>
                        /// <exception cref="PythonException">Thrown when the call raises.</exception>
                        internal static PyObject is_bounded(PyObject space, string manner) {
                            using PyObject method = space.GetAttr("is_bounded");
                            using var text = new PyString(manner);
                            return method.Invoke(text);
                        }
                    }
                }
            }

            /// <summary><c>gymnasium.utils</c>.</summary>
            internal static class utils {
                /// <summary><c>gymnasium.utils.seeding</c>.</summary>
                internal static class seeding {
                    /// <summary><c>seeding.np_random(seed)</c>.</summary>
                    /// <param name="seed">The seed as a Python int (any size), or <see langword="null"/> for entropy; stays owned by the caller.</param>
                    /// <returns>A new owned reference to <c>(rng, np_seed)</c>.</returns>
                    /// <exception cref="PythonException">Thrown when the seed is rejected (<c>gymnasium.error.Error</c>).</exception>
                    internal static PyObject np_random(PyObject? seed) {
                        using PyObject function = PythonRuntime.Import("gymnasium.utils.seeding").GetAttr("np_random");
                        return seed is null ? function.Invoke() : function.Invoke(seed);
                    }
                }
            }

            /// <summary><c>gymnasium.envs</c>, the env families.</summary>
            internal static class envs {
                /// <summary><c>gymnasium.envs.classic_control</c>.</summary>
                internal static class classic_control {
                    /// <summary><c>gymnasium.envs.classic_control.cartpole</c>.</summary>
                    internal static class cartpole {
                        /// <summary><c>CartPoleEnv</c> instance attributes (on the unwrapped env).</summary>
                        internal static class CartPoleEnv {
                            /// <summary><c>env.state</c>: the float64 state array, or <c>None</c> before the first reset.</summary>
                            /// <param name="env">The unwrapped env.</param>
                            /// <returns>A new owned reference.</returns>
                            /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                            internal static PyObject state(PyObject env) => env.GetAttr("state");

                            /// <summary><c>env.state = value</c> (state injection).</summary>
                            /// <param name="env">The unwrapped env.</param>
                            /// <param name="value">The new state; stays owned by the caller (the env keeps its own reference).</param>
                            /// <exception cref="PythonException">Thrown when the assignment fails.</exception>
                            internal static void state(PyObject env, PyObject value) => env.SetAttr("state", value);

                            /// <summary><c>env.steps_beyond_terminated</c>: <c>None</c>, or the steps taken since the pole fell.</summary>
                            /// <param name="env">The unwrapped env.</param>
                            /// <returns>A new owned reference.</returns>
                            /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                            internal static PyObject steps_beyond_terminated(PyObject env) => env.GetAttr("steps_beyond_terminated");

                            /// <summary><c>env.kinematics_integrator = value</c>: <c>euler</c>, or anything else for semi-implicit Euler.</summary>
                            /// <param name="env">The unwrapped env.</param>
                            /// <param name="value">The integrator's name.</param>
                            /// <exception cref="PythonException">Thrown when the assignment fails.</exception>
                            internal static void kinematics_integrator(PyObject env, string value) {
                                using var text = new PyString(value);
                                env.SetAttr("kinematics_integrator", text);
                            }

                            /// <summary><c>env.tau = value</c>: the integration step in seconds.</summary>
                            /// <param name="env">The unwrapped env.</param>
                            /// <param name="value">The step.</param>
                            /// <exception cref="PythonException">Thrown when the assignment fails.</exception>
                            internal static void tau(PyObject env, double value) {
                                using var number = new PyFloat(value);
                                env.SetAttr("tau", number);
                            }
                        }
                    }
                }

                /// <summary><c>gymnasium.envs.box2d</c>.</summary>
                internal static class box2d {
                    /// <summary><c>gymnasium.envs.box2d.lunar_lander</c>.</summary>
                    internal static class lunar_lander {
                        /// <summary><c>lunar_lander.heuristic(env, s)</c>, Gymnasium's own controller.</summary>
                        /// <param name="env">The unwrapped lander.</param>
                        /// <param name="s">The observation; stays owned by the caller.</param>
                        /// <returns>A new owned reference: an int (discrete) or a float64 array of two throttles (continuous).</returns>
                        /// <exception cref="PythonException">Thrown when the controller raises.</exception>
                        internal static PyObject heuristic(PyObject env, PyObject s) {
                            using PyObject function = PythonRuntime.Import("gymnasium.envs.box2d.lunar_lander").GetAttr("heuristic");
                            return function.Invoke(env, s);
                        }
                    }
                }
            }
        }

        /// <summary><c>import pygame</c>, the rasterizer oracle.</summary>
        internal static class pygame {
            /// <summary>The cached module; never dispose it.</summary>
            private static PyObject Module => PythonRuntime.Import("pygame");

            /// <summary><c>pygame.Surface((width, height))</c>: a 32-bit XRGB surface, as Gymnasium's envs create them.</summary>
            /// <param name="width">Columns.</param>
            /// <param name="height">Rows.</param>
            /// <returns>A new owned reference.</returns>
            /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
            internal static PyObject Surface(int width, int height) {
                using PyObject constructor = Module.GetAttr("Surface");
                using PyObject size = Oracle.PyConvert.ToTuple(new long[] { width, height });
                return constructor.Invoke(size);
            }

            /// <summary><c>pygame.Vector2(x, y)</c>.</summary>
            /// <param name="x">The x component.</param>
            /// <param name="y">The y component.</param>
            /// <returns>A new owned reference.</returns>
            /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
            internal static PyObject Vector2(double x, double y) {
                using PyObject constructor = Module.GetAttr("Vector2");
                using var xValue = new PyFloat(x);
                using var yValue = new PyFloat(y);
                return constructor.Invoke(xValue, yValue);
            }

            /// <summary><c>pygame.version</c>.</summary>
            internal static class version {
                /// <summary><c>pygame.version.ver</c>, the pygame-ce version.</summary>
                /// <exception cref="PythonException">Thrown when the import or lookup fails.</exception>
                internal static string ver {
                    get {
                        using PyObject value = PythonRuntime.Import("pygame.version").GetAttr("ver");
                        return value.As<string>();
                    }
                }

                /// <summary><c>str(pygame.version.SDL)</c>, the bundled SDL version.</summary>
                /// <exception cref="PythonException">Thrown when the import or lookup fails.</exception>
                internal static string SDL {
                    get {
                        using PyObject value = PythonRuntime.Import("pygame.version").GetAttr("SDL");
                        return value.ToString() ?? "";
                    }
                }
            }

            /// <summary><c>pygame.display</c>.</summary>
            internal static class display {
                /// <summary><c>pygame.display.init()</c>: <c>gfxdraw</c> requires the video system (headless under <c>SDL_VIDEODRIVER=dummy</c>).</summary>
                /// <exception cref="PythonException">Thrown when initialization fails.</exception>
                internal static void init() {
                    using PyObject function = PythonRuntime.Import("pygame.display").GetAttr("init");
                    using PyObject _ = function.Invoke();
                }
            }

            /// <summary><c>pygame.surface</c>, where <c>Surface</c> is defined.</summary>
            internal static class surface {
                /// <summary><c>Surface</c> instance methods.</summary>
                internal static class Surface {
                    /// <summary><c>surface.fill(color)</c>.</summary>
                    /// <param name="surface">The surface.</param>
                    /// <param name="color">The colour as <c>(r, g, b)</c>.</param>
                    /// <exception cref="PythonException">Thrown when filling raises.</exception>
                    internal static void fill(PyObject surface, (int R, int G, int B) color) {
                        using PyObject method = surface.GetAttr("fill");
                        using PyObject rgb = Oracle.PyConvert.ToTuple(new long[] { color.R, color.G, color.B });
                        using PyObject _ = method.Invoke(rgb);
                    }
                }
            }

            /// <summary><c>pygame.gfxdraw</c>, SDL_gfx's primitives.</summary>
            internal static class gfxdraw {
                /// <summary>The cached module; never dispose it.</summary>
                private static PyObject Module => PythonRuntime.Import("pygame.gfxdraw");

                /// <summary><c>gfxdraw.aapolygon(surface, points, color)</c>.</summary>
                /// <param name="surface">The surface.</param>
                /// <param name="points">The vertices, one <c>(x, y)</c> row each; they cross as a float64 NumPy array.</param>
                /// <param name="color">The colour as <c>(r, g, b, a)</c>.</param>
                /// <exception cref="PythonException">Thrown when drawing raises.</exception>
                internal static void aapolygon(PyObject surface, double[,] points, (int R, int G, int B, int A) color) => Polygon("aapolygon", surface, points, color);

                /// <summary><c>gfxdraw.filled_polygon(surface, points, color)</c>.</summary>
                /// <param name="surface">The surface.</param>
                /// <param name="points">The vertices, one <c>(x, y)</c> row each; they cross as a float64 NumPy array.</param>
                /// <param name="color">The colour as <c>(r, g, b, a)</c>.</param>
                /// <exception cref="PythonException">Thrown when drawing raises.</exception>
                internal static void filled_polygon(PyObject surface, double[,] points, (int R, int G, int B, int A) color) => Polygon("filled_polygon", surface, points, color);

                /// <summary><c>gfxdraw.aacircle(surface, x, y, r, color)</c>.</summary>
                /// <param name="surface">The surface.</param>
                /// <param name="x">The centre's column.</param>
                /// <param name="y">The centre's row.</param>
                /// <param name="r">The radius.</param>
                /// <param name="color">The colour as <c>(r, g, b, a)</c>.</param>
                /// <exception cref="PythonException">Thrown when drawing raises.</exception>
                internal static void aacircle(PyObject surface, long x, long y, long r, (int R, int G, int B, int A) color) => Ints("aacircle", surface, new long[] { x, y, r }, color);

                /// <summary><c>gfxdraw.filled_circle(surface, x, y, r, color)</c>.</summary>
                /// <param name="surface">The surface.</param>
                /// <param name="x">The centre's column.</param>
                /// <param name="y">The centre's row.</param>
                /// <param name="r">The radius.</param>
                /// <param name="color">The colour as <c>(r, g, b, a)</c>.</param>
                /// <exception cref="PythonException">Thrown when drawing raises.</exception>
                internal static void filled_circle(PyObject surface, long x, long y, long r, (int R, int G, int B, int A) color) => Ints("filled_circle", surface, new long[] { x, y, r }, color);

                /// <summary><c>gfxdraw.hline(surface, x1, x2, y, color)</c>.</summary>
                /// <param name="surface">The surface.</param>
                /// <param name="x1">One end's column.</param>
                /// <param name="x2">The other end's column.</param>
                /// <param name="y">The row.</param>
                /// <param name="color">The colour as <c>(r, g, b, a)</c>.</param>
                /// <exception cref="PythonException">Thrown when drawing raises.</exception>
                internal static void hline(PyObject surface, long x1, long x2, long y, (int R, int G, int B, int A) color) => Ints("hline", surface, new long[] { x1, x2, y }, color);

                /// <summary><c>gfxdraw.vline(surface, x, y1, y2, color)</c>.</summary>
                /// <param name="surface">The surface.</param>
                /// <param name="x">The column.</param>
                /// <param name="y1">One end's row.</param>
                /// <param name="y2">The other end's row.</param>
                /// <param name="color">The colour as <c>(r, g, b, a)</c>.</param>
                /// <exception cref="PythonException">Thrown when drawing raises.</exception>
                internal static void vline(PyObject surface, long x, long y1, long y2, (int R, int G, int B, int A) color) => Ints("vline", surface, new long[] { x, y1, y2 }, color);

                /// <summary>Calls a polygon primitive with the vertices crossed through NumSharp's interop.</summary>
                /// <param name="name">The primitive.</param>
                /// <param name="surface">The surface.</param>
                /// <param name="points">The vertices.</param>
                /// <param name="color">The colour.</param>
                private static void Polygon(string name, PyObject surface, double[,] points, (int R, int G, int B, int A) color) {
                    using PyObject function = Module.GetAttr(name);
                    using NDArray vertices = np.array(points);
                    using PyObject pythonVertices = vertices.ToNumpyCopy();
                    using PyObject rgba = Oracle.PyConvert.ToTuple(new long[] { color.R, color.G, color.B, color.A });
                    using PyObject _ = function.Invoke(surface, pythonVertices, rgba);
                }

                /// <summary>Calls a primitive whose geometry is a few ints.</summary>
                /// <param name="name">The primitive.</param>
                /// <param name="surface">The surface.</param>
                /// <param name="geometry">The ints, in the primitive's order.</param>
                /// <param name="color">The colour.</param>
                private static void Ints(string name, PyObject surface, long[] geometry, (int R, int G, int B, int A) color) {
                    using PyObject function = Module.GetAttr(name);
                    var arguments = new List<PyObject> { surface };
                    try {
                        foreach (long value in geometry)
                            arguments.Add(new PyInt(value));
                        arguments.Add(Oracle.PyConvert.ToTuple(new long[] { color.R, color.G, color.B, color.A }));
                        using PyObject _ = function.Invoke(arguments.ToArray());
                    } finally {
                        // The surface is the caller's; everything after it was created here.
                        for (int i = 1; i < arguments.Count; i++)
                            arguments[i].Dispose();
                    }
                }
            }

            /// <summary><c>pygame.transform</c>.</summary>
            internal static class transform {
                /// <summary><c>pygame.transform.flip(surface, flip_x, flip_y)</c>.</summary>
                /// <param name="surface">The surface.</param>
                /// <param name="flip_x">Whether to mirror horizontally.</param>
                /// <param name="flip_y">Whether to mirror vertically.</param>
                /// <returns>A new owned reference to the flipped copy.</returns>
                /// <exception cref="PythonException">Thrown when flipping raises.</exception>
                internal static PyObject flip(PyObject surface, bool flip_x, bool flip_y) {
                    using PyObject function = PythonRuntime.Import("pygame.transform").GetAttr("flip");
                    using PyObject x = flip_x.ToPython();
                    using PyObject y = flip_y.ToPython();
                    return function.Invoke(surface, x, y);
                }
            }

            /// <summary><c>pygame.surfarray</c>.</summary>
            internal static class surfarray {
                /// <summary><c>pygame.surfarray.array3d(surface)</c>: a <c>(W, H, 3)</c> uint8 copy of the pixels.</summary>
                /// <param name="surface">The surface.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the copy fails.</exception>
                internal static PyObject array3d(PyObject surface) {
                    using PyObject function = PythonRuntime.Import("pygame.surfarray").GetAttr("array3d");
                    return function.Invoke(surface);
                }
            }

            /// <summary><c>pygame.math</c>, where <c>Vector2</c> is defined.</summary>
            internal static class math {
                /// <summary><c>Vector2</c> instance members.</summary>
                internal static class Vector2 {
                    /// <summary><c>v.rotate_rad(angle)</c>.</summary>
                    /// <param name="v">The vector.</param>
                    /// <param name="angle">The counter-clockwise angle in radians.</param>
                    /// <returns>A new owned reference to the rotated vector.</returns>
                    /// <exception cref="PythonException">Thrown when the call raises.</exception>
                    internal static PyObject rotate_rad(PyObject v, double angle) {
                        using PyObject method = v.GetAttr("rotate_rad");
                        using var value = new PyFloat(angle);
                        return method.Invoke(value);
                    }

                    /// <summary><c>v.x</c>, a Python <c>float</c>; describe it with <see cref="Oracle.ValueFacts.FromPython"/> to compare its bits.</summary>
                    /// <param name="v">The vector.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                    internal static PyObject x(PyObject v) => v.GetAttr("x");

                    /// <summary><c>v.y</c>, a Python <c>float</c>.</summary>
                    /// <param name="v">The vector.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                    internal static PyObject y(PyObject v) => v.GetAttr("y");
                }
            }
        }

        /// <summary><c>import Box2D</c>, the physics oracle of the box2d envs.</summary>
        internal static class Box2D {
            /// <summary><c>Box2D.__version__</c>.</summary>
            /// <exception cref="PythonException">Thrown when the import or lookup fails.</exception>
            internal static string __version__ {
                get {
                    using PyObject value = PythonRuntime.Import("Box2D").GetAttr("__version__");
                    return value.As<string>();
                }
            }
        }
    }
}
