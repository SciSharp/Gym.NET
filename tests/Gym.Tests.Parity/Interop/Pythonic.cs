using System;
using System.Collections.Generic;
using System.Linq;
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

            /// <summary><c>str(x)</c>: the informal text, which for an env is <c>Env.__str__</c> (<c>&lt;TimeLimit&lt;...&gt;&gt;</c>).</summary>
            /// <param name="x">The value; stays owned by the caller.</param>
            /// <returns>A new owned reference to the <c>str</c>.</returns>
            /// <exception cref="PythonException">Thrown when <c>str()</c> raises.</exception>
            internal static PyObject str(PyObject x) {
                using PyObject function = PythonRuntime.Builtins.GetAttr("str");
                return function.Invoke(x);
            }

            /// <summary><c>getattr(obj, name)</c>: plain attribute access (<c>env.gravity</c>), which a wrapper does not forward.</summary>
            /// <param name="obj">The object; stays owned by the caller.</param>
            /// <param name="name">The attribute.</param>
            /// <returns>A new owned reference to the value.</returns>
            /// <exception cref="PythonException">Thrown when the object has no such attribute (<c>AttributeError</c>).</exception>
            internal static PyObject getattr(PyObject obj, string name) {
                using PyObject function = PythonRuntime.Builtins.GetAttr("getattr");
                using var attribute = new PyString(name);
                return function.Invoke(obj, attribute);
            }

            /// <summary><c>setattr(obj, name, value)</c>: plain attribute assignment (<c>env.gravity = 5.0</c>).</summary>
            /// <param name="obj">The object; stays owned by the caller.</param>
            /// <param name="name">The attribute.</param>
            /// <param name="value">The value; stays owned by the caller.</param>
            /// <returns>A new owned reference to <c>None</c>, so the call can be captured like any other.</returns>
            /// <exception cref="PythonException">Thrown when the assignment raises (a property without a setter).</exception>
            internal static PyObject setattr(PyObject obj, string name, PyObject value) {
                using PyObject function = PythonRuntime.Builtins.GetAttr("setattr");
                using var attribute = new PyString(name);
                return function.Invoke(obj, attribute, value);
            }

            /// <summary><c>divmod(a, b)</c>: <c>(a // b, a % b)</c>.</summary>
            /// <param name="a">The dividend; stays owned by the caller.</param>
            /// <param name="b">The divisor; stays owned by the caller.</param>
            /// <returns>A new owned reference to the pair.</returns>
            /// <exception cref="PythonException">Thrown for a zero divisor (<c>ZeroDivisionError</c>).</exception>
            internal static PyObject divmod(PyObject a, PyObject b) {
                using PyObject function = PythonRuntime.Builtins.GetAttr("divmod");
                return function.Invoke(a, b);
            }

            /// <summary><c>round(x)</c>: the nearest int, ties to even.</summary>
            /// <param name="x">The number; stays owned by the caller.</param>
            /// <returns>A new owned reference to the int.</returns>
            /// <exception cref="PythonException">Thrown for NaN (<c>ValueError</c>) and the infinities (<c>OverflowError</c>).</exception>
            internal static PyObject round(PyObject x) {
                using PyObject function = PythonRuntime.Builtins.GetAttr("round");
                return function.Invoke(x);
            }

            /// <summary><c>round(x, ndigits)</c>: the float rounded to <paramref name="ndigits"/> decimal places.</summary>
            /// <param name="x">The number; stays owned by the caller.</param>
            /// <param name="ndigits">The number of places; stays owned by the caller.</param>
            /// <returns>A new owned reference to the float.</returns>
            /// <exception cref="PythonException">Thrown when the result overflows (<c>OverflowError</c>).</exception>
            internal static PyObject round(PyObject x, PyObject ndigits) {
                using PyObject function = PythonRuntime.Builtins.GetAttr("round");
                return function.Invoke(x, ndigits);
            }

            /// <summary><c>min(a, b)</c>.</summary>
            /// <param name="a">The first value; stays owned by the caller.</param>
            /// <param name="b">The second value; stays owned by the caller.</param>
            /// <returns>A new owned reference to the one Python picks.</returns>
            /// <exception cref="PythonException">Thrown when the values cannot be compared.</exception>
            internal static PyObject min(PyObject a, PyObject b) {
                using PyObject function = PythonRuntime.Builtins.GetAttr("min");
                return function.Invoke(a, b);
            }

            /// <summary><c>max(a, b)</c>.</summary>
            /// <param name="a">The first value; stays owned by the caller.</param>
            /// <param name="b">The second value; stays owned by the caller.</param>
            /// <returns>A new owned reference to the one Python picks.</returns>
            /// <exception cref="PythonException">Thrown when the values cannot be compared.</exception>
            internal static PyObject max(PyObject a, PyObject b) {
                using PyObject function = PythonRuntime.Builtins.GetAttr("max");
                return function.Invoke(a, b);
            }

            /// <summary><c>list(x)</c>: drains an iterable (Gymnasium's <c>iterate</c> returns an iterator) into a list.</summary>
            /// <param name="x">The iterable; stays owned by the caller.</param>
            /// <returns>A new owned reference to the list.</returns>
            /// <exception cref="PythonException">Thrown when iterating raises.</exception>
            internal static PyObject list(PyObject x) {
                using PyObject function = PythonRuntime.Builtins.GetAttr("list");
                return function.Invoke(x);
            }

            /// <summary><c>len(x)</c>.</summary>
            /// <param name="x">The sized object; stays owned by the caller.</param>
            /// <returns>A new owned reference to the int.</returns>
            /// <exception cref="PythonException">Thrown when the object has no length.</exception>
            internal static PyObject len(PyObject x) {
                using PyObject function = PythonRuntime.Builtins.GetAttr("len");
                return function.Invoke(x);
            }
        }

        /// <summary><c>import operator</c>: Python's operators as functions, so <c>a // b</c> and <c>a % b</c> are facade rows.</summary>
        internal static class @operator {
            /// <summary>The cached module; never dispose it.</summary>
            private static PyObject Module => PythonRuntime.Import("operator");

            /// <summary><c>operator.floordiv(a, b)</c>, Python's <c>a // b</c>.</summary>
            /// <param name="a">The dividend; stays owned by the caller.</param>
            /// <param name="b">The divisor; stays owned by the caller.</param>
            /// <returns>A new owned reference to the quotient.</returns>
            /// <exception cref="PythonException">Thrown for a zero divisor (<c>ZeroDivisionError</c>).</exception>
            internal static PyObject floordiv(PyObject a, PyObject b) {
                using PyObject function = Module.GetAttr("floordiv");
                return function.Invoke(a, b);
            }

            /// <summary><c>operator.mod(a, b)</c>, Python's <c>a % b</c>.</summary>
            /// <param name="a">The dividend; stays owned by the caller.</param>
            /// <param name="b">The divisor; stays owned by the caller.</param>
            /// <returns>A new owned reference to the remainder.</returns>
            /// <exception cref="PythonException">Thrown for a zero divisor (<c>ZeroDivisionError</c>).</exception>
            internal static PyObject mod(PyObject a, PyObject b) {
                using PyObject function = Module.GetAttr("mod");
                return function.Invoke(a, b);
            }

            /// <summary><c>operator.eq(a, b)</c>, Python's <c>a == b</c> (a space's <c>__eq__</c>).</summary>
            /// <param name="a">The left operand; stays owned by the caller.</param>
            /// <param name="b">The right operand; stays owned by the caller.</param>
            /// <returns>A new owned reference to the result.</returns>
            /// <exception cref="PythonException">Thrown when the comparison raises.</exception>
            internal static PyObject eq(PyObject a, PyObject b) {
                using PyObject function = Module.GetAttr("eq");
                return function.Invoke(a, b);
            }

            /// <summary><c>operator.getitem(a, b)</c>, Python's <c>a[b]</c> (a composite space's subspace, a tuple's item).</summary>
            /// <param name="a">The container; stays owned by the caller.</param>
            /// <param name="b">The index or key; stays owned by the caller.</param>
            /// <returns>A new owned reference to the item.</returns>
            /// <exception cref="PythonException">Thrown when indexing raises (<c>IndexError</c>, <c>KeyError</c>).</exception>
            internal static PyObject getitem(PyObject a, PyObject b) {
                using PyObject function = Module.GetAttr("getitem");
                return function.Invoke(a, b);
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

            /// <summary><c>np.clip(a, a_min, a_max)</c>.</summary>
            /// <param name="a">The value; stays owned by the caller.</param>
            /// <param name="a_min">The lower bound; stays owned by the caller.</param>
            /// <param name="a_max">The upper bound; stays owned by the caller.</param>
            /// <returns>A new owned reference: a NumPy scalar for scalar arguments.</returns>
            /// <exception cref="PythonException">Thrown when the call fails.</exception>
            internal static PyObject clip(PyObject a, PyObject a_min, PyObject a_max) {
                using PyObject function = Module.GetAttr("clip");
                return function.Invoke(a, a_min, a_max);
            }

            /// <summary><c>np.sign(x)</c>.</summary>
            /// <param name="x">The value; stays owned by the caller.</param>
            /// <returns>A new owned reference: a NumPy scalar for a scalar argument.</returns>
            /// <exception cref="PythonException">Thrown when the call fails.</exception>
            internal static PyObject sign(PyObject x) {
                using PyObject function = Module.GetAttr("sign");
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

            /// <summary><c>gymnasium.spec(env_id)</c>: the registered spec, or the registry's error for a missing id.</summary>
            /// <param name="env_id">The id.</param>
            /// <returns>A new owned reference to the spec.</returns>
            /// <exception cref="PythonException">Thrown with Gymnasium's error for a malformed or unregistered id.</exception>
            internal static PyObject spec(string env_id) {
                using PyObject function = Module.GetAttr("spec");
                using var id = new PyString(env_id);
                return function.Invoke(id);
            }

            /// <summary><c>gymnasium.register(id, entry_point, **options)</c>: registers an id in the live registry.</summary>
            /// <param name="id">The id.</param>
            /// <param name="entry_point">The entry-point string.</param>
            /// <param name="options">
            ///     <c>register</c>'s other keyword arguments in Python spelling (<c>max_episode_steps</c>, <c>kwargs</c>, ...), or
            ///     <see langword="null"/>; values converted by <see cref="Oracle.PyConvert.ToPython"/>.
            /// </param>
            /// <returns>A new owned reference to <c>None</c>.</returns>
            /// <exception cref="PythonException">Thrown with Gymnasium's error for a malformed id or a registration conflict.</exception>
            internal static PyObject register(string id, string entry_point, IReadOnlyDictionary<string, object?>? options = null) {
                using PyObject function = Module.GetAttr("register");
                using var idValue = new PyString(id);
                using var entryPoint = new PyString(entry_point);
                using var arguments = Oracle.PyConvert.ToKwargs(options);
                return function.Invoke(new PyObject[] { idValue, entryPoint }, arguments);
            }

            /// <summary><c>gymnasium.registry</c>, the live registry dict.</summary>
            internal static class registry {
                /// <summary><c>gymnasium.registry.pop(env_id, None)</c>: unregisters an id, as the tests' cleanups do.</summary>
                /// <param name="env_id">The id.</param>
                /// <returns>A new owned reference to the removed spec, or <c>None</c>.</returns>
                /// <exception cref="PythonException">Thrown when the call fails.</exception>
                internal static PyObject pop(string env_id) {
                    using PyObject registryDict = Module.GetAttr("registry");
                    using PyObject method = registryDict.GetAttr("pop");
                    using var id = new PyString(env_id);
                    using PyObject none = Oracle.PyConvert.ToPython(null);
                    return method.Invoke(id, none);
                }
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

                /// <summary><c>env.np_random = value</c>: after it, <c>np_random_seed</c> is -1.</summary>
                /// <param name="env">The env.</param>
                /// <param name="value">The generator; stays owned by the caller (the env keeps its own reference).</param>
                /// <exception cref="PythonException">Thrown when the assignment fails.</exception>
                internal static void np_random(PyObject env, PyObject value) => env.SetAttr("np_random", value);

                /// <summary><c>env.np_random_seed</c>. Like <c>np_random</c>, reading it creates a generator when none exists.</summary>
                /// <param name="env">The env.</param>
                /// <returns>A new owned reference to the int.</returns>
                /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                internal static PyObject np_random_seed(PyObject env) => env.GetAttr("np_random_seed");

                /// <summary><c>env.spec</c>: the env's spec, or for a wrapper its env's extended with the wrapper.</summary>
                /// <param name="env">The env.</param>
                /// <returns>A new owned reference to the <c>EnvSpec</c>, or <c>None</c>.</returns>
                /// <exception cref="PythonException">Thrown when computing the spec raises.</exception>
                internal static PyObject spec(PyObject env) => env.GetAttr("spec");

                /// <summary><c>env.spec = value</c>, what <c>make</c> does to the unwrapped env.</summary>
                /// <param name="env">The env (a wrapper refuses: its <c>spec</c> has no setter).</param>
                /// <param name="value">The spec; stays owned by the caller.</param>
                /// <exception cref="PythonException">Thrown when the assignment raises.</exception>
                internal static void spec(PyObject env, PyObject value) => env.SetAttr("spec", value);

                /// <summary><c>env.render_mode</c>.</summary>
                /// <param name="env">The env.</param>
                /// <returns>A new owned reference to the <c>str</c>, or <c>None</c>.</returns>
                /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                internal static PyObject render_mode(PyObject env) => env.GetAttr("render_mode");

                /// <summary><c>env.has_wrapper_attr(name)</c>.</summary>
                /// <param name="env">The env.</param>
                /// <param name="name">The attribute.</param>
                /// <returns>A new owned reference to the bool.</returns>
                /// <exception cref="PythonException">Thrown when the call raises.</exception>
                internal static PyObject has_wrapper_attr(PyObject env, string name) {
                    using PyObject method = env.GetAttr("has_wrapper_attr");
                    using var attribute = new PyString(name);
                    return method.Invoke(attribute);
                }

                /// <summary><c>env.get_wrapper_attr(name)</c>.</summary>
                /// <param name="env">The env.</param>
                /// <param name="name">The attribute.</param>
                /// <returns>A new owned reference to the value.</returns>
                /// <exception cref="PythonException">Thrown when no layer has the attribute (<c>AttributeError</c>).</exception>
                internal static PyObject get_wrapper_attr(PyObject env, string name) {
                    using PyObject method = env.GetAttr("get_wrapper_attr");
                    using var attribute = new PyString(name);
                    return method.Invoke(attribute);
                }

                /// <summary><c>env.set_wrapper_attr(name, value, force=force)</c>.</summary>
                /// <param name="env">The env.</param>
                /// <param name="name">The attribute.</param>
                /// <param name="value">The value; stays owned by the caller.</param>
                /// <param name="force">Create the attribute on the outermost layer when no layer has it.</param>
                /// <returns>A new owned reference to the bool: whether a layer was assigned.</returns>
                /// <exception cref="PythonException">Thrown when the assignment raises.</exception>
                internal static PyObject set_wrapper_attr(PyObject env, string name, PyObject value, bool force = true) {
                    using PyObject method = env.GetAttr("set_wrapper_attr");
                    using var attribute = new PyString(name);
                    using var arguments = new PyDict();
                    using PyObject forceValue = force.ToPython();
                    arguments["force"] = forceValue;
                    return method.Invoke(new[] { attribute, value }, arguments);
                }
            }

            /// <summary><c>gymnasium.wrappers</c>: the wrapper constructors (their members live in <c>gymnasium.wrappers.common</c>).</summary>
            internal static class wrappers {
                /// <summary>The cached module; never dispose it.</summary>
                private static PyObject Module => PythonRuntime.Import("gymnasium.wrappers");

                /// <summary><c>wrappers.TimeLimit(env, max_episode_steps)</c>.</summary>
                /// <param name="env">The env; the wrapper keeps its own reference.</param>
                /// <param name="max_episode_steps">The limit, any Python value (<c>None</c> and negatives are refused); stays owned by the caller.</param>
                /// <returns>A new owned reference to the wrapper.</returns>
                /// <exception cref="PythonException">Thrown for a limit that is not a positive int (<c>TypeError</c>, <c>ValueError</c>).</exception>
                internal static PyObject TimeLimit(PyObject env, PyObject max_episode_steps) {
                    using PyObject constructor = Module.GetAttr("TimeLimit");
                    return constructor.Invoke(env, max_episode_steps);
                }

                /// <summary><c>wrappers.OrderEnforcing(env, disable_render_order_enforcing=...)</c>.</summary>
                /// <param name="env">The env; the wrapper keeps its own reference.</param>
                /// <param name="disable_render_order_enforcing">Allow <c>render()</c> before the first reset.</param>
                /// <returns>A new owned reference to the wrapper.</returns>
                /// <exception cref="PythonException">Thrown when the construction fails.</exception>
                internal static PyObject OrderEnforcing(PyObject env, bool disable_render_order_enforcing = false) {
                    using PyObject constructor = Module.GetAttr("OrderEnforcing");
                    using var arguments = new PyDict();
                    using PyObject flag = disable_render_order_enforcing.ToPython();
                    arguments["disable_render_order_enforcing"] = flag;
                    return constructor.Invoke(new[] { env }, arguments);
                }

                /// <summary><c>gymnasium.wrappers.common</c>, where the common wrappers are defined.</summary>
                internal static class common {
                    /// <summary><c>OrderEnforcing</c> instance members.</summary>
                    internal static class OrderEnforcing {
                        /// <summary><c>env.has_reset</c>.</summary>
                        /// <param name="env">The wrapper.</param>
                        /// <returns>A new owned reference to the bool.</returns>
                        /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
                        internal static PyObject has_reset(PyObject env) => env.GetAttr("has_reset");
                    }
                }
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

                /// <summary><c>spaces.Discrete(n, seed=seed, start=start, dtype=dtype)</c>.</summary>
                /// <param name="n">The number of elements.</param>
                /// <param name="seed">The space's seed, or <see langword="null"/>.</param>
                /// <param name="start">The smallest element.</param>
                /// <param name="dtype">The integer dtype; stays owned by the caller.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
                internal static PyObject Discrete(long n, long? seed, long start, PyObject dtype) {
                    using PyObject constructor = Module.GetAttr("Discrete");
                    using var count = new PyInt(n);
                    using var arguments = new PyDict();
                    using PyObject seedValue = Oracle.PyConvert.ToPython(seed);
                    using var startValue = new PyInt(start);
                    arguments["seed"] = seedValue;
                    arguments["start"] = startValue;
                    arguments["dtype"] = dtype;
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

                /// <summary><c>spaces.MultiBinary(n, seed=seed)</c>.</summary>
                /// <param name="n">The shape: a Python <c>int</c> or a list/tuple of ints; stays owned by the caller.</param>
                /// <param name="seed">The space's seed, or <see langword="null"/>.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
                internal static PyObject MultiBinary(PyObject n, long? seed = null) {
                    using PyObject constructor = Module.GetAttr("MultiBinary");
                    using var arguments = new PyDict();
                    using PyObject seedValue = Oracle.PyConvert.ToPython(seed);
                    arguments["seed"] = seedValue;
                    return constructor.Invoke(new[] { n }, arguments);
                }

                /// <summary><c>spaces.MultiDiscrete(nvec, dtype=dtype, seed=seed, start=start)</c>.</summary>
                /// <param name="nvec">The counts: an array or a (nested) list; stays owned by the caller.</param>
                /// <param name="dtype">The dtype, or <see langword="null"/> for Gymnasium's default <c>np.int64</c>; stays owned by the caller.</param>
                /// <param name="seed">The space's seed, or <see langword="null"/>.</param>
                /// <param name="start">The starts, or <see langword="null"/>; stays owned by the caller.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
                internal static PyObject MultiDiscrete(PyObject nvec, PyObject? dtype = null, long? seed = null, PyObject? start = null) {
                    using PyObject constructor = Module.GetAttr("MultiDiscrete");
                    using var arguments = new PyDict();
                    using PyObject seedValue = Oracle.PyConvert.ToPython(seed);
                    arguments["seed"] = seedValue;
                    if (dtype != null)
                        arguments["dtype"] = dtype;
                    if (start != null)
                        arguments["start"] = start;
                    return constructor.Invoke(new[] { nvec }, arguments);
                }

                /// <summary><c>spaces.Tuple(spaces, seed=seed)</c>.</summary>
                /// <param name="spaces">The subspaces, a Python tuple or list; stays owned by the caller.</param>
                /// <param name="seed">The space's seed, or <see langword="null"/>.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
                internal static PyObject Tuple(PyObject spaces, long? seed = null) {
                    using PyObject constructor = Module.GetAttr("Tuple");
                    using var arguments = new PyDict();
                    using PyObject seedValue = Oracle.PyConvert.ToPython(seed);
                    arguments["seed"] = seedValue;
                    return constructor.Invoke(new[] { spaces }, arguments);
                }

                /// <summary><c>spaces.Dict(spaces, seed=seed, sort_keys=sort_keys)</c>.</summary>
                /// <param name="spaces">The subspaces: a <c>dict</c> or a list of <c>(key, space)</c> pairs; stays owned by the caller.</param>
                /// <param name="seed">The space's seed, or <see langword="null"/>.</param>
                /// <param name="sortKeys">Gymnasium's <c>sort_keys</c>.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
                internal static PyObject Dict(PyObject spaces, long? seed = null, bool sortKeys = true) {
                    using PyObject constructor = Module.GetAttr("Dict");
                    using var arguments = new PyDict();
                    using PyObject seedValue = Oracle.PyConvert.ToPython(seed);
                    using PyObject sort = sortKeys.ToPython();
                    arguments["seed"] = seedValue;
                    arguments["sort_keys"] = sort;
                    return constructor.Invoke(new[] { spaces }, arguments);
                }

                /// <summary><c>spaces.Text(max_length, min_length=min_length, charset=charset, seed=seed)</c>.</summary>
                /// <param name="maxLength">The longest text.</param>
                /// <param name="minLength">The shortest text.</param>
                /// <param name="charset">The characters (a <c>str</c> or a <c>set</c>), or <see langword="null"/> for Gymnasium's alphanumeric default; stays owned by the caller.</param>
                /// <param name="seed">The space's seed, or <see langword="null"/>.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
                internal static PyObject Text(long maxLength, long minLength = 1, PyObject? charset = null, long? seed = null) {
                    using PyObject constructor = Module.GetAttr("Text");
                    using var maximum = new PyInt(maxLength);
                    using var arguments = new PyDict();
                    using var minimum = new PyInt(minLength);
                    using PyObject seedValue = Oracle.PyConvert.ToPython(seed);
                    arguments["min_length"] = minimum;
                    arguments["seed"] = seedValue;
                    if (charset != null)
                        arguments["charset"] = charset;
                    return constructor.Invoke(new PyObject[] { maximum }, arguments);
                }

                /// <summary><c>spaces.Sequence(space, seed=seed, stack=stack)</c>.</summary>
                /// <param name="space">The feature space; stays owned by the caller.</param>
                /// <param name="seed">The space's seed, or <see langword="null"/>.</param>
                /// <param name="stack">Gymnasium's <c>stack</c>.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
                internal static PyObject Sequence(PyObject space, long? seed = null, bool stack = false) {
                    using PyObject constructor = Module.GetAttr("Sequence");
                    using var arguments = new PyDict();
                    using PyObject seedValue = Oracle.PyConvert.ToPython(seed);
                    using PyObject stacked = stack.ToPython();
                    arguments["seed"] = seedValue;
                    arguments["stack"] = stacked;
                    return constructor.Invoke(new[] { space }, arguments);
                }

                /// <summary><c>spaces.Graph(node_space, edge_space, seed=seed)</c>.</summary>
                /// <param name="nodeSpace">The node feature space; stays owned by the caller.</param>
                /// <param name="edgeSpace">The edge feature space, or Python's <c>None</c>; stays owned by the caller.</param>
                /// <param name="seed">The space's seed, or <see langword="null"/>.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
                internal static PyObject Graph(PyObject nodeSpace, PyObject edgeSpace, long? seed = null) {
                    using PyObject constructor = Module.GetAttr("Graph");
                    using var arguments = new PyDict();
                    using PyObject seedValue = Oracle.PyConvert.ToPython(seed);
                    arguments["seed"] = seedValue;
                    return constructor.Invoke(new[] { nodeSpace, edgeSpace }, arguments);
                }

                /// <summary><c>spaces.GraphInstance(nodes, edges, edge_links)</c>, a graph to test membership with.</summary>
                /// <param name="nodes">The node features; stays owned by the caller.</param>
                /// <param name="edges">The edge features or <c>None</c>; stays owned by the caller.</param>
                /// <param name="edgeLinks">The links or <c>None</c>; stays owned by the caller.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
                internal static PyObject GraphInstance(PyObject nodes, PyObject edges, PyObject edgeLinks) {
                    using PyObject constructor = Module.GetAttr("GraphInstance");
                    return constructor.Invoke(nodes, edges, edgeLinks);
                }

                /// <summary><c>spaces.OneOf(spaces, seed=seed)</c>.</summary>
                /// <param name="spaces">The subspaces, a Python tuple or list; stays owned by the caller.</param>
                /// <param name="seed">The space's seed, or <see langword="null"/>.</param>
                /// <returns>A new owned reference.</returns>
                /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
                internal static PyObject OneOf(PyObject spaces, long? seed = null) {
                    using PyObject constructor = Module.GetAttr("OneOf");
                    using var arguments = new PyDict();
                    using PyObject seedValue = Oracle.PyConvert.ToPython(seed);
                    arguments["seed"] = seedValue;
                    return constructor.Invoke(new[] { spaces }, arguments);
                }

                /// <summary><c>gymnasium.spaces.utils</c>: flattening spaces and their elements.</summary>
                internal static class utils {
                    /// <summary>The cached module; never dispose it.</summary>
                    private static PyObject Module => PythonRuntime.Import("gymnasium.spaces.utils");

                    /// <summary><c>flatdim(space)</c>.</summary>
                    /// <param name="space">The space; stays owned by the caller.</param>
                    /// <returns>A new owned reference to the int.</returns>
                    /// <exception cref="PythonException">Thrown for a space that cannot be flattened.</exception>
                    internal static PyObject flatdim(PyObject space) {
                        using PyObject function = Module.GetAttr("flatdim");
                        return function.Invoke(space);
                    }

                    /// <summary><c>flatten(space, x)</c>.</summary>
                    /// <param name="space">The space; stays owned by the caller.</param>
                    /// <param name="x">The element; stays owned by the caller.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when flattening raises.</exception>
                    internal static PyObject flatten(PyObject space, PyObject x) {
                        using PyObject function = Module.GetAttr("flatten");
                        return function.Invoke(space, x);
                    }

                    /// <summary><c>unflatten(space, x)</c>.</summary>
                    /// <param name="space">The space; stays owned by the caller.</param>
                    /// <param name="x">The flattened element; stays owned by the caller.</param>
                    /// <returns>A new owned reference.</returns>
                    /// <exception cref="PythonException">Thrown when unflattening raises.</exception>
                    internal static PyObject unflatten(PyObject space, PyObject x) {
                        using PyObject function = Module.GetAttr("unflatten");
                        return function.Invoke(space, x);
                    }

                    /// <summary><c>flatten_space(space)</c>.</summary>
                    /// <param name="space">The space; stays owned by the caller.</param>
                    /// <returns>A new owned reference to the flattened space.</returns>
                    /// <exception cref="PythonException">Thrown for a space that cannot be flattened.</exception>
                    internal static PyObject flatten_space(PyObject space) {
                        using PyObject function = Module.GetAttr("flatten_space");
                        return function.Invoke(space);
                    }

                    /// <summary><c>is_space_dtype_shape_equiv(space_1, space_2)</c>.</summary>
                    /// <param name="space1">The first space; stays owned by the caller.</param>
                    /// <param name="space2">The second space; stays owned by the caller.</param>
                    /// <returns>A new owned reference to the boolean.</returns>
                    /// <exception cref="PythonException">Thrown when the check raises.</exception>
                    internal static PyObject is_space_dtype_shape_equiv(PyObject space1, PyObject space2) {
                        using PyObject function = Module.GetAttr("is_space_dtype_shape_equiv");
                        return function.Invoke(space1, space2);
                    }
                }

                /// <summary><c>gymnasium.spaces.graph</c>, where <c>Graph</c> is defined.</summary>
                internal static class graph {
                    /// <summary><c>Graph</c> members.</summary>
                    internal static class Graph {
                        /// <summary><c>space.sample(mask=mask, probability=probability, num_nodes=num_nodes, num_edges=num_edges)</c>.</summary>
                        /// <param name="space">The graph space.</param>
                        /// <param name="mask">The mask pair, or <see langword="null"/>; stays owned by the caller.</param>
                        /// <param name="probability">The probability pair, or <see langword="null"/>; stays owned by the caller.</param>
                        /// <param name="numNodes">The number of nodes.</param>
                        /// <param name="numEdges">The number of edges, or <see langword="null"/> to draw it.</param>
                        /// <returns>A new owned reference to the <c>GraphInstance</c>.</returns>
                        /// <exception cref="PythonException">Thrown when sampling raises.</exception>
                        internal static PyObject sample(PyObject space, PyObject? mask, PyObject? probability, long numNodes, long? numEdges) {
                            using PyObject method = space.GetAttr("sample");
                            using var arguments = new PyDict();
                            using var nodes = new PyInt(numNodes);
                            using PyObject edges = Oracle.PyConvert.ToPython(numEdges);
                            if (mask != null)
                                arguments["mask"] = mask;
                            if (probability != null)
                                arguments["probability"] = probability;
                            arguments["num_nodes"] = nodes;
                            arguments["num_edges"] = edges;
                            return method.Invoke(Array.Empty<PyObject>(), arguments);
                        }
                    }
                }

                /// <summary><c>gymnasium.spaces.space</c>, where the <c>Space</c> base class is defined.</summary>
                internal static class space {
                    /// <summary><c>Space</c> protocol members, called on any space (Python dispatches them to the subclass).</summary>
                    internal static class Space {
                        /// <summary><c>space.sample()</c>, <c>space.sample(mask=mask)</c>, <c>space.sample(probability=probability)</c>, or both keywords.</summary>
                        /// <param name="space">The space.</param>
                        /// <param name="mask">The mask, or <see langword="null"/>; stays owned by the caller.</param>
                        /// <param name="probability">The probability mask, or <see langword="null"/>; stays owned by the caller.</param>
                        /// <returns>A new owned reference.</returns>
                        /// <exception cref="PythonException">Thrown when sampling raises.</exception>
                        internal static PyObject sample(PyObject space, PyObject? mask = null, PyObject? probability = null) {
                            using PyObject method = space.GetAttr("sample");
                            if (mask is null && probability is null)
                                return method.Invoke();
                            using var arguments = new PyDict();
                            if (mask != null)
                                arguments["mask"] = mask;
                            if (probability != null)
                                arguments["probability"] = probability;
                            return method.Invoke(Array.Empty<PyObject>(), arguments);
                        }

                        /// <summary><c>space.seed(seed)</c> with a seed of any form (<c>None</c>, an int, a list, a dict).</summary>
                        /// <param name="space">The space.</param>
                        /// <param name="seed">The seed; stays owned by the caller.</param>
                        /// <returns>A new owned reference to the seeds used.</returns>
                        /// <exception cref="PythonException">Thrown when seeding raises.</exception>
                        internal static PyObject seed(PyObject space, PyObject seed) {
                            using PyObject method = space.GetAttr("seed");
                            return method.Invoke(seed);
                        }

                        /// <summary><c>space.to_jsonable(sample_n)</c>.</summary>
                        /// <param name="space">The space.</param>
                        /// <param name="samples">A list of samples; stays owned by the caller.</param>
                        /// <returns>A new owned reference to the JSON data.</returns>
                        /// <exception cref="PythonException">Thrown when the conversion raises.</exception>
                        internal static PyObject to_jsonable(PyObject space, PyObject samples) {
                            using PyObject method = space.GetAttr("to_jsonable");
                            return method.Invoke(samples);
                        }

                        /// <summary><c>space.from_jsonable(sample_n)</c>.</summary>
                        /// <param name="space">The space.</param>
                        /// <param name="json">The JSON data; stays owned by the caller.</param>
                        /// <returns>A new owned reference to the list of samples.</returns>
                        /// <exception cref="PythonException">Thrown when the conversion raises.</exception>
                        internal static PyObject from_jsonable(PyObject space, PyObject json) {
                            using PyObject method = space.GetAttr("from_jsonable");
                            return method.Invoke(json);
                        }

                        /// <summary><c>space.is_np_flattenable</c>.</summary>
                        /// <param name="space">The space.</param>
                        /// <returns>A new owned reference to the boolean.</returns>
                        /// <exception cref="PythonException">Thrown when the property raises.</exception>
                        internal static PyObject is_np_flattenable(PyObject space) => space.GetAttr("is_np_flattenable");

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

            /// <summary><c>gymnasium.vector</c>.</summary>
            internal static class vector {
                /// <summary><c>gymnasium.vector.utils</c>: batching spaces and their samples.</summary>
                internal static class utils {
                    /// <summary>The cached module; never dispose it.</summary>
                    private static PyObject Module => PythonRuntime.Import("gymnasium.vector.utils");

                    /// <summary><c>batch_space(space, n)</c>.</summary>
                    /// <param name="space">The space; stays owned by the caller.</param>
                    /// <param name="n">The number of copies.</param>
                    /// <returns>A new owned reference to the batched space.</returns>
                    /// <exception cref="PythonException">Thrown when batching raises.</exception>
                    internal static PyObject batch_space(PyObject space, long n) {
                        using PyObject function = Module.GetAttr("batch_space");
                        using var count = new PyInt(n);
                        return function.Invoke(space, count);
                    }

                    /// <summary><c>batch_differing_spaces(spaces)</c>.</summary>
                    /// <param name="spaces">A list of spaces; stays owned by the caller.</param>
                    /// <returns>A new owned reference to the batched space.</returns>
                    /// <exception cref="PythonException">Thrown when batching raises.</exception>
                    internal static PyObject batch_differing_spaces(PyObject spaces) {
                        using PyObject function = Module.GetAttr("batch_differing_spaces");
                        return function.Invoke(spaces);
                    }

                    /// <summary><c>iterate(space, items)</c>, an iterator; drain it with <see cref="builtins.list"/>.</summary>
                    /// <param name="space">The batched space; stays owned by the caller.</param>
                    /// <param name="items">The batched sample; stays owned by the caller.</param>
                    /// <returns>A new owned reference to the iterator.</returns>
                    /// <exception cref="PythonException">Thrown when the space cannot iterate.</exception>
                    internal static PyObject iterate(PyObject space, PyObject items) {
                        using PyObject function = Module.GetAttr("iterate");
                        return function.Invoke(space, items);
                    }

                    /// <summary><c>concatenate(space, items, out)</c>.</summary>
                    /// <param name="space">The single space; stays owned by the caller.</param>
                    /// <param name="items">The samples, a list or tuple; stays owned by the caller.</param>
                    /// <param name="out">The destination from <see cref="create_empty_array"/>; stays owned by the caller.</param>
                    /// <returns>A new owned reference to the batch.</returns>
                    /// <exception cref="PythonException">Thrown when concatenating raises.</exception>
                    internal static PyObject concatenate(PyObject space, PyObject items, PyObject @out) {
                        using PyObject function = Module.GetAttr("concatenate");
                        return function.Invoke(space, items, @out);
                    }

                    /// <summary><c>create_empty_array(space, n)</c> with Gymnasium's default <c>np.zeros</c>.</summary>
                    /// <param name="space">The single space; stays owned by the caller.</param>
                    /// <param name="n">The number of envs.</param>
                    /// <returns>A new owned reference to the empty batch.</returns>
                    /// <exception cref="PythonException">Thrown when the space has no empty form.</exception>
                    internal static PyObject create_empty_array(PyObject space, long n) {
                        using PyObject function = Module.GetAttr("create_empty_array");
                        using var count = new PyInt(n);
                        return function.Invoke(space, count);
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

                /// <summary><c>gymnasium.utils.step_api_compatibility</c>: conversions between the done and terminated/truncated step APIs.</summary>
                internal static class step_api_compatibility {
                    /// <summary>The cached module; never dispose it.</summary>
                    private static PyObject Module => PythonRuntime.Import("gymnasium.utils.step_api_compatibility");

                    /// <summary><c>convert_to_done_step_api(step_returns)</c> for a single env.</summary>
                    /// <param name="step_returns">A 5-tuple (or a 4-tuple, passed through); stays owned by the caller, and its info is edited in place.</param>
                    /// <returns>A new owned reference to the 4-tuple.</returns>
                    /// <exception cref="PythonException">Thrown when the conversion raises.</exception>
                    internal static PyObject convert_to_done_step_api(PyObject step_returns) {
                        using PyObject function = Module.GetAttr("convert_to_done_step_api");
                        return function.Invoke(step_returns);
                    }

                    /// <summary><c>convert_to_terminated_truncated_step_api(step_returns)</c> for a single env.</summary>
                    /// <param name="step_returns">A 4-tuple (or a 5-tuple, passed through); stays owned by the caller, and its info is edited in place.</param>
                    /// <returns>A new owned reference to the 5-tuple.</returns>
                    /// <exception cref="PythonException">Thrown when the conversion raises.</exception>
                    internal static PyObject convert_to_terminated_truncated_step_api(PyObject step_returns) {
                        using PyObject function = Module.GetAttr("convert_to_terminated_truncated_step_api");
                        return function.Invoke(step_returns);
                    }
                }
            }

            /// <summary><c>gymnasium.envs</c>, the env families.</summary>
            internal static class envs {
                /// <summary><c>gymnasium.envs.registration</c>: the id grammar and the spec dataclasses.</summary>
                internal static class registration {
                    /// <summary>The cached module; never dispose it.</summary>
                    private static PyObject Module => PythonRuntime.Import("gymnasium.envs.registration");

                    /// <summary><c>parse_env_id(env_id)</c>.</summary>
                    /// <param name="env_id">The id.</param>
                    /// <returns>A new owned reference to <c>(namespace, name, version)</c>.</returns>
                    /// <exception cref="PythonException">Thrown for a malformed id (<c>gymnasium.error.Error</c>).</exception>
                    internal static PyObject parse_env_id(string env_id) {
                        using PyObject function = Module.GetAttr("parse_env_id");
                        using var id = new PyString(env_id);
                        return function.Invoke(id);
                    }

                    /// <summary><c>get_env_id(ns, name, version)</c>.</summary>
                    /// <param name="ns">The namespace, or <see langword="null"/>.</param>
                    /// <param name="name">The name.</param>
                    /// <param name="version">The version, or <see langword="null"/>.</param>
                    /// <returns>A new owned reference to the id.</returns>
                    /// <exception cref="PythonException">Thrown when the call fails.</exception>
                    internal static PyObject get_env_id(string? ns, string name, long? version) {
                        using PyObject function = Module.GetAttr("get_env_id");
                        using PyObject nsValue = Oracle.PyConvert.ToPython(ns);
                        using var nameValue = new PyString(name);
                        using PyObject versionValue = Oracle.PyConvert.ToPython(version);
                        return function.Invoke(nsValue, nameValue, versionValue);
                    }

                    /// <summary><c>WrapperSpec(name, entry_point, kwargs)</c>.</summary>
                    /// <param name="name">The wrapper's class name.</param>
                    /// <param name="entry_point">Its location.</param>
                    /// <param name="kwargs">Its arguments (a dict) or <c>None</c>; stays owned by the caller.</param>
                    /// <returns>A new owned reference to the dataclass.</returns>
                    /// <exception cref="PythonException">Thrown when the construction fails.</exception>
                    internal static PyObject WrapperSpec(string name, string entry_point, PyObject kwargs) {
                        using PyObject constructor = Module.GetAttr("WrapperSpec");
                        using var nameValue = new PyString(name);
                        using var entryPoint = new PyString(entry_point);
                        return constructor.Invoke(nameValue, entryPoint, kwargs);
                    }

                    /// <summary>
                    ///     <c>EnvSpec</c> instance members. The constructor, kwarg-heavy, is a snippet
                    ///     (<see cref="evalStmts"/>) in the cells that build specs.
                    /// </summary>
                    internal static class EnvSpec {
                        /// <summary><c>spec.pprint(disable_print=True, include_entry_points=..., print_all=...)</c>.</summary>
                        /// <param name="spec">The spec.</param>
                        /// <param name="include_entry_points">Also print the entry points.</param>
                        /// <param name="print_all">Print every field, defaults included.</param>
                        /// <returns>A new owned reference to the text (<c>disable_print=True</c> returns it).</returns>
                        /// <exception cref="PythonException">Thrown when printing fails.</exception>
                        internal static PyObject pprint(PyObject spec, bool include_entry_points = false, bool print_all = false) {
                            using PyObject method = spec.GetAttr("pprint");
                            using var arguments = new PyDict();
                            using PyObject disable = true.ToPython();
                            using PyObject entryPoints = include_entry_points.ToPython();
                            using PyObject all = print_all.ToPython();
                            arguments["disable_print"] = disable;
                            arguments["include_entry_points"] = entryPoints;
                            arguments["print_all"] = all;
                            return method.Invoke(Array.Empty<PyObject>(), arguments);
                        }

                        /// <summary><c>spec.to_json()</c>.</summary>
                        /// <param name="spec">The spec.</param>
                        /// <returns>A new owned reference to the JSON text.</returns>
                        /// <exception cref="PythonException">Thrown when serializing fails (a callable field, an unserializable kwarg).</exception>
                        internal static PyObject to_json(PyObject spec) {
                            using PyObject method = spec.GetAttr("to_json");
                            return method.Invoke();
                        }

                        /// <summary><c>EnvSpec.from_json(json_env_spec)</c>.</summary>
                        /// <param name="json_env_spec">The JSON text.</param>
                        /// <returns>A new owned reference to the spec.</returns>
                        /// <exception cref="PythonException">Thrown for malformed JSON or fields that do not make a spec.</exception>
                        internal static PyObject from_json(string json_env_spec) {
                            using PyObject type = Module.GetAttr("EnvSpec");
                            using PyObject method = type.GetAttr("from_json");
                            using var text = new PyString(json_env_spec);
                            return method.Invoke(text);
                        }
                    }

                    /// <summary><c>pprint_registry(print_registry, num_cols=..., exclude_namespaces=..., disable_print=True)</c>.</summary>
                    /// <param name="print_registry">The registry dict to print (stays owned by the caller).</param>
                    /// <param name="num_cols">Ids per line.</param>
                    /// <param name="exclude_namespaces">Namespaces to leave out, or <see langword="null"/>.</param>
                    /// <returns>A new owned reference to the text.</returns>
                    /// <exception cref="PythonException">Thrown when printing fails.</exception>
                    internal static PyObject pprint_registry(PyObject print_registry, long num_cols = 3, IReadOnlyList<string>? exclude_namespaces = null) {
                        using PyObject function = Module.GetAttr("pprint_registry");
                        using var arguments = new PyDict();
                        using PyObject columns = Oracle.PyConvert.ToPython(num_cols);
                        using PyObject excluded = Oracle.PyConvert.ToPython(exclude_namespaces?.Select(ns => (object?)ns).ToList());
                        using PyObject disable = true.ToPython();
                        arguments["num_cols"] = columns;
                        arguments["exclude_namespaces"] = excluded;
                        arguments["disable_print"] = disable;
                        return function.Invoke(new[] { print_registry }, arguments);
                    }
                }

                /// <summary><c>gymnasium.envs.classic_control</c>.</summary>
                internal static class classic_control {
                    /// <summary>
                    ///     <c>classic_control.CartPoleEnv(sutton_barto_reward=..., render_mode=...)</c>: the bare env, as its constructor
                    ///     builds it (no spec, no wrappers); its members are under <c>classic_control.cartpole.CartPoleEnv</c>.
                    /// </summary>
                    /// <param name="render_mode">The render mode, or <see langword="null"/>.</param>
                    /// <param name="sutton_barto_reward">Sutton and Barto's reward scheme.</param>
                    /// <returns>A new owned reference to the env.</returns>
                    /// <exception cref="PythonException">Thrown when the construction fails.</exception>
                    internal static PyObject CartPoleEnv(string? render_mode = null, bool sutton_barto_reward = false) {
                        using PyObject constructor = PythonRuntime.Import("gymnasium.envs.classic_control").GetAttr("CartPoleEnv");
                        using var arguments = new PyDict();
                        using PyObject mode = Oracle.PyConvert.ToPython(render_mode);
                        using PyObject reward = sutton_barto_reward.ToPython();
                        arguments["render_mode"] = mode;
                        arguments["sutton_barto_reward"] = reward;
                        return constructor.Invoke(Array.Empty<PyObject>(), arguments);
                    }

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
