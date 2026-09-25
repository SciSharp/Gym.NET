using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Tests.Parity.Session;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Oracle {
    /// <summary>
    ///     A Python expression passed as an argument: <c>PyGym.Evaluate("...")</c>'s escape hatch at argument position, for
    ///     values C# cannot spell the way Gymnasium expects them (<c>np.array([1, 0, 1], dtype=np.int8)</c>).
    /// </summary>
    /// <param name="Code">The expression, evaluated by <c>gym_oracle.evaluate</c> (so <c>np</c>, <c>spaces</c>, ... are bound).</param>
    public sealed record PyExpression(string Code) {
        /// <inheritdoc />
        public override string ToString() => Code;
    }

    /// <summary>
    ///     Live Gymnasium, as the tests see it: builds Python objects from expressions, describes Python values, and converts
    ///     .NET arguments to Python (NumSharp's interop plays NumPy's part for arrays). The Python-side counterpart of
    ///     NumSharp's <c>Python.np</c> test builders.
    /// </summary>
    /// <remarks>Every member takes the GIL itself, so tests call them without holding it.</remarks>
    public static class PyGym {
        /// <summary>Evaluates an expression and keeps the result alive as a handle, for objects that are used repeatedly (a space, a generator).</summary>
        /// <param name="expression">A Python expression; <c>np</c>, <c>gymnasium</c>, <c>gym</c>, <c>spaces</c> and <c>seeding</c> are bound.</param>
        /// <returns>The handle; the caller disposes it.</returns>
        /// <exception cref="PythonException">Thrown when the expression raises; use <see cref="Invoke(string, object?[])"/> for calls whose errors are compared.</exception>
        public static PyHandle Evaluate(string expression) {
            using (Py.GIL()) {
                using var code = new PyString(expression);
                return new PyHandle(GymPythonSession.Oracle.InvokeMethod("evaluate", code), expression);
            }
        }

        /// <summary>Evaluates an expression and describes the result.</summary>
        /// <param name="expression">A Python expression, as for <see cref="Evaluate"/>.</param>
        /// <returns>The result's description, or the exception it raised.</returns>
        public static Outcome Invoke(string expression) {
            using (Py.GIL()) {
                using var code = new PyString(expression);
                using PyObject evaluate = GymPythonSession.Oracle.GetAttr("evaluate");
                using PyObject result = GymPythonSession.Oracle.InvokeMethod("invoke", evaluate, code);
                return ReadOutcome(result);
            }
        }

        /// <summary>Calls a Python callable given by an expression with converted .NET arguments, and describes the result.</summary>
        /// <param name="callable">An expression that evaluates to the callable, for example <c>seeding.np_random</c>.</param>
        /// <param name="arguments">The arguments, converted by <see cref="ToPython"/>.</param>
        /// <returns>The result's description, or the exception the call raised.</returns>
        /// <exception cref="PythonException">Thrown when <paramref name="callable"/> itself fails to evaluate.</exception>
        public static Outcome Invoke(string callable, params object?[] arguments) {
            using PyHandle function = Evaluate(callable);
            return function.Call(arguments);
        }

        /// <summary>Describes a Python value with <c>gym_oracle.describe</c>. Must run under the GIL.</summary>
        /// <param name="value">The value; it stays owned by the caller.</param>
        /// <returns>The description.</returns>
        public static ValueFacts Describe(PyObject value) {
            using PyObject descriptor = GymPythonSession.Oracle.InvokeMethod("describe", value);
            return ValueFacts.FromDescriptor(descriptor);
        }

        /// <summary>
        ///     Converts a .NET argument into the Python value a Gymnasium user would pass. Must run under the GIL; the result is
        ///     a new reference the caller disposes. A <see cref="PyHandle"/> is not converted here but borrowed by
        ///     <see cref="PyArguments"/>, because pythonnet offers no public way to take a second reference to an object.
        /// </summary>
        /// <param name="argument">
        ///     <see langword="null"/> (<c>None</c>), <see cref="bool"/>, an integer (<c>int</c>), <see cref="BigInteger"/>
        ///     (<c>int</c>), <see cref="double"/> or <see cref="float"/> (<c>float</c>), <see cref="string"/>, an
        ///     <see cref="NDArray"/> (a numpy copy with the same dtype and bytes), or a <see cref="PyExpression"/> (evaluated).
        /// </param>
        /// <returns>The Python value.</returns>
        /// <exception cref="ArgumentException">Thrown for a type with no defined conversion, so a test never passes something by accident.</exception>
        /// <remarks>
        ///     Arrays cross as independent copies (<c>ToNumpyCopy</c>): Gymnasium may keep or mutate what it is given (an env
        ///     stores its injected state), and a shared view would let that reach back into the .NET side's array.
        /// </remarks>
        internal static PyObject ToPython(object? argument) {
            switch (argument) {
                case null: {
                    // A fresh reference to None, rather than PyObject.None: pythonnet shares that instance process-wide, and
                    // the caller's using would dispose it.
                    using var none = new PyString("None");
                    return GymPythonSession.Oracle.InvokeMethod("evaluate", none);
                }
                case bool flag:
                    return flag.ToPython();
                case sbyte or byte or short or ushort or int or uint or long:
                    return new PyInt(Convert.ToInt64(argument, System.Globalization.CultureInfo.InvariantCulture));
                case ulong unsigned:
                    return new PyInt(unsigned);
                case BigInteger big:
                    return new PyInt(big);
                case double d:
                    return new PyFloat(d);
                case float f:
                    return new PyFloat(f);
                case string s:
                    return new PyString(s);
                case NDArray array:
                    return array.ToNumpyCopy();
                case PyExpression expression: {
                    using var code = new PyString(expression.Code);
                    return GymPythonSession.Oracle.InvokeMethod("evaluate", code);
                }
                default:
                    throw new ArgumentException($"No Python conversion is defined for {argument.GetType().FullName}.", nameof(argument));
            }
        }

        /// <summary>Reads the <c>("ok", descriptor)</c> / <c>("raised", type, message)</c> tuple of <c>gym_oracle.invoke</c>. Must run under the GIL.</summary>
        /// <param name="result">The tuple.</param>
        /// <returns>The outcome.</returns>
        /// <exception cref="InvalidOperationException">Thrown for a tag this reader does not know.</exception>
        internal static Outcome ReadOutcome(PyObject result) {
            using PyObject tag = result[0];
            switch (tag.As<string>()) {
                case "ok": {
                    using PyObject descriptor = result[1];
                    return Outcome.Returned(ValueFacts.FromDescriptor(descriptor));
                }
                case "raised": {
                    using PyObject type = result[1];
                    using PyObject message = result[2];
                    return Outcome.Threw(new RaisedError(type.As<string>(), message.As<string>()));
                }
                default:
                    throw new InvalidOperationException("gym_oracle.invoke returned an unknown tag; Oracle/PyGym.cs and Oracle/gym_oracle.py must change together.");
            }
        }
    }

    /// <summary>
    ///     A Python object kept alive across calls (a space, a generator, an env), with its calls described as
    ///     <see cref="Outcome"/>s.
    /// </summary>
    /// <remarks>Every member takes the GIL itself. Dispose it, which also releases the Python object under the GIL.</remarks>
    public sealed class PyHandle : IDisposable {
        /// <summary>The object, or <see langword="null"/> once disposed.</summary>
        private PyObject? _object;

        /// <summary>Takes ownership of a Python object.</summary>
        /// <param name="value">The object; the handle disposes it.</param>
        /// <param name="description">How the object was made, for reports.</param>
        internal PyHandle(PyObject value, string description) {
            _object = value;
            Description = description;
        }

        /// <summary>How the object was made (its expression), for reports.</summary>
        public string Description { get; }

        /// <summary>The Python object, for harness code running under the GIL.</summary>
        /// <exception cref="ObjectDisposedException">Thrown after <see cref="Dispose"/>.</exception>
        internal PyObject Object => _object ?? throw new ObjectDisposedException(Description);

        /// <summary>Calls a method with converted arguments and describes the result or the exception.</summary>
        /// <param name="method">The method's name.</param>
        /// <param name="arguments">The arguments, converted by <see cref="PyGym.ToPython"/>.</param>
        /// <returns>The outcome.</returns>
        public Outcome Invoke(string method, params object?[] arguments) {
            using (Py.GIL()) {
                using PyObject bound = Object.GetAttr(method);
                return CallUnderGil(bound, arguments);
            }
        }

        /// <summary>
        ///     Evaluates an expression with this object bound as <c>x</c> (and <c>np</c>, <c>spaces</c>, ... as in
        ///     <see cref="PyGym.Evaluate"/>), and describes the result: <c>x.np_random.bit_generator.state</c>,
        ///     <c>np.atleast_1d(x.choice(...))</c>.
        /// </summary>
        /// <param name="expression">The expression.</param>
        /// <returns>The outcome.</returns>
        public Outcome Evaluate(string expression) {
            using (Py.GIL()) {
                using var code = new PyString(expression);
                using var names = new PyDict();
                names.SetItem("x", Object);
                using PyObject evaluate = GymPythonSession.Oracle.GetAttr("evaluate");
                using PyObject invoke = GymPythonSession.Oracle.GetAttr("invoke");
                using PyObject result = invoke.Invoke(new[] { evaluate, code }, names);
                return PyGym.ReadOutcome(result);
            }
        }

        /// <summary>Calls the object itself (it is a callable) with converted arguments.</summary>
        /// <param name="arguments">The arguments, converted by <see cref="PyGym.ToPython"/>.</param>
        /// <returns>The outcome.</returns>
        public Outcome Call(params object?[] arguments) {
            using (Py.GIL())
                return CallUnderGil(Object, arguments);
        }

        /// <summary>Reads an attribute and describes it.</summary>
        /// <param name="attribute">The attribute's name.</param>
        /// <returns>The outcome (an <c>AttributeError</c> is an outcome too).</returns>
        public Outcome Get(string attribute) {
            using (Py.GIL()) {
                using var name = new PyString(attribute);
                using PyObject builtins = Py.Import("builtins");
                using PyObject getattr = builtins.GetAttr("getattr");
                using PyObject result = GymPythonSession.Oracle.InvokeMethod("invoke", getattr, Object, name);
                return PyGym.ReadOutcome(result);
            }
        }

        /// <summary>Calls a method and keeps its raw result as a new handle, for results used repeatedly (<c>env.action_space</c>).</summary>
        /// <param name="method">The method's name.</param>
        /// <param name="arguments">The arguments, converted by <see cref="PyGym.ToPython"/>.</param>
        /// <returns>The result's handle; the caller disposes it.</returns>
        /// <exception cref="PythonException">Thrown when the call raises.</exception>
        public PyHandle InvokeHandle(string method, params object?[] arguments) {
            using (Py.GIL()) {
                using var converted = new PyArguments(null, arguments);
                return new PyHandle(Object.InvokeMethod(method, converted.ToArray()), $"{Description}.{method}(...)");
            }
        }

        /// <summary>Releases the Python object under the GIL. Safe to call twice.</summary>
        public void Dispose() {
            if (_object is null)
                return;
            using (Py.GIL()) {
                _object.Dispose();
                _object = null;
            }
        }

        /// <inheritdoc />
        public override string ToString() => Description;

        /// <summary>Calls <paramref name="callable"/> through <c>gym_oracle.invoke</c>, so an exception comes back as data. Must run under the GIL.</summary>
        /// <param name="callable">The callable.</param>
        /// <param name="arguments">The .NET arguments.</param>
        /// <returns>The outcome.</returns>
        private static Outcome CallUnderGil(PyObject callable, object?[] arguments) {
            // invoke(callable, *arguments): the callable goes first, borrowed, so the argument list never disposes it.
            using var converted = new PyArguments(callable, arguments);
            using PyObject invoke = GymPythonSession.Oracle.GetAttr("invoke");
            using PyObject result = invoke.Invoke(converted.ToArray());
            return PyGym.ReadOutcome(result);
        }
    }

    /// <summary>
    ///     Positional Python arguments converted from .NET values, released together. Converted values are owned and
    ///     disposed; <see cref="PyHandle"/> objects (and an optional leading borrowed object) are borrowed and left alone,
    ///     since pythonnet has no public way to take a second reference to an existing object.
    /// </summary>
    /// <remarks>Create, use and dispose it under the GIL.</remarks>
    internal sealed class PyArguments : IDisposable {
        /// <summary>The arguments in order, each with whether this instance must dispose it.</summary>
        private readonly List<(PyObject Value, bool Owned)> _items = new();

        /// <summary>Converts the arguments; if a conversion throws, the ones already made are released before rethrowing.</summary>
        /// <param name="borrowedFirst">An object to pass first without taking ownership, or <see langword="null"/> for none.</param>
        /// <param name="arguments">The .NET arguments, converted by <see cref="PyGym.ToPython"/> except <see cref="PyHandle"/>s, which are borrowed.</param>
        /// <exception cref="ArgumentException">Thrown for an argument with no defined conversion.</exception>
        public PyArguments(PyObject? borrowedFirst, IEnumerable<object?> arguments) {
            try {
                if (borrowedFirst != null)
                    _items.Add((borrowedFirst, false));
                foreach (object? argument in arguments)
                    _items.Add(argument is PyHandle handle ? (handle.Object, false) : (PyGym.ToPython(argument), true));
            } catch {
                Dispose();
                throw;
            }
        }

        /// <summary>The arguments as the array pythonnet's call methods take.</summary>
        /// <returns>A new array over the same objects; the objects stay owned by this instance.</returns>
        public PyObject[] ToArray() {
            var array = new PyObject[_items.Count];
            for (int i = 0; i < array.Length; i++)
                array[i] = _items[i].Value;
            return array;
        }

        /// <summary>Disposes the owned arguments. Must run under the GIL.</summary>
        public void Dispose() {
            foreach ((PyObject value, bool owned) in _items) {
                if (owned)
                    value.Dispose();
            }

            _items.Clear();
        }
    }
}
