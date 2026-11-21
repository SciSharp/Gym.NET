using System;
using System.Collections.Generic;
using Gym.Tests.Parity.Interop;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace Gym.Tests.Parity.Oracle {
    /// <summary>An exception one side raised, reduced to what parity compares: its type and its message.</summary>
    /// <param name="Type">
    ///     The type: for Python as <c>ValueFacts.PythonTypeName</c> prints it (<c>ValueError</c>, <c>gymnasium.error.Error</c>), for
    ///     .NET its full name (<c>NumSharp.ValueError</c>).
    /// </param>
    /// <param name="Message">The message: Python's <c>str(exception)</c>, .NET's <see cref="Exception.Message"/>.</param>
    public sealed record RaisedError(string Type, string Message) {
        /// <summary>
        ///     Describes a .NET exception. For a NumSharp exception built on <see cref="ArgumentException"/> (its
        ///     <c>AxisError</c>, a <c>ValueError</c> naming a parameter), the message is NumPy's text alone: .NET appends
        ///     <c> (Parameter 'axis2')</c> and, for an <see cref="ArgumentOutOfRangeException"/>, an <c>Actual value was 1.</c>
        ///     line, which NumSharp's own docs set apart from NumPy's message.
        /// </summary>
        /// <param name="exception">The exception.</param>
        /// <returns>Its full type name and message.</returns>
        public static RaisedError FromDotNet(Exception exception) {
            string message = exception.Message;
            if (exception is ArgumentException { ParamName.Length: > 0 } argument && exception is NumSharp.INumSharpException) {
                // Undo exactly what ArgumentOutOfRangeException.Message and ArgumentException.Message append, in that order.
                if (argument is ArgumentOutOfRangeException { ActualValue: { } actual }) {
                    string actualLine = Environment.NewLine + $"Actual value was {actual}.";
                    if (message.EndsWith(actualLine, StringComparison.Ordinal))
                        message = message.Substring(0, message.Length - actualLine.Length);
                }

                string parameter = $" (Parameter '{argument.ParamName}')";
                if (message.EndsWith(parameter, StringComparison.Ordinal))
                    message = message.Substring(0, message.Length - parameter.Length);
            }

            return new RaisedError(exception.GetType().FullName ?? exception.GetType().Name, message);
        }

        /// <inheritdoc />
        public override string ToString() => $"{Type}: {Message}";
    }

    /// <summary>
    ///     The result of one call on one side: the value it returned, or the exception it raised. Errors are outcomes to
    ///     compare, not test failures: Gymnasium raising <c>ValueError("...")</c> must be matched by Gym.NET throwing
    ///     <c>NumSharp.ValueError</c> with the same message.
    /// </summary>
    public sealed class Outcome {
        /// <summary>Creates an outcome; exactly one of the arguments is non-null.</summary>
        /// <param name="value">The returned value.</param>
        /// <param name="raised">The raised exception.</param>
        private Outcome(ValueFacts? value, RaisedError? raised) {
            Value = value;
            Raised = raised;
        }

        /// <summary>The returned value, or <see langword="null"/> when the call raised.</summary>
        public ValueFacts? Value { get; }

        /// <summary>The raised exception, or <see langword="null"/> when the call returned.</summary>
        public RaisedError? Raised { get; }

        /// <summary>An outcome that returned <paramref name="value"/>.</summary>
        /// <param name="value">The described value.</param>
        /// <returns>The outcome.</returns>
        public static Outcome Returned(ValueFacts value) => new(value, null);

        /// <summary>An outcome that raised <paramref name="error"/>.</summary>
        /// <param name="error">The described exception.</param>
        /// <returns>The outcome.</returns>
        public static Outcome Threw(RaisedError error) => new(null, error);

        /// <summary>
        ///     Runs a Gym.NET call and describes what it did. When <paramref name="disposeResult"/> is set, the returned value
        ///     is disposed after it is described, since the caller only keeps the facts: an <see cref="IDisposable"/> itself,
        ///     and the arrays inside a composite sample (a tuple's <see langword="object"/>?[], a dict's <c>Info</c>, nested).
        /// </summary>
        /// <param name="call">The call; its exceptions are caught and described.</param>
        /// <param name="disposeResult">
        ///     Whether the call's result is owned by the caller and must be released after it is described (a fresh
        ///     <c>Sample()</c> array); pass <see langword="false"/> for results that alias state the callee still uses.
        /// </param>
        /// <returns>The outcome.</returns>
        public static Outcome Capture(Func<object?> call, bool disposeResult = true) {
            object? result;
            try {
                result = call();
            } catch (Exception exception) when (exception is not UnitTestAssertException) {
                // Test-framework assertions are not the callee's errors: let them fail the test instead of becoming data.
                return Threw(RaisedError.FromDotNet(exception));
            }

            try {
                return Returned(ValueFacts.Of(result));
            } finally {
                if (disposeResult)
                    DisposeOwned(result, depth: 0);
            }
        }

        /// <summary>
        ///     Releases a Gym.NET value a cell owns (a composite sample, a list of samples) the way <see cref="Capture"/> releases
        ///     a result: its arrays and other disposables, nested ones included.
        /// </summary>
        /// <param name="value">The value; <see langword="null"/> is ignored.</param>
        internal static void Release(object? value) => DisposeOwned(value, depth: 0);

        /// <summary>
        ///     Disposes a described result: an <see cref="IDisposable"/> (an array, a <c>GraphInstance</c>) itself, and the
        ///     items of materialized containers (arrays, lists, tuples, dictionaries) recursively.
        /// </summary>
        /// <param name="value">The result or one of its items.</param>
        /// <param name="depth">How deep the walk is; containers below 16 levels are left to the finalizer, so a self-containing value cannot recurse forever.</param>
        /// <remarks>
        ///     Lazy sequences are not walked: enumerating one again would run its query a second time (creating new arrays
        ///     only to dispose them) rather than reach the items <see cref="ValueFacts.Of"/> described.
        /// </remarks>
        private static void DisposeOwned(object? value, int depth) {
            if (depth > 16)
                return;
            switch (value) {
                case IDisposable disposable:
                    disposable.Dispose();
                    break;
                case string:
                    break;
                case System.Collections.IDictionary dictionary:
                    foreach (System.Collections.DictionaryEntry entry in dictionary)
                        DisposeOwned(entry.Value, depth + 1);
                    break;
                case ICollection<KeyValuePair<string, object?>> entries:
                    foreach (KeyValuePair<string, object?> entry in entries)
                        DisposeOwned(entry.Value, depth + 1);
                    break;
                case System.Runtime.CompilerServices.ITuple tuple:
                    for (int i = 0; i < tuple.Length; i++)
                        DisposeOwned(tuple[i], depth + 1);
                    break;
                case System.Collections.ICollection items:
                    foreach (object? item in items)
                        DisposeOwned(item, depth + 1);
                    break;
            }
        }

        /// <summary>
        ///     Runs a Gymnasium call (a <see cref="Python"/> facade row) under the GIL and describes what it did: the returned
        ///     value, which is disposed after it is described, or the Python exception it raised, reduced to its type (as
        ///     <see cref="ValueFacts.PythonTypeName"/> prints it) and <c>str(exception)</c>.
        /// </summary>
        /// <param name="call">The call; it returns a new reference that this method disposes.</param>
        /// <returns>The outcome.</returns>
        public static Outcome CapturePython(Func<PyObject> call) {
            return PythonRuntime.Run(() => {
                PyObject result;
                try {
                    result = call();
                } catch (PythonException exception) {
                    // Type is the exception's own field, not a new reference: disposing it would break the exception object.
                    return Threw(new RaisedError(ValueFacts.QualifiedName(exception.Type), exception.Message));
                }

                using (result)
                    return Returned(ValueFacts.FromPython(result));
            });
        }

        /// <summary>The returned value, or a test failure naming the exception when the call raised.</summary>
        /// <param name="context">What was called, for the failure message.</param>
        /// <returns>The value.</returns>
        /// <exception cref="AssertFailedException">Thrown when the call raised.</exception>
        public ValueFacts Expect(string context) {
            if (Raised != null)
                Assert.Fail($"{context} raised {Raised}.");
            return Value!;
        }

        /// <summary>Renders the outcome for a report.</summary>
        /// <returns>The value, or <c>raised Type: message</c>.</returns>
        public override string ToString() => Raised != null ? $"raised {Raised}" : Value!.Render();
    }

    /// <summary>
    ///     Which .NET exception type a Python exception must correspond to, following the port plan's §3.4: Python's
    ///     built-ins map to NumSharp's counterparts (decisions D17 and DR-02), <c>gymnasium.error</c> types to the classes of
    ///     the same names in <c>Gym.Exceptions</c>, and the built-ins NumSharp still lacks (open item N-3) to
    ///     <c>Gym.Exceptions</c> as well, until NumSharp adds them.
    /// </summary>
    public static class ExceptionMap {
        /// <summary>Python built-ins that NumSharp has counterparts for, by the name <c>ValueFacts.PythonTypeName</c> prints for them.</summary>
        private static readonly Dictionary<string, string> BuiltIns = new(StringComparer.Ordinal) {
            ["ValueError"] = "NumSharp.ValueError",
            ["TypeError"] = "NumSharp.TypeError",
            ["KeyError"] = "NumSharp.KeyError",
            ["IndexError"] = "NumSharp.IndexError",
            ["AttributeError"] = "NumSharp.AttributeError",
            ["RuntimeError"] = "NumSharp.RuntimeError",
            ["NameError"] = "NumSharp.NameError",
            // NumSharp 0.81.0 has Python's classes for these two (DR-76): NumPy's OverflowError ("high - low range exceeds
            // valid bounds") and ZeroDivisionError arrive as them, and Gym.NET throws them for Gymnasium's, pygame's and
            // CPython's, so one Python type keeps one .NET type. (0.80.0 threw .NET's OverflowException and
            // DivideByZeroException, which Gym.NET then matched.)
            ["OverflowError"] = "NumSharp.OverflowError",
            ["ZeroDivisionError"] = "NumSharp.ZeroDivisionError",
            // NumSharp 0.80.0 added AssertionError, so Gym.NET throws NumSharp's (DR-02).
            ["AssertionError"] = "NumSharp.AssertionError",
            // NumSharp's since 0.81.0 (DR-76), where Gym.Exceptions held stop-gaps until then (N-3).
            ["NotImplementedError"] = "NumSharp.NotImplementedError",
            ["ImportError"] = "NumSharp.ImportError",
            ["ModuleNotFoundError"] = "NumSharp.ModuleNotFoundError",
            // json.loads' error is a ValueError subclass; EnvSpec.FromJson raises its base, as NumSharp has no JSONDecodeError.
            ["json.decoder.JSONDecodeError"] = "NumSharp.ValueError",
            // A ufunc without a loop for the operands' dtypes (np.sign of booleans in BipedalWalker's step) raises this
            // TypeError subclass; NumSharp raises its TypeError with the same text, so Gym.NET raises that base too.
            ["numpy._core._exceptions._UFuncNoLoopError"] = "NumSharp.TypeError",
            // Not built-ins, but errors of Gymnasium's play module and of the libraries under it, whose .NET counterparts
            // live where Gym.NET's play does (the port plan's §3.4 has no rule for them).
            ["gymnasium.utils.play.MissingKeysToAction"] = "Gym.Utils.MissingKeysToAction",
            ["numpy.exceptions.AxisError"] = "NumSharp.AxisError",
            ["pygame.error"] = "Gym.Rendering.SdlError",
        };

        /// <summary>The .NET type a Python exception type must correspond to.</summary>
        /// <param name="pythonType">The Python type as <c>ValueFacts.PythonTypeName</c> prints it.</param>
        /// <returns>
        ///     The expected full .NET type name, or <see langword="null"/> when §3.4 has no rule for the type, in which case
        ///     only the messages can be compared.
        /// </returns>
        public static string? DotNetTypeFor(string pythonType) {
            if (BuiltIns.TryGetValue(pythonType, out string? mapped))
                return mapped;
            const string gymnasiumErrors = "gymnasium.error.";
            if (pythonType.StartsWith(gymnasiumErrors, StringComparison.Ordinal))
                return "Gym.Exceptions." + pythonType.Substring(gymnasiumErrors.Length);
            return null;
        }

        /// <summary>
        ///     The Python exception type a .NET exception stands for: the reverse of <see cref="DotNetTypeFor"/>, which the
        ///     reverse bridge uses to raise in Python what a Gym.NET env threw, so Gymnasium's tools see an
        ///     <c>AssertionError</c> where a Gymnasium env would fail an <c>assert</c> (<c>NumSharp.AssertionError</c>), and
        ///     <c>gymnasium.error.ResetNeeded</c> where it would raise that (<c>Gym.Exceptions.ResetNeeded</c>).
        /// </summary>
        /// <param name="exception">The exception.</param>
        /// <returns>
        ///     The Python type's name (<c>AssertionError</c>, <c>gymnasium.error.Error</c>), taken from the exception's own type
        ///     or else its nearest base type that maps; <see langword="null"/> when none does (an <see cref="ArgumentException"/>,
        ///     an <see cref="ObjectDisposedException"/>), so the caller lets the CLR exception through as it is.
        /// </returns>
        /// <remarks>
        ///     A <c>Gym.Exceptions</c> name maps to <c>gymnasium.error</c> by name alone, and Gym.NET keeps classes Gymnasium
        ///     does not have (the legacy API's errors), so the caller must check that the Python type exists. Only built-in names
        ///     reverse from <see cref="BuiltIns"/>: <c>json.decoder.JSONDecodeError</c> shares <c>NumSharp.ValueError</c> with
        ///     <c>ValueError</c>, and <c>ValueError</c> is what a Gym.NET <c>ValueError</c> stands for.
        /// </remarks>
        public static string? PythonTypeFor(Exception exception) {
            for (Type? type = exception.GetType(); type != null && type != typeof(Exception); type = type.BaseType) {
                string fullName = type.FullName ?? type.Name;
                foreach ((string python, string dotnet) in BuiltIns) {
                    if (string.Equals(dotnet, fullName, StringComparison.Ordinal) && !python.Contains('.'))
                        return python;
                }

                const string gymErrors = "Gym.Exceptions.";
                if (fullName.StartsWith(gymErrors, StringComparison.Ordinal))
                    return "gymnasium.error." + fullName.Substring(gymErrors.Length);
            }

            return null;
        }
    }
}
