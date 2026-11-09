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
        /// <summary>Describes a .NET exception.</summary>
        /// <param name="exception">The exception.</param>
        /// <returns>Its full type name and message.</returns>
        public static RaisedError FromDotNet(Exception exception) {
            return new RaisedError(exception.GetType().FullName ?? exception.GetType().Name, exception.Message);
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
            // NumSharp throws .NET's own OverflowException for NumPy's OverflowError ("high - low range exceeds valid
            // bounds"), so Gym.NET does the same for Gymnasium's and pygame's, and one Python type keeps one .NET type.
            ["OverflowError"] = "System.OverflowException",
            // .NET's own counterpart, which PyMath throws with Python's messages, as OverflowError maps to OverflowException.
            ["ZeroDivisionError"] = "System.DivideByZeroException",
            // NumSharp 0.80.0 added AssertionError, so Gym.NET throws NumSharp's (DR-02).
            ["AssertionError"] = "NumSharp.AssertionError",
            // Not in NumSharp 0.80.0: Gym.Exceptions keeps them until NumSharp adds them (DR-01's stuck-case workaround,
            // open item N-3); this row then moves to NumSharp's type names.
            ["NotImplementedError"] = "Gym.Exceptions.NotImplementedError",
            ["ImportError"] = "Gym.Exceptions.ImportError",
            ["ModuleNotFoundError"] = "Gym.Exceptions.ModuleNotFoundError",
            // json.loads' error is a ValueError subclass; EnvSpec.FromJson raises its base, as NumSharp has no JSONDecodeError.
            ["json.decoder.JSONDecodeError"] = "NumSharp.ValueError",
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
    }
}
