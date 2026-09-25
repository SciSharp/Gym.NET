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
        ///     Runs a Gym.NET call and describes what it did. The returned value is described and then disposed when it is
        ///     <see cref="IDisposable"/> and <paramref name="disposeResult"/> is set, since the caller only keeps the facts.
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
                if (disposeResult && result is IDisposable disposable)
                    disposable.Dispose();
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
    ///     built-ins map to NumSharp's ports of them (decision D17), <c>gymnasium.error</c> types to their ports in
    ///     <c>Gym.Exceptions</c>, and built-ins NumSharp lacks to <c>Gym.Exceptions</c> as well.
    /// </summary>
    public static class ExceptionMap {
        /// <summary>Python built-ins that NumSharp ports, by the name <c>ValueFacts.PythonTypeName</c> prints for them.</summary>
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
            // Not in NumSharp: the port plan puts them in Gym.Exceptions (D17), where the ports will define them.
            ["AssertionError"] = "Gym.Exceptions.AssertionError",
            ["NotImplementedError"] = "Gym.Exceptions.NotImplementedError",
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
