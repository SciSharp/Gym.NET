using System;
using Gym.Tests.Parity.Oracle;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Pair {
    /// <summary>
    ///     One action in both sides' representations, carrying identical bits: a Python object for Gymnasium's <c>step</c>
    ///     and a .NET value (a boxed <see langword="long"/> or an <see cref="NDArray"/>) for Gym.NET's.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Actions built in C# cross to Python through NumSharp's interop (<c>ToNumpyCopy</c>); actions a Python policy
    ///     picks cross back the same way (<c>ToNDArray</c>), and the Python side keeps its own original object. Either way
    ///     the two representations have the same dtype and the same bytes, which is what makes the lockstep comparison a
    ///     comparison of dynamics alone.
    ///     </para>
    ///     <para>
    ///     Owns both representations: dispose it after the step, which releases the Python object under the GIL and the
    ///     NDArray when this instance made it.
    ///     </para>
    /// </remarks>
    public sealed class ParityAction : IDisposable {
        /// <summary>The Python representation, or <see langword="null"/> until first needed (or after disposal).</summary>
        private PyObject? _python;

        /// <summary>Whether <see cref="DotNet"/> is an array this instance must dispose.</summary>
        private readonly bool _ownsDotNet;

        /// <summary>Creates an action from both representations.</summary>
        /// <param name="python">The Python object, owned; <see langword="null"/> to create it from the .NET value on first use.</param>
        /// <param name="dotnet">The .NET value.</param>
        /// <param name="ownsDotNet">Whether <paramref name="dotnet"/> is an array this instance disposes.</param>
        /// <param name="facts">The action's description, for reports.</param>
        private ParityAction(PyObject? python, object dotnet, bool ownsDotNet, ValueFacts facts) {
            _python = python;
            DotNet = dotnet;
            _ownsDotNet = ownsDotNet;
            Facts = facts;
        }

        /// <summary>The .NET representation, what <see cref="IDotNetEnv.Step"/> receives.</summary>
        public object DotNet { get; }

        /// <summary>The action as the Python side holds it, described; reports print it.</summary>
        public ValueFacts Facts { get; }

        /// <summary>A discrete action: a Python <c>int</c> on one side, a <see langword="long"/> on the other.</summary>
        /// <param name="value">The action.</param>
        /// <returns>The action.</returns>
        public static ParityAction Discrete(long value) => new(null, value, false, ValueFacts.FromInteger(value, "", "int"));

        /// <summary>A continuous action: an <see cref="NDArray"/> on the .NET side and a numpy copy of it on the Python side.</summary>
        /// <param name="values">The array; ownership passes to the action.</param>
        /// <returns>The action.</returns>
        public static ParityAction Continuous(NDArray values) => new(null, values, true, ValueFacts.Of(values));

        /// <summary>
        ///     An action a Python policy picked. The Python side keeps the policy's own object; the .NET side gets a
        ///     <see langword="long"/> for an integer and a NumSharp copy (<c>ToNDArray</c>) for an array. Must run under the GIL.
        /// </summary>
        /// <param name="python">The policy's action; ownership passes to the action.</param>
        /// <returns>The action.</returns>
        /// <exception cref="NotSupportedException">Thrown for an action that is neither an integer nor an array.</exception>
        internal static ParityAction FromPython(PyObject python) {
            ValueFacts facts = PyGym.Describe(python);
            switch (facts.Kind) {
                case FactKind.Int:
                    return new ParityAction(python, (long)facts.Integer, false, facts);
                case FactKind.Array:
                    return new ParityAction(python, python.ToNDArray(), true, facts);
                default:
                    python.Dispose();
                    throw new NotSupportedException($"A policy returned {facts.Render()} ({facts.TypeName}); only integer and array actions are supported.");
            }
        }

        /// <summary>
        ///     An action sampled on each side independently (the sampled source), after the two samples were compared. Each
        ///     side steps with its own sample, as it would outside the harness. Must run under the GIL.
        /// </summary>
        /// <param name="python">Gymnasium's sample; ownership passes to the action.</param>
        /// <param name="dotnet">Gym.NET's sample; ownership passes to the action when it is disposable.</param>
        /// <returns>The action.</returns>
        internal static ParityAction Sampled(PyObject python, object dotnet) => new(python, dotnet, dotnet is IDisposable, PyGym.Describe(python));

        /// <summary>The Python representation, created from the .NET value on first use. Must run under the GIL.</summary>
        /// <returns>The object, still owned by this action.</returns>
        internal PyObject Python() {
            _python ??= PyGym.ToPython(DotNet);
            return _python;
        }

        /// <summary>Releases the Python object under the GIL, and the .NET array when this action owns it.</summary>
        public void Dispose() {
            if (_python != null) {
                using (Py.GIL())
                    _python.Dispose();
                _python = null;
            }

            if (_ownsDotNet && DotNet is IDisposable disposable)
                disposable.Dispose();
        }

        /// <inheritdoc />
        public override string ToString() => Facts.Render();
    }
}
