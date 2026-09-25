using System;
using Python.Runtime;

namespace Gym.Tests.Parity.Interop {
    /// <summary>
    ///     An owned Python object a test keeps across several GIL sections (a space it samples ten times, a generator it
    ///     draws from pattern after pattern), released under the GIL when the test disposes it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A bare <see cref="PyObject"/> in a <c>using</c> would be disposed without the GIL, which pythonnet only defers to its
    ///     finalizer queue; this holder takes the GIL for the release, so the leak gate settles when the test ends.
    ///     </para>
    ///     <para>
    ///     <see cref="Object"/> is borrowed: pass it to <see cref="Python"/> rows inside <see cref="PythonRuntime.Run(Action)"/>
    ///     (or <see cref="Oracle.Outcome.CapturePython"/>), never dispose it, and never keep it past this holder's disposal.
    ///     </para>
    /// </remarks>
    internal sealed class PyRef : IDisposable {
        /// <summary>The object, or <see langword="null"/> once released.</summary>
        private PyObject? _object;

        /// <summary>Takes ownership of <paramref name="value"/>.</summary>
        /// <param name="value">A new reference the holder now owns.</param>
        private PyRef(PyObject value) => _object = value;

        /// <summary>The held object, borrowed. Use it under the GIL.</summary>
        /// <exception cref="ObjectDisposedException">Thrown after <see cref="Dispose"/>.</exception>
        internal PyObject Object => _object ?? throw new ObjectDisposedException(nameof(PyRef));

        /// <summary>Runs <paramref name="create"/> under the GIL and holds what it returns.</summary>
        /// <param name="create">A <see cref="Python"/> row or chain returning a new reference (its intermediates disposed).</param>
        /// <returns>The holder; the caller disposes it.</returns>
        /// <exception cref="PythonException">Thrown when <paramref name="create"/> raises; nothing is held then.</exception>
        internal static PyRef Of(Func<PyObject> create) => PythonRuntime.Run(() => new PyRef(create()));

        /// <summary>Releases the object under the GIL. Safe to call twice.</summary>
        public void Dispose() {
            if (_object is null)
                return;
            PythonRuntime.Run(() => {
                _object.Dispose();
                _object = null;
            });
        }
    }
}
