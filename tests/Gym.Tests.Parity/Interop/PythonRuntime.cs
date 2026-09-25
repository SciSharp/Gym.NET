using System;
using System.Collections.Concurrent;
using Python.Runtime;

namespace Gym.Tests.Parity.Interop {
    /// <summary>
    ///     The library-agnostic backend of the parity suite's pythonnet bridge (the pythonic skill's §6): it starts the one
    ///     embedded CPython of the process, owns the GIL entry point every Python call goes through, and caches imported
    ///     modules. The <see cref="Python"/> tree and the harness sit on top of it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     CPython and NumPy cannot be initialized again after <c>Py_Finalize</c>, so the lifecycle is process-scoped:
    ///     <see cref="Initialize"/> runs once from the assembly initializer and <see cref="Shutdown"/> once from the assembly
    ///     cleanup.
    ///     </para>
    ///     <para>
    ///     <b>GIL discipline:</b> anything that touches a <see cref="PyObject"/>, its <c>Dispose</c> included, runs inside
    ///     <see cref="Run(Action)"/>/<see cref="Run{T}(Func{T})"/>. <c>Py.GIL()</c> is re-entrant, so a body may call harness
    ///     code that takes it again. Never hand a <see cref="PyObject"/> out of a <c>Run</c> body unless something owns it
    ///     and disposes it under the GIL later.
    ///     </para>
    /// </remarks>
    internal static class PythonRuntime {
        /// <summary>Imported modules by name: process-scoped handles, disposed only by <see cref="Shutdown"/>.</summary>
        private static readonly ConcurrentDictionary<string, PyObject> Modules = new(StringComparer.Ordinal);

        /// <summary>Whether the embedded engine is running.</summary>
        internal static bool IsInitialized => PythonEngine.IsInitialized;

        /// <summary>
        ///     Starts the embedded engine on <paramref name="library"/> and releases the GIL, so that any thread can take it
        ///     with <see cref="Run(Action)"/>. Performs no imports: what the process imports is the caller's decision.
        /// </summary>
        /// <param name="library">The Python shared library to load (<c>python312.dll</c>, <c>libpython3.12.so</c>).</param>
        /// <exception cref="PythonInteropException">Thrown when the engine is already running or fails to start.</exception>
        internal static void Initialize(string library) {
            if (PythonEngine.IsInitialized)
                throw new PythonInteropException("The embedded Python engine is already running; it starts once per process.");
            try {
                Runtime.PythonDLL = library;
                PythonEngine.Initialize();
                // Initialize leaves this thread holding the GIL; release it, so test threads can take it with Py.GIL().
                PythonEngine.BeginAllowThreads();
            } catch (Exception e) when (e is not PythonInteropException) {
                throw new PythonInteropException($"The embedded Python engine did not start from {library}: {e.GetType().Name}: {e.Message}", e);
            }
        }

        /// <summary>Runs <paramref name="body"/> holding the GIL.</summary>
        /// <param name="body">The Python-touching work; it disposes the temporaries it creates before returning.</param>
        /// <exception cref="PythonException">Thrown when a Python call in <paramref name="body"/> raises.</exception>
        internal static void Run(Action body) {
            using (Py.GIL())
                body();
        }

        /// <summary>Runs <paramref name="body"/> holding the GIL and returns its result.</summary>
        /// <typeparam name="T">The result's type; a CLR value, or a <see cref="PyObject"/> the caller will own and dispose under the GIL.</typeparam>
        /// <param name="body">The Python-touching work.</param>
        /// <returns>What <paramref name="body"/> returned.</returns>
        /// <exception cref="PythonException">Thrown when a Python call in <paramref name="body"/> raises.</exception>
        internal static T Run<T>(Func<T> body) {
            using (Py.GIL())
                return body();
        }

        /// <summary>
        ///     <c>import name</c>, cached for the life of the process. Must run under the GIL. The returned handle belongs to
        ///     the cache: never dispose it (a <c>using</c> would decref the cached module out from under every other caller).
        /// </summary>
        /// <param name="name">The module's dotted name, for example <c>gymnasium.spaces</c>.</param>
        /// <returns>The module.</returns>
        /// <exception cref="PythonException">Thrown when the import fails.</exception>
        internal static PyObject Import(string name) {
            if (Modules.TryGetValue(name, out PyObject? cached))
                return cached;
            PyObject module = Py.Import(name);
            // Two threads can race here; the loser disposes its duplicate handle and uses the winner's.
            if (!Modules.TryAdd(name, module)) {
                module.Dispose();
                return Modules[name];
            }

            return module;
        }

        /// <summary>The <c>builtins</c> module, cached like every import. Must be used under the GIL; never dispose it.</summary>
        internal static PyObject Builtins => Import("builtins");

        /// <summary>
        ///     Releases the cached modules under the GIL and shuts the engine down. Runs once, from the assembly cleanup; no
        ///     Python call is possible afterwards in this process.
        /// </summary>
        internal static void Shutdown() {
            if (!PythonEngine.IsInitialized)
                return;
            Run(() => {
                foreach (PyObject module in Modules.Values)
                    module.Dispose();
                Modules.Clear();
            });
            // pythonnet 3.0.x stashes runtime state with BinaryFormatter, which .NET 8+ removed; NoopFormatter opts out.
            RuntimeData.FormatterType = typeof(NoopFormatter);
            PythonEngine.Shutdown();
        }
    }

    /// <summary>The bridge's own failure: the embedded engine could not be started or used, as opposed to a Python exception.</summary>
    internal sealed class PythonInteropException : Exception {
        /// <summary>A failure with a message.</summary>
        /// <param name="message">What went wrong and what to do about it.</param>
        public PythonInteropException(string message) : base(message) { }

        /// <summary>A failure with a message and its cause.</summary>
        /// <param name="message">What went wrong and what to do about it.</param>
        /// <param name="innerException">The underlying exception.</param>
        public PythonInteropException(string message, Exception innerException) : base(message, innerException) { }
    }
}
