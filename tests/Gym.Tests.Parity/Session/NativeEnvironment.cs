using System;
using System.Runtime.InteropServices;

namespace Gym.Tests.Parity.Session {
    /// <summary>
    ///     Sets environment variables where the embedded CPython actually reads them: in the C runtime's copy of the
    ///     environment, not only in .NET's.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <see cref="Environment.SetEnvironmentVariable(string, string)"/> is not enough. On Windows it updates the Win32
    ///     environment block, while CPython's <c>Py_Initialize</c> and <c>os.environ</c> read the C runtime's copy, which the
    ///     CRT took at process start (measured on 2026-09-25: with the managed call alone, <c>PYTHONHASHSEED=0</c> was
    ///     invisible to Python and <c>sys.flags.hash_randomization</c> stayed 1; through <c>_wputenv_s</c> it was 0). On
    ///     Linux and macOS .NET keeps a managed copy and never calls <c>setenv</c>.
    ///     </para>
    ///     <para>
    ///     So both are set: the managed copy for .NET readers and child processes, the native one for CPython and the
    ///     libraries it loads (SDL, OpenBLAS). Values must be set before <c>PythonEngine.Initialize</c> for the interpreter's
    ///     own configuration (<c>PYTHONHASHSEED</c>, <c>PYTHONDONTWRITEBYTECODE</c>) and before the library's import for the
    ///     others.
    ///     </para>
    /// </remarks>
    internal static class NativeEnvironment {
        /// <summary>Sets <paramref name="name"/> to <paramref name="value"/> in both the managed and the C runtime environment.</summary>
        /// <param name="name">The variable's name.</param>
        /// <param name="value">Its value; an empty string is a valid value, not a removal.</param>
        /// <exception cref="InvalidOperationException">Thrown when the C runtime rejects the assignment.</exception>
        public static void Set(string name, string value) {
            Environment.SetEnvironmentVariable(name, value);
            int status = OperatingSystem.IsWindows() ? _wputenv_s(name, value) : setenv(name, value, 1);
            if (status != 0)
                throw new InvalidOperationException($"The C runtime rejected {name}={value} (status {status}).");
        }

        /// <summary>
        ///     The Universal CRT's environment setter. Every module in the process that uses the UCRT, python3XY.dll included,
        ///     shares <c>ucrtbase.dll</c>'s single environment, so this is the copy CPython reads.
        /// </summary>
        /// <param name="name">The variable's name.</param>
        /// <param name="value">Its value.</param>
        /// <returns>0 on success, an <c>errno</c> value otherwise.</returns>
        [DllImport("ucrtbase.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
        private static extern int _wputenv_s(string name, string value);

        /// <summary>POSIX <c>setenv</c>, for the process environment libpython reads on Linux and macOS.</summary>
        /// <param name="name">The variable's name.</param>
        /// <param name="value">Its value.</param>
        /// <param name="overwrite">Non-zero to replace an existing value.</param>
        /// <returns>0 on success, -1 otherwise.</returns>
        [DllImport("libc", ExactSpelling = true)]
        private static extern int setenv(string name, string value, int overwrite);
    }
}
