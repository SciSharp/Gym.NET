using System;
using Gym.Envs;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Wrappers {
    /// <summary>
    ///     What the wrapper parity cells share: running a pair through several seeded episodes, building Gym.NET's
    ///     <c>make(id)</c> typed, and comparing a Python attribute of the outer wrapper with Gym.NET's.
    /// </summary>
    internal static class WrapperParity {
        /// <summary>Runs a pair through episodes of seeded sampled actions, resetting after each end, for <paramref name="steps"/> steps in all.</summary>
        /// <param name="pair">The pair.</param>
        /// <param name="steps">The steps across all episodes.</param>
        /// <param name="context">Where each episode's summary is written.</param>
        /// <param name="frameEvery">Compare the frames every so many steps; 0 for none.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        internal static void RunEpisodes(GymPair pair, int steps, TestContext context, int frameEvery = 0) {
            int taken = 0;
            for (long seed = 0; taken < steps; seed++) {
                PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: steps - taken, frameEvery: frameEvery);
                taken += Math.Max(run.Steps, 1);
                context.WriteLine(run.ToString());
            }
        }

        /// <summary>Gym.NET's <c>make(id)</c>, typed as the array env every built-in Box/Discrete env is.</summary>
        /// <param name="id">The id.</param>
        /// <param name="renderMode">The render mode, or <see langword="null"/>.</param>
        /// <returns>The env; the caller (or the wrapper over it) owns it.</returns>
        internal static Env<NDArray, NDArray> Make(string id, string? renderMode = null) =>
            (Env<NDArray, NDArray>)(renderMode is null ? Gymnasium.Make(id) : Gymnasium.Make(id, new { render_mode = renderMode }));

        /// <summary>
        ///     Compares a Python attribute read by a snippet on Gymnasium's outermost wrapper with the value Gym.NET's gives,
        ///     after the pair's run.
        /// </summary>
        /// <param name="pair">The pair.</param>
        /// <param name="label">The comparison's name in reports.</param>
        /// <param name="statements">A snippet setting <c>result</c> from <c>env</c>, Gymnasium's outermost wrapper.</param>
        /// <param name="dotnet">Reads the value from Gym.NET's outermost wrapper.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        internal static void AssertAttribute(GymPair pair, string label, string statements, Func<IEnv, object?> dotnet) {
            Outcome python = Outcome.CapturePython(() => Python.evalStmts(statements, "result", ("env", pair.Python.Stepped)));
            ByteContract.AssertSame(label, python, Outcome.Capture(() => dotnet(((GymNetEnv)pair.DotNet).Env)));
        }

        /// <summary>A Python function from a <c>lambda</c> expression (<c>"lambda obs: obs * 2"</c>). Must run under the GIL.</summary>
        /// <param name="lambda">The expression.</param>
        /// <returns>A new owned reference to the function.</returns>
        /// <exception cref="PythonException">Thrown when the expression does not compile.</exception>
        internal static PyObject Lambda(string lambda) => Python.evalStmts("import numpy as np\nresult = " + lambda + "\n", "result");
    }
}
