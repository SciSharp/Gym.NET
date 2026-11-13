using System;
using System.Threading;
using Gym.Tests.Parity.Interop;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace Gym.Tests.Parity.Frames {
    /// <summary>
    ///     Whether Gymnasium can draw MuJoCo frames in this process, and on which GPU, found out before any cell lets it
    ///     render: the cells that need Gymnasium's MuJoCo frames call <see cref="RequireRendering"/> or
    ///     <see cref="RequireGpu"/> first, so a machine where Gymnasium cannot draw them, or draws them on another GPU than a
    ///     measurement's, reports the cell Inconclusive (DR-71) instead of failing inside Gymnasium's <c>render()</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A machine without a GPU cannot run MuJoCo's renderer: it requires OpenGL 1.5 with
    ///     <c>ARB_framebuffer_object</c> (<c>mujoco.dll</c>: "OpenGL version 1.5 or higher required"), and Windows without a
    ///     GPU driver, as on GitHub's hosted runners, offers only its own OpenGL 1.1 software renderer. Gymnasium's first
    ///     render raises there, from GLFW or from <c>MjrContext</c>. The frame cells read the GPU only after that render,
    ///     while its context is current, so before this probe they failed on such a machine, where DR-71 rules them
    ///     Inconclusive. Measured with <c>MUJOCO_GL=wgl</c>, which makes Gymnasium refuse its backend at the same point: the
    ///     66 cells that use the probe (the 44 frame cells, the overlay cell, the 21 MuJoCo goldens' live cells) all failed
    ///     without it, and all report Inconclusive with it; the whole gate under it passed otherwise (2,850 cells), so no
    ///     other cell lets Gymnasium draw MuJoCo frames.
    ///     </para>
    ///     <para>
    ///     The probe builds what Gymnasium's first <c>rgb_array</c> or <c>depth_array</c> render builds: its own
    ///     <c>OffScreenViewer</c>, with the same backend choice (<c>MUJOCO_GL</c> included), the same GLFW context and the
    ///     same <c>MjrContext</c>, over a one-geom model. It reads <c>GL_RENDERER</c> and <c>GL_VERSION</c> while that
    ///     context is current, frees it, and makes current again whatever context was current before. It runs once per
    ///     process, since the GPU does not change while one runs. On the reference machine it reads exactly what the cells
    ///     read after Gymnasium's own renders (<c>NVIDIA GeForce RTX 4060 Ti/PCIe/SSE2 | 4.6.0 NVIDIA 616.56</c>, measured
    ///     2026-11-13, three probes in a row and again after an env's <c>close()</c>), so the cells still gate there.
    ///     </para>
    ///     <para>
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/mujoco_rendering.py#L174">gymnasium/envs/mujoco/mujoco_rendering.py</see>
    ///     (<c>OffScreenViewer</c>, the viewer the probe builds).
    ///     </para>
    /// </remarks>
    internal static class MujocoGpuProbe {
        /// <summary>
        ///     The probe, run in a fresh scope: builds Gymnasium's <c>OffScreenViewer</c> over a one-geom model, reads the
        ///     GPU while the viewer's context is current, frees the viewer's MuJoCo and GLFW contexts, and makes current again
        ///     the context that was current before. Leaves in <c>result</c> <c>(True, "GL_RENDERER | GL_VERSION")</c>, or
        ///     <c>(False, "Type: message")</c> when the viewer could not be built.
        /// </summary>
        /// <remarks>
        ///     <list type="bullet">
        ///         <item>
        ///             <b>One geom, not an empty model:</b> the viewer aims its camera at <c>np.median</c> of the geoms'
        ///             positions, which over no geoms is a mean of nothing, with NumPy's warning.
        ///         </item>
        ///         <item>
        ///             <b><c>free()</c>, not <c>close()</c>:</b> Gymnasium's <c>close()</c> also calls
        ///             <c>glfw.terminate()</c>, which would destroy every other GLFW window in the process.
        ///         </item>
        ///         <item>
        ///             <b>The previous context is made current again</b> because a GL context is current per thread and
        ///             Gymnasium's offscreen viewer makes its own current only when it is created (this project's
        ///             <c>CLAUDE.md</c>, pitfalls), so an env built before the probe keeps drawing into its own.
        ///             <c>glfw.init()</c> first, since asking GLFW for the current context before it is initialized is an
        ///             error that <c>glfw</c> reports as a warning.
        ///         </item>
        ///         <item>
        ///             <b>The imports stay outside the <c>try</c>:</b> a stack without GLFW, PyOpenGL or Gymnasium's MuJoCo
        ///             rendering module cannot run any MuJoCo env, which the stack check reports, so the probe raises
        ///             rather than call that machine GPU-less.
        ///         </item>
        ///     </list>
        /// </remarks>
        private const string ProbeStatements = @"
import glfw
import mujoco
from OpenGL import GL
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

previous = glfw.get_current_context() if glfw.init() else None
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=""1""/></worldbody></mujoco>')
data = mujoco.MjData(model)
viewer = None
try:
    viewer = OffScreenViewer(model, data, 64, 64)
    result = (True, GL.glGetString(GL.GL_RENDERER).decode() + ' | ' + GL.glGetString(GL.GL_VERSION).decode())
except Exception as error:
    result = (False, type(error).__name__ + ': ' + str(error))
finally:
    if viewer is not None:
        viewer.con.free()
        viewer.free()
    if previous:
        glfw.make_context_current(previous)
";

        /// <summary>
        ///     The probe's answer, computed by the first cell that asks; a probe that raises (a stack without GLFW, PyOpenGL
        ///     or Gymnasium's MuJoCo rendering) raises again in every cell that asks, as such a stack fails every MuJoCo cell.
        /// </summary>
        private static readonly Lazy<(bool Rendered, string Text)> Probed = new(Probe, LazyThreadSafetyMode.ExecutionAndPublication);

        /// <summary>
        ///     The GPU Gymnasium draws MuJoCo frames on in this process, as <c>GL_RENDERER | GL_VERSION</c>, the format of the
        ///     frame baseline's <c>gpu</c> field; empty when Gymnasium cannot draw them here (<see cref="Failure"/>).
        /// </summary>
        /// <exception cref="PythonException">The probe could not run: GLFW, PyOpenGL or Gymnasium's MuJoCo rendering module does not import.</exception>
        internal static string Gpu => Probed.Value.Rendered ? Probed.Value.Text : "";

        /// <summary>
        ///     Why Gymnasium cannot draw MuJoCo frames in this process (what building its offscreen viewer raised, as
        ///     <c>Type: message</c>), or <see langword="null"/> when it can.
        /// </summary>
        /// <exception cref="PythonException">The probe could not run: GLFW, PyOpenGL or Gymnasium's MuJoCo rendering module does not import.</exception>
        internal static string? Failure => Probed.Value.Rendered ? null : Probed.Value.Text;

        /// <summary>
        ///     Reports the calling cell Inconclusive when Gymnasium cannot draw MuJoCo frames here; for a cell that uses
        ///     Gymnasium's frames on any GPU, such as a MuJoCo golden's live cell, which measures Gym.NET's frames against them
        ///     without gating on the measurement. Call it before the cell's first Gymnasium render.
        /// </summary>
        /// <param name="cell">What the cell needs the frames for, the start of the report, e.g. <c>The golden mujoco/Ant-v5.sampled.seed0 measures Gym.NET's frames against Gymnasium's</c>.</param>
        /// <exception cref="AssertInconclusiveException">Gymnasium cannot draw MuJoCo frames in this process.</exception>
        /// <exception cref="PythonException">The probe could not run: GLFW, PyOpenGL or Gymnasium's MuJoCo rendering module does not import.</exception>
        internal static void RequireRendering(string cell) {
            if (Failure is { } failure)
                Assert.Inconclusive($"{cell}, and Gymnasium cannot draw MuJoCo frames here: building its offscreen viewer, as its first render does, raised {failure}");
        }

        /// <summary>
        ///     Reports the calling cell Inconclusive unless Gymnasium draws MuJoCo frames here on the GPU a measurement was
        ///     taken on; for a cell gated on numbers measured on one GPU (DR-71), since part of Gymnasium's pixels are that
        ///     GPU's own choices. Call it before the cell's first Gymnasium render.
        /// </summary>
        /// <param name="recordedGpu">The measurement's GPU, <c>GL_RENDERER | GL_VERSION</c> (<see cref="MujocoFrameBaseline.Gpu"/>).</param>
        /// <param name="measurement">What was measured on it, the start of the report, e.g. <c>The MuJoCo frame baseline was recorded</c>.</param>
        /// <exception cref="AssertInconclusiveException">Gymnasium cannot draw MuJoCo frames in this process, or draws them on another GPU or driver.</exception>
        /// <exception cref="PythonException">The probe could not run: GLFW, PyOpenGL or Gymnasium's MuJoCo rendering module does not import.</exception>
        internal static void RequireGpu(string recordedGpu, string measurement) {
            RequireRendering($"{measurement} on '{recordedGpu}'");
            if (Gpu != recordedGpu)
                Assert.Inconclusive($"{measurement} on '{recordedGpu}'; Gymnasium draws on '{Gpu}' here, whose own choices decide part of its pixels (DR-71).");
        }

        /// <summary>Runs <see cref="ProbeStatements"/> under the GIL and reads its result.</summary>
        /// <returns>Whether the viewer was built, and the GPU it reported or the exception that stopped it.</returns>
        /// <exception cref="PythonException">GLFW, PyOpenGL or Gymnasium's MuJoCo rendering module does not import.</exception>
        private static (bool Rendered, string Text) Probe() =>
            PythonRuntime.Run(() => {
                using PyObject result = Python.evalStmts(ProbeStatements, "result");
                using PyObject rendered = result.GetItem(0);
                using PyObject text = result.GetItem(1);
                return (rendered.IsTrue(), text.As<string>());
            });
    }
}
