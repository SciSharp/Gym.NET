using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Gym.Environments;
using Gym.Environments.Envs.MuJoCo;
using Gym.Environments.Envs.MuJoCo.Rendering;
using Gym.Envs;
using Gym.MuJoCo;
using Gym.Rendering;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Tests.Parity.Tests.Envs {
    /// <summary>
    ///     The MuJoCo envs' <c>render_mode="human"</c> window (DR-70) against Gymnasium's and MuJoCo's own code: Gymnasium's
    ///     <c>WindowViewer</c> (its render loop, key, mouse and scroll callbacks, menu and <c>mjv_moveCamera</c> calls) run with
    ///     only GLFW, the GL calls, the clock and the image writer replaced by scripted stand-ins; MuJoCo's <c>mjr_overlay</c>
    ///     drawing the menus over Gymnasium's own frames; and MuJoCo's <c>mjv_updateScene</c> building the contact and frame
    ///     decorations the C and E keys switch on.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Gymnasium's real window cannot run here: it opens a visible GLFW window, and this suite never uses the screen. So the
    ///     controls cell builds a <c>WindowViewer</c> without its <c>__init__</c> (which opens the window), gives it the state
    ///     <c>__init__</c> sets, and swaps the module's <c>glfw</c>, <c>mujoco</c>, <c>time</c> and <c>imageio</c> for stand-ins
    ///     for the length of the run: GLFW's input queries answer from the script, <c>poll_events</c> delivers the next scripted
    ///     batch through the viewer's own callbacks, <c>mjr_render</c> and <c>mjr_overlay</c> record instead of drawing, the
    ///     clock advances 37.5 ms a read on both sides. Everything else is Gymnasium's and MuJoCo's own code.
    ///     </para>
    ///     <para>
    ///     The overlay cell is byte-exact: MuJoCo's text over Gymnasium's single-sampled frames (<c>offsamples = 0</c>, as the
    ///     window's buffer is) against Gym.NET's overlay drawn over the same Gymnasium frames. It reads the GPU's storage and
    ///     blending, which Gym.NET reproduces from the reference GPU's measurements, so it is pinned to the GPU the frame
    ///     baseline was recorded on (Inconclusive elsewhere), as the frame cells are (DR-71).
    ///     </para>
    /// </remarks>
    [TestClass]
    public sealed class MujocoWindowLiveParityTests : ParityTestBase {
        /// <summary>What pins the cells to the reference platform: <c>mjv_moveCamera</c>'s and the engine's sin, cos and sqrt.</summary>
        private const string Platform = "MuJoCo 4.0.0's win-x64 arithmetic (the C runtime's sin/cos/pow, its AVX kernels)";

        /// <summary>The scripted clock's step: each read of either side's clock is this much later than the last.</summary>
        private const double ClockStep = 0.0375;

        /// <summary>The window's size in the controls cell.</summary>
        private const int WindowWidth = 96, WindowHeight = 72;

        /// <summary>
        ///     Gymnasium's <c>WindowViewer</c> on stand-ins: binds <c>env</c> (the unwrapped env), <c>width</c>, <c>height</c>,
        ///     <c>image_path</c>, <c>step</c> (the clock's step) and <c>script</c> (per render call, per frame, the input
        ///     events as <c>(kind, name, x, y)</c>); leaves in <c>result</c>, per render call, the frames' overlays and the
        ///     viewer's state as lines.
        /// </summary>
        private const string WindowViewerStatements =
            "import struct\n" +
            "import types\n" +
            "\n" +
            "import glfw\n" +
            "import mujoco\n" +
            "from gymnasium.envs.mujoco import mujoco_rendering\n" +
            "\n" +
            "\n" +
            "def bits(x):\n" +
            "    # a float's IEEE-754 bits as 16 hex digits\n" +
            "    return '%016x' % struct.unpack('<Q', struct.pack('<d', float(x)))[0]\n" +
            "\n" +
            "\n" +
            "class StandInGlfw:\n" +
            "    # GLFW as the viewer uses it: input queries answered from the script, poll_events delivering its next batch\n" +
            "    def __init__(self, width, height):\n" +
            "        self.width, self.height = width, height\n" +
            "        self.keys, self.buttons, self.cursor = set(), set(), (0.0, 0.0)\n" +
            "        self.batches, self.viewer = [], None\n" +
            "\n" +
            "    def __getattr__(self, name):\n" +
            "        return getattr(glfw, name)\n" +
            "\n" +
            "    def window_should_close(self, window):\n" +
            "        return False\n" +
            "\n" +
            "    def get_framebuffer_size(self, window):\n" +
            "        return self.width, self.height\n" +
            "\n" +
            "    def swap_buffers(self, window):\n" +
            "        pass\n" +
            "\n" +
            "    def get_key(self, window, key):\n" +
            "        return glfw.PRESS if key in self.keys else glfw.RELEASE\n" +
            "\n" +
            "    def get_mouse_button(self, window, button):\n" +
            "        return glfw.PRESS if button in self.buttons else glfw.RELEASE\n" +
            "\n" +
            "    def get_cursor_pos(self, window):\n" +
            "        return self.cursor\n" +
            "\n" +
            "    def poll_events(self):\n" +
            "        v = self.viewer\n" +
            "        batch = self.batches.pop(0) if self.batches else []\n" +
            "        for kind, name, x, y in batch:\n" +
            "            if kind in ('key_down', 'key_up'):\n" +
            "                key = getattr(glfw, name)\n" +
            "                if kind == 'key_down':\n" +
            "                    self.keys.add(key)\n" +
            "                else:\n" +
            "                    self.keys.discard(key)\n" +
            "                v._key_callback(v.window, key, 0, glfw.PRESS if kind == 'key_down' else glfw.RELEASE, 0)\n" +
            "            elif kind in ('button_down', 'button_up'):\n" +
            "                button = getattr(glfw, name)\n" +
            "                if kind == 'button_down':\n" +
            "                    self.buttons.add(button)\n" +
            "                else:\n" +
            "                    self.buttons.discard(button)\n" +
            "                self.cursor = (x, y)\n" +
            "                v._mouse_button_callback(v.window, button, glfw.PRESS if kind == 'button_down' else glfw.RELEASE, 0)\n" +
            "            elif kind == 'move':\n" +
            "                self.cursor = (x, y)\n" +
            "                v._cursor_pos_callback(v.window, x, y)\n" +
            "            elif kind == 'scroll':\n" +
            "                v._scroll_callback(v.window, x, y)\n" +
            "            elif kind == 'resize':\n" +
            "                self.width, self.height = int(x), int(y)\n" +
            "\n" +
            "\n" +
            "class StandInMujoco:\n" +
            "    # the mujoco module but for the GL calls: mjr_render starts a frame, mjr_overlay records its texts\n" +
            "    def __init__(self):\n" +
            "        self.frames = []\n" +
            "\n" +
            "    def __getattr__(self, name):\n" +
            "        return getattr(mujoco, name)\n" +
            "\n" +
            "    def mjr_render(self, viewport, scn, con):\n" +
            "        self.frames.append([])\n" +
            "\n" +
            "    def mjr_overlay(self, font, gridpos, viewport, text1, text2, con):\n" +
            "        self.frames[-1].append((int(gridpos), text1, text2))\n" +
            "\n" +
            "    def mjr_readPixels(self, rgb, depth, viewport, con):\n" +
            "        pass\n" +
            "\n" +
            "\n" +
            "class StandInClock:\n" +
            "    # time.time(): each read the step later than the last\n" +
            "    def __init__(self, step):\n" +
            "        self.reads, self.step = 0, step\n" +
            "\n" +
            "    def time(self):\n" +
            "        self.reads += 1\n" +
            "        return self.reads * self.step\n" +
            "\n" +
            "\n" +
            "class StandInImageio:\n" +
            "    # imageio.imwrite: records the path\n" +
            "    def __init__(self):\n" +
            "        self.saved = []\n" +
            "\n" +
            "    def imwrite(self, path, image):\n" +
            "        self.saved.append(path)\n" +
            "\n" +
            "\n" +
            "model, data = env.model, env.data\n" +
            "v = mujoco_rendering.WindowViewer.__new__(mujoco_rendering.WindowViewer)\n" +
            "v.model, v.data = model, data\n" +
            "v._markers, v._overlays = [], {}\n" +
            "v.viewport = mujoco.MjrRect(0, 0, width, height)\n" +
            "v.scn = mujoco.MjvScene(model, 1000)\n" +
            "v.cam, v.vopt, v.pert = mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvPerturb()\n" +
            "v.con = None\n" +
            "v._button_left_pressed = False\n" +
            "v._button_right_pressed = False\n" +
            "v._last_mouse_x = 0\n" +
            "v._last_mouse_y = 0\n" +
            "v._paused = False\n" +
            "v._transparent = False\n" +
            "v._contacts = False\n" +
            "v._render_every_frame = True\n" +
            "v._image_idx = 0\n" +
            "v._image_path = image_path\n" +
            "v._time_per_render = 1 / 60.0\n" +
            "v._run_speed = 1.0\n" +
            "v._loop_count = 0\n" +
            "v._advance_by_one_step = False\n" +
            "v._hide_menu = False\n" +
            "v.window = 'scripted'\n" +
            "v.width, v.height = width, height\n" +
            "v._scale = 1.0\n" +
            "mujoco_rendering.MujocoRenderer._set_cam_config(types.SimpleNamespace(viewer=v, default_cam_config=env.mujoco_renderer.default_cam_config))\n" +
            "\n" +
            "g, m, clock, writer = StandInGlfw(width, height), StandInMujoco(), StandInClock(step), StandInImageio()\n" +
            "g.viewer = v\n" +
            "saved = (mujoco_rendering.glfw, mujoco_rendering.mujoco, mujoco_rendering.time, mujoco_rendering.imageio)\n" +
            "mujoco_rendering.glfw, mujoco_rendering.mujoco, mujoco_rendering.time, mujoco_rendering.imageio = g, m, clock, writer\n" +
            "result = []\n" +
            "try:\n" +
            "    for batches in script:\n" +
            "        g.batches = [list(batch) for batch in batches]\n" +
            "        m.frames = []\n" +
            "        v.render()\n" +
            "        lines = []\n" +
            "        for i, frame in enumerate(m.frames):\n" +
            "            for gridpos, text1, text2 in frame:\n" +
            "                lines.append('frame %d gridpos %d %r %r' % (i, gridpos, text1, text2))\n" +
            "        lines.append('frames %d' % len(m.frames))\n" +
            "        lines.append('cam type=%d fixedcamid=%d trackbodyid=%d orthographic=%d' % (v.cam.type, v.cam.fixedcamid, v.cam.trackbodyid, v.cam.orthographic))\n" +
            "        lines.append('cam lookat=' + ' '.join(bits(x) for x in v.cam.lookat))\n" +
            "        lines.append('cam distance=%s azimuth=%s elevation=%s' % (bits(v.cam.distance), bits(v.cam.azimuth), bits(v.cam.elevation)))\n" +
            "        lines.append('vopt contactpoint=%d contactforce=%d frame=%d geomgroup=%s' % (\n" +
            "            v.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT], v.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE], v.vopt.frame,\n" +
            "            ''.join(str(int(x)) for x in v.vopt.geomgroup)))\n" +
            "        lines.append('alphas=' + model.geom_rgba[:, 3].astype('<f4').tobytes().hex())\n" +
            "        lines.append('paused=%s advance=%s transparent=%s contacts=%s every_frame=%s hide_menu=%s' % (\n" +
            "            v._paused, v._advance_by_one_step, v._transparent, v._contacts, v._render_every_frame, v._hide_menu))\n" +
            "        lines.append('image_idx=%d run_speed=%s loop_count=%s time_per_render=%s' % (\n" +
            "            v._image_idx, bits(v._run_speed), bits(v._loop_count), bits(v._time_per_render)))\n" +
            "        lines.append('mouse left=%s right=%s last=(%d, %d)' % (v._button_left_pressed, v._button_right_pressed, v._last_mouse_x, v._last_mouse_y))\n" +
            "        lines.append('saved=' + ','.join(writer.saved))\n" +
            "        result.append(lines)\n" +
            "finally:\n" +
            "    mujoco_rendering.glfw, mujoco_rendering.mujoco, mujoco_rendering.time, mujoco_rendering.imageio = saved\n" +
            "    # the viewer never owned a window: its __del__ must not free one\n" +
            "    v.window = None\n";

        /// <summary>
        ///     Gymnasium's offscreen frames with MuJoCo's overlays: binds <c>env</c> (the unwrapped env, not rendered yet) and
        ///     <c>overlays</c> (sets of <c>(gridpos, text1, text2)</c>); makes the context single-sampled as the window's buffer is
        ///     (<c>offsamples = 0</c>, before the first render creates it), and leaves in <c>result</c> the frame without
        ///     overlays, the same frame again, and the frame of each set.
        /// </summary>
        private const string OverlayStatements = @"
env.model.vis.quality.offsamples = 0
first = env.render()
again = env.render()
viewer = env.mujoco_renderer.viewer
frames = []
for overlay in overlays:
    viewer._overlays = {gridpos: [text1, text2] for gridpos, text1, text2 in overlay}
    frames.append(env.render())
viewer._overlays = {}
result = (first, again, frames)
";

        /// <summary>
        ///     MuJoCo's scene: binds <c>env</c> (the unwrapped env), <c>point</c>, <c>force</c>, <c>split</c> (the contact
        ///     flags) and <c>frame</c> (<c>mjvOption.frame</c>); runs <c>mjv_updateScene</c> with the default free camera into a
        ///     fresh scene and leaves in <c>result</c> each geom's type and category (int32) and its size, position, orientation,
        ///     colour, emission, specular, shininess and reflectance (float32, 23 per geom).
        /// </summary>
        private const string SceneStatements = @"
import mujoco
import numpy as np

vopt = mujoco.MjvOption()
vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = point
vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = force
vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = split
vopt.frame = frame
scn = mujoco.MjvScene(env.model, 1000)
mujoco.mjv_updateScene(env.model, env.data, vopt, mujoco.MjvPerturb(), mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL.value, scn)
geoms = [scn.geoms[i] for i in range(scn.ngeom)]
kinds = np.array([[g.type, g.category] for g in geoms], dtype=np.int32).reshape(-1, 2)
floats = np.array([np.concatenate([g.size, g.pos, g.mat.ravel(), g.rgba, [g.emission, g.specular, g.shininess, g.reflectance]]) for g in geoms],
                  dtype=np.float32).reshape(-1, 23)
result = (kinds, floats)
";

        /// <summary>
        ///     Gymnasium's window and Gym.NET's driven by the same input over 25 render calls of Humanoid-v5 (two model cameras):
        ///     after each call the frames drawn and each frame's menu text for text, then the camera (float64 bits), the scene
        ///     options, the model's geom alphas (float32 bits), and every field of the viewer's state, equal.
        /// </summary>
        /// <exception cref="AssertFailedException">A render call's frames, menus or state differ.</exception>
        /// <exception cref="AssertInconclusiveException">Off the reference platform.</exception>
        [TestMethod]
        public void GymNet_WindowControls_AreGymnasiumsWindowViewer() {
            SkipOffReferencePlatform(Platform);
            string? difference = CompareWindows(ClockStep, out int calls, out int lines);
            if (difference != null)
                Assert.Fail(difference);
            TestContext.WriteLine($"{calls} render calls, {lines} lines, equal");
        }

        /// <summary>
        ///     The controls cell's negative control: Gym.NET's window measured with a clock twice as slow as Gymnasium's stand-in,
        ///     so its frame time, and from it the menu's frame rate and the frames real time owes, drift; the comparison must
        ///     report it at the first render call's state.
        /// </summary>
        /// <exception cref="AssertFailedException">The planted difference passed the comparison, or was reported elsewhere.</exception>
        /// <exception cref="AssertInconclusiveException">Off the reference platform.</exception>
        [TestMethod]
        public void PlantedSlowerClock_FailsTheControlsComparison() {
            SkipOffReferencePlatform(Platform);
            string? difference = CompareWindows(2 * ClockStep, out _, out _);
            Assert.IsNotNull(difference, "a window with another clock compared equal");
            StringAssert.StartsWith(difference, "render call 0,");
            StringAssert.Contains(difference, "time_per_render=");
        }

        /// <summary>
        ///     Runs the controls script through Gymnasium's window on stand-ins and Gym.NET's window, and compares the two
        ///     descriptions render call by render call, line by line.
        /// </summary>
        /// <param name="dotnetClockStep">Gym.NET's clock step (the stand-in's is <see cref="ClockStep"/>); a negative control changes it.</param>
        /// <param name="calls">The render calls run.</param>
        /// <param name="lines">Gymnasium's description's lines, all calls together.</param>
        /// <returns>The first difference, with both lines, or <see langword="null"/> when every line is equal.</returns>
        /// <exception cref="AssertFailedException">Either side ran another number of render calls than the script has.</exception>
        /// <exception cref="PythonException">Gymnasium's code raised.</exception>
        private static string? CompareWindows(double dotnetClockStep, out int calls, out int lines) {
            string folder = Path.Combine(Path.GetTempPath(), "gymnet-window-parity-" + Guid.NewGuid().ToString("N")).Replace('\\', '/');
            try {
                List<List<Event[]>> script = Script();
                List<List<string>> python = PythonWindow(script, folder + "/frame_%07d.png");
                List<List<string>> dotnet = DotNetWindow(script, folder + "/frame_{0:D7}.png", dotnetClockStep);
                Assert.AreEqual(script.Count, python.Count, "Gymnasium's render calls");
                Assert.AreEqual(script.Count, dotnet.Count, "Gym.NET's render calls");
                calls = script.Count;
                lines = python.Sum(c => c.Count);
                for (int call = 0; call < script.Count; call++) {
                    for (int line = 0; line < Math.Max(python[call].Count, dotnet[call].Count); line++) {
                        string p = line < python[call].Count ? python[call][line] : "(nothing)", d = line < dotnet[call].Count ? dotnet[call][line] : "(nothing)";
                        if (p != d)
                            return $"render call {call}, line {line}:{Environment.NewLine}  Gymnasium: {p}{Environment.NewLine}  Gym.NET:   {d}";
                    }
                }

                return null;
            } finally {
                if (Directory.Exists(folder))
                    Directory.Delete(folder, recursive: true);
            }
        }

        /// <summary>
        ///     MuJoCo's <c>mjr_overlay</c> and Gym.NET's overlay over the same Gymnasium frame of InvertedPendulum-v5, single-
        ///     sampled as the window's buffer is: the window's menus (running, paused and piled up past 500 characters, with a
        ///     run speed and a saved path), every printable character in both columns at each of the eight grid positions, one
        ///     column, boxes larger than the frame, lines without a final newline: every pixel equal.
        /// </summary>
        /// <exception cref="AssertFailedException">A set's pixels differ.</exception>
        /// <exception cref="AssertInconclusiveException">Off the reference platform, on another GPU than the baseline's, or when Gymnasium's two plain renders differ.</exception>
        [TestMethod]
        public void GymNet_Overlay_IsMujocosMjrOverlay() {
            SkipOffReferencePlatform(Platform);
            List<(int Gridpos, string Text1, string Text2)[]> sets = OverlaySets();
            Frame? first = null, again = null;
            var frames = new List<Frame>();
            string gpu = "";
            PythonRuntime.Run(() => {
                using PyObject made = Python.gymnasium.make("InvertedPendulum-v5", "rgb_array", new Dictionary<string, object?> { ["width"] = 480, ["height"] = 480 });
                using PyObject env = Python.gymnasium.Env.unwrapped(made);
                using PyObject reset = Python.gymnasium.Env.reset(env, 0, null);
                using PyObject overlays = PyConvert.ToPython(sets.Select(set => (object?)set.Select(o => (object?)(o.Gridpos, o.Text1, o.Text2)).ToList()).ToList());
                using PyObject result = Python.evalStmts(OverlayStatements, "result", ("env", env), ("overlays", overlays));
                ValueFacts facts = ValueFacts.FromPython(result);
                // the GPU is read right after Gymnasium's renders, while its offscreen context is current on this thread
                gpu = $"{Python.OpenGL.GL.glGetString(Python.OpenGL.GL.GL_RENDERER)} | {Python.OpenGL.GL.glGetString(Python.OpenGL.GL.GL_VERSION)}";
                first = Frame.FromFacts(facts.Items[0]);
                again = Frame.FromFacts(facts.Items[1]);
                frames.AddRange(facts.Items[2].Items.Select(Frame.FromFacts));
            });
            string baselineGpu = MujocoFrameBaseline.Load().Gpu;
            if (gpu != baselineGpu)
                Assert.Inconclusive($"The overlay's blending was measured on '{baselineGpu}'; Gymnasium draws on '{gpu}' here.");
            if (!FrameDiff.Compare(first!, again!).Identical)
                Assert.Inconclusive("Gymnasium's two renders of one state differ, so a frame with an overlay has no fixed base to compare against.");

            var failures = new List<string>();
            for (int i = 0; i < sets.Count; i++) {
                // a set MuJoCo did not draw would compare a plain frame with itself
                Assert.IsFalse(FrameDiff.Compare(first!, frames[i]).Identical, $"MuJoCo drew overlay set {i}");
                Frame expected = frames[i], drawn = DrawOverlays(first!, sets[i]);
                FrameComparison comparison = FrameDiff.Compare(expected, drawn);
                string label = $"overlay set {i}: " + string.Join(" + ", sets[i].Select(o => GridName(o.Gridpos)));
                TestContext.WriteLine($"{label}: {comparison.Describe()}");
                if (!comparison.Identical) {
                    FrameDiff.WriteArtifacts(expected, drawn, comparison, label);
                    failures.Add($"{label}: {comparison.Describe()}");
                }
            }

            if (failures.Count > 0)
                Assert.Fail(string.Join(Environment.NewLine, failures));
        }

        /// <summary>
        ///     The decorations the window's C and E keys switch on, as MuJoCo's <c>mjv_updateScene</c> builds them: for seven
        ///     ids after a seeded reset, after 20 seeded sampled steps in lockstep and after 60 (or where the episode ends), the
        ///     scene with contact points and forces (also split into normal and friction), every body's frame and the world's:
        ///     each geom's type, category, size, position, orientation, colour and material, float32 bit for bit. HumanoidStandup
        ///     lies on the floor (a dozen contacts) and Pusher's object rests on the table, so single and double arrows and
        ///     the split arrows all occur.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <exception cref="AssertFailedException">A scene's geoms differ.</exception>
        /// <exception cref="AssertInconclusiveException">Off the reference platform.</exception>
        /// <exception cref="ParityException">The trajectory diverged before a scene was built.</exception>
        [TestMethod]
        [DataRow("Ant-v5")]
        [DataRow("Humanoid-v5")]
        [DataRow("HumanoidStandup-v5")]
        [DataRow("HalfCheetah-v5")]
        [DataRow("Hopper-v5")]
        [DataRow("Walker2d-v5")]
        [DataRow("Pusher-v5")]
        public void GymNet_Decorations_AreMujocosSceneGeoms(string id) {
            SkipOffReferencePlatform(Platform);
            using var pair = new GymPair(PyGymEnv.Make(id), new GymNetEnv((IEnv)PyKwargs.Construct((Type)Gymnasium.Spec(id).EntryPoint!, new Dictionary<string, object?>())));
            var failures = new List<string>();
            pair.Reset(0);
            CompareScenes("after reset");
            IActionSource actions = ActionSources.Sampled(0);
            actions.Start(pair);
            foreach (int steps in new[] { 20, 60 }) {
                while (pair.StepsTaken < steps && !pair.Ended) {
                    using ParityAction action = actions.Next(pair);
                    pair.Step(action);
                }

                CompareScenes($"after step {pair.StepsTaken.ToString(CultureInfo.InvariantCulture)}");
                if (pair.Ended)
                    break;
            }
            if (failures.Count > 0)
                Assert.Fail(string.Join(Environment.NewLine, failures));

            void CompareScenes(string state) {
                var env = (MujocoEnv)((GymNetEnv)pair.DotNet).Env.Unwrapped;
                TestContext.WriteLine($"{id} {state}: {env.Data.ncon} contacts");
                foreach ((bool point, bool force, bool split, int frame) in new[] {
                             (true, true, false, MujocoScene.FrameNone), (true, true, true, MujocoScene.FrameNone),
                             (false, false, false, MujocoScene.FrameBody), (false, false, false, MujocoScene.FrameWorld),
                         }) {
                    string label = $"{id} {state}, contact point {point}, force {force}, split {split}, frame {frame}";
                    (int[] kinds, float[] floats) = PythonScene(pair.Python.Stepped, point, force, split, frame);
                    var scene = new MujocoScene();
                    scene.VisFlags[MujocoScene.VisContactPoint] = point;
                    scene.VisFlags[MujocoScene.VisContactForce] = force;
                    scene.VisFlags[MujocoScene.VisContactSplit] = split;
                    scene.Frame = frame;
                    scene.Update(env.Model, env.Data, new MjvCamera());
                    string? difference = FirstDifference(kinds, floats, scene);
                    TestContext.WriteLine($"{label}: {scene.Geoms.Count} geoms, {scene.Geoms.Count(g => g.Category == MujocoScene.CategoryDecor)} decorations{(difference is null ? ", equal" : "")}");
                    if (difference != null)
                        failures.Add($"{label}: {difference}");
                }
            }
        }

        /// <summary>
        ///     The controls cell's script: per render call, per frame, the input the frame's event poll delivers. The first
        ///     call is the reset's frame.
        /// </summary>
        /// <returns>The script.</returns>
        private static List<List<Event[]>> Script() {
            var script = new List<List<Event[]>> {
                Call(Batch()),
                Call(Key("KEY_TAB")),
                // a fixed camera does not move
                Call(Batch(Button("button_down", "MOUSE_BUTTON_LEFT", 10.7, 20.2), Move(30.9, 12.5), Move(-3.5, 70.25), Button("button_up", "MOUSE_BUTTON_LEFT", -3.5, 70.25))),
                Call(Key("KEY_TAB")),
                Call(Key("KEY_TAB")),
                // orbit, vertically then horizontally with Shift held
                Call(Batch(Button("button_down", "MOUSE_BUTTON_LEFT", 10.7, 20.2), Move(30.9, 12.5), Down("KEY_LEFT_SHIFT"), Move(-3.5, 70.25), Up("KEY_LEFT_SHIFT"),
                           Button("button_up", "MOUSE_BUTTON_LEFT", -3.5, 70.25))),
                // pan, then with the right Shift; then the wheel, whole and fractional notches
                Call(Batch(Button("button_down", "MOUSE_BUTTON_RIGHT", 40, 40), Move(36, 47), Down("KEY_RIGHT_SHIFT"), Move(30, 50), Up("KEY_RIGHT_SHIFT"),
                           Button("button_up", "MOUSE_BUTTON_RIGHT", 30, 50), Scroll(0, 2), Scroll(0.5, -1.25))),
                // both buttons: the right one wins; the middle button only moves the drag's start
                Call(Batch(Button("button_down", "MOUSE_BUTTON_LEFT", 50, 30), Button("button_down", "MOUSE_BUTTON_RIGHT", 51, 31), Move(60.2, 20.9),
                           Button("button_up", "MOUSE_BUTTON_RIGHT", 60.2, 20.9), Move(70, 25), Button("button_down", "MOUSE_BUTTON_MIDDLE", 10, 10), Move(12, 14),
                           Button("button_up", "MOUSE_BUTTON_LEFT", 12, 14), Move(90, 90), Button("button_up", "MOUSE_BUTTON_MIDDLE", 90, 90))),
                Call(Keys("KEY_S", "KEY_S", "KEY_F", "KEY_D")),
            };
            // rendering every frame off: the scripted clock decides which calls draw
            for (int i = 0; i < 6; i++)
                script.Add(Call(Batch()));
            script.Add(Call(Keys("KEY_D", "KEY_C", "KEY_E", "KEY_R", "KEY_0", "KEY_2")));
            script.Add(Call(Key("KEY_SPACE")));
            // paused: frames until → is released, the menu piling up
            script.Add(Call(Batch(), Batch(), Key("KEY_RIGHT")));
            script.Add(Call(Key("KEY_T"), Key("KEY_SPACE")));
            script.Add(Call(Key("KEY_RIGHT")));
            script.Add(Call(Key("KEY_SPACE")));
            // a resize, and a drag measured over the new size
            script.Add(Call(Batch(Resize(50, 40), Button("button_down", "MOUSE_BUTTON_LEFT", 5, 5), Move(25, 15), Button("button_up", "MOUSE_BUTTON_LEFT", 25, 15))));
            script.Add(Call(Key("KEY_H")));
            script.Add(Call(Batch()));
            script.Add(Call(Keys("KEY_H", "KEY_R", "KEY_C", "KEY_4")));
            return script;
        }

        /// <summary>One render call of the script: the input of each of its frames.</summary>
        /// <param name="batches">One batch per frame.</param>
        /// <returns>The call.</returns>
        private static List<Event[]> Call(params Event[][] batches) => batches.ToList();

        /// <summary>One frame's input.</summary>
        /// <param name="events">The events, in order.</param>
        /// <returns>The batch.</returns>
        private static Event[] Batch(params Event[] events) => events;

        /// <summary>One frame's input: taps of several keys in order.</summary>
        /// <param name="names">The GLFW keys' names.</param>
        /// <returns>The batch.</returns>
        private static Event[] Keys(params string[] names) => names.SelectMany(Key).ToArray();

        /// <summary>Runs the script through Gymnasium's <c>WindowViewer</c> on stand-ins.</summary>
        /// <param name="script">The script.</param>
        /// <param name="imagePath">The T key's path, a Python <c>%</c> format.</param>
        /// <returns>Per render call, the lines describing it.</returns>
        /// <exception cref="PythonException">Gymnasium's code raised.</exception>
        private static List<List<string>> PythonWindow(List<List<Event[]>> script, string imagePath) {
            var lines = new List<List<string>>();
            PythonRuntime.Run(() => {
                using PyObject made = Python.gymnasium.make("Humanoid-v5", "rgb_array", new Dictionary<string, object?> { ["width"] = WindowWidth, ["height"] = WindowHeight });
                using PyObject env = Python.gymnasium.Env.unwrapped(made);
                using PyObject reset = Python.gymnasium.Env.reset(env, 0, null);
                using PyObject pyScript = PyConvert.ToPython(script.Select(call => (object?)call.Select(batch => (object?)batch.Select(e => (object?)(e.Kind, e.Name, e.X, e.Y)).ToList()).ToList()).ToList());
                using PyObject width = PyConvert.ToPython(WindowWidth), height = PyConvert.ToPython(WindowHeight), path = PyConvert.ToPython(imagePath), step = PyConvert.ToPython(ClockStep);
                using PyObject result = Python.evalStmts(WindowViewerStatements, "result", ("env", env), ("width", width), ("height", height), ("image_path", path),
                                                         ("step", step), ("script", pyScript));
                ValueFacts facts = ValueFacts.FromPython(result);
                lines.AddRange(facts.Items.Select(call => call.Items.Select(line => line.Text).ToList()));
            });
            return lines;
        }

        /// <summary>
        ///     Runs the script through Gym.NET's window: Humanoid-v5 in <c>human</c> mode over a scripted viewer, its clock
        ///     scripted as the stand-in's is, the reset drawing the first call.
        /// </summary>
        /// <param name="script">The script.</param>
        /// <param name="imagePath">The T key's path, a composite format.</param>
        /// <param name="clockStep">The clock's step: <see cref="ClockStep"/>, as the stand-in's, unless a negative control plants another.</param>
        /// <returns>Per render call, the lines describing it, as the snippet writes them.</returns>
        private static List<List<string>> DotNetWindow(List<List<Event[]>> script, string imagePath, double clockStep) {
            var viewer = new ScriptedViewer();
            using var env = new HumanoidEnvV5(renderMode: "human", width: WindowWidth, height: WindowHeight, viewerFactory: viewer.Factory);
            int reads = 0;
            env.MujocoRenderer.WindowClock = () => ++reads * clockStep;
            viewer.Capture = () => env.MujocoRenderer.Window!.LastMenu;
            var lines = new List<List<string>>();
            for (int call = 0; call < script.Count; call++) {
                viewer.Load(script[call].Select(batch => batch.Select(ToInput).ToArray()));
                if (call == 0) {
                    using (env.Reset(seed: 0)) { }
                    env.MujocoRenderer.Window!.ImagePath = imagePath;
                } else {
                    env.Render();
                }

                MujocoWindow w = env.MujocoRenderer.Window!;
                var described = new List<string>();
                for (int i = 0; i < viewer.Menus.Count; i++) {
                    foreach ((int gridpos, string text1, string text2) in viewer.Menus[i])
                        described.Add($"frame {i} gridpos {gridpos} {PyBuiltins.StrRepr(text1)} {PyBuiltins.StrRepr(text2)}");
                }

                MjvCamera cam = w.Camera;
                described.Add($"frames {viewer.Menus.Count}");
                described.Add($"cam type={cam.Type} fixedcamid={cam.FixedCamId} trackbodyid={cam.TrackBodyId} orthographic={(cam.Orthographic ? 1 : 0)}");
                described.Add("cam lookat=" + string.Join(" ", cam.Lookat.Select(Bits)));
                described.Add($"cam distance={Bits(cam.Distance)} azimuth={Bits(cam.Azimuth)} elevation={Bits(cam.Elevation)}");
                described.Add($"vopt contactpoint={One(w.Scene.VisFlags[MujocoScene.VisContactPoint])} contactforce={One(w.Scene.VisFlags[MujocoScene.VisContactForce])} " +
                              $"frame={w.Scene.Frame} geomgroup={string.Concat(w.Scene.GeomGroup.Take(6).Select(One))}");
                float[] alphas = Enumerable.Range(0, env.Model.ngeom).Select(i => env.Model.geom_rgba[4 * i + 3]).ToArray();
                described.Add("alphas=" + Convert.ToHexString(MemoryMarshal.AsBytes(alphas.AsSpan())).ToLowerInvariant());
                described.Add($"paused={w.Paused} advance={w.AdvanceByOneStep} transparent={w.Transparent} contacts={w.Contacts} every_frame={w.RenderEveryFrame} hide_menu={w.HideMenu}");
                described.Add($"image_idx={w.ImageIdx} run_speed={Bits(w.RunSpeed)} loop_count={Bits(w.LoopCount)} time_per_render={Bits(w.TimePerRender)}");
                described.Add($"mouse left={w.ButtonLeftPressed} right={w.ButtonRightPressed} last=({w.LastMouseX}, {w.LastMouseY})");
                described.Add("saved=" + string.Join(",", Enumerable.Range(0, w.ImageIdx).Select(i => string.Format(CultureInfo.InvariantCulture, imagePath, i)).Where(File.Exists)));
                lines.Add(described);
            }

            return lines;

            static string One(bool flag) => flag ? "1" : "0";
        }

        /// <summary>
        ///     The overlay cell's sets: the window's menus, every printable character in both columns at the eight positions, one
        ///     column at the eight positions, oversized boxes, and lines without a final newline.
        /// </summary>
        /// <returns>The sets, each drawn in order over the plain frame.</returns>
        private static List<(int Gridpos, string Text1, string Text2)[]> OverlaySets() {
            const string running = "\nRen[d]er every frame\nSwitch camera (#cams = 1)\n[C]ontact forces\nT[r]ansparent\nStop\nReferenc[e] frames\n[H]ide Menu\nCap[t]ure frame\nToggle geomgroup visibility\n";
            const string values = "\nOn\n[Tab] (camera ID = -1)\nOff\nOff\n[Space]\nOff\n\n\n0-4\n";
            const string paused = "\nRen[d]er every frame\nSwitch camera (#cams = 1)\n[C]ontact forces\nT[r]ansparent\nStart\nAdvance simulation by one step\nReferenc[e] frames\n[H]ide Menu\nCap[t]ure frame\nToggle geomgroup visibility\n";
            const string pausedValues = "\nOn\n[Tab] (camera ID = -1)\nOff\nOff\n[Space]\n[right arrow]\nOff\n\n\n0-4\n";
            const string stats = "FPS\nSolver iterations\nStep\ntimestep\n";
            string printable = new(Enumerable.Range(32, 95).Select(c => (char)c).ToArray());
            string lines = string.Join("\n", Enumerable.Range(0, 5).Select(i => printable.Substring(i * 19, 19))) + "\n";
            string reversed = string.Join("\n", Enumerable.Range(0, 5).Select(i => new string(printable.Substring(i * 19, 19).Reverse().ToArray()))) + "\n";
            int[] positions = Enumerable.Range(MujocoOverlay.TopLeft, 8).ToArray();
            return new List<(int, string, string)[]> {
                new[] { (MujocoOverlay.TopLeft, running, values), (MujocoOverlay.BottomLeft, stats, "60\n1\n0\n0.02000\n") },
                new[] {
                    (MujocoOverlay.TopLeft, string.Concat(Enumerable.Repeat(paused, 3)), string.Concat(Enumerable.Repeat(pausedValues, 3))),
                    (MujocoOverlay.BottomLeft, string.Concat(Enumerable.Repeat(stats, 3)), "26\n4\n12\n0.02000\n26\n4\n12\n0.02000\n27\n4\n12\n0.02000\n"),
                },
                new[] {
                    (MujocoOverlay.TopLeft, "Run speed = 0.125 x real time\n" + running.Substring(1),
                     "[S]lower, [F]aster\nOff\n[Tab] (camera ID = 0)\nOn\nOn\n[Space]\nOn\n\nSaved as /tmp/frame_0000003.png\n0-4\n"),
                },
                positions.Select(p => (p, lines, reversed)).ToArray(),
                positions.Select(p => (p, lines, "")).ToArray(),
                new[] {
                    (MujocoOverlay.Left, string.Concat(Enumerable.Repeat("tall\n", 30)), "x\n"), (MujocoOverlay.Top, new string('W', 120) + "\n", ""),
                    (MujocoOverlay.Right, string.Concat(Enumerable.Repeat("|\n", 30)), ""),
                },
                new[] { (MujocoOverlay.Bottom, new string('m', 600), ""), (MujocoOverlay.TopRight, "no newline", "at the end"), (MujocoOverlay.BottomRight, "\n\n\n", "x") },
            };
        }

        /// <summary>Gym.NET's overlays over a frame: the frame as a single-sampled buffer, each overlay drawn in order, read back.</summary>
        /// <param name="frame">Gymnasium's plain frame, top row first.</param>
        /// <param name="overlays">The overlays.</param>
        /// <returns>The frame with the overlays.</returns>
        private static Frame DrawOverlays(Frame frame, (int Gridpos, string Text1, string Text2)[] overlays) {
            int w = frame.Width, h = frame.Height;
            var target = new GlTarget(w, h, 1, color: true, stencil: false);
            for (int row = 0; row < h; row++) {
                for (int x = 0; x < w; x++) {
                    int o = 4 * ((h - 1 - row) * w + x), p = 3 * (row * w + x);
                    target.Color![o] = frame.Rgb[p];
                    target.Color[o + 1] = frame.Rgb[p + 1];
                    target.Color[o + 2] = frame.Rgb[p + 2];
                    target.Color[o + 3] = 255;
                }
            }

            foreach ((int gridpos, string text1, string text2) in overlays)
                MujocoOverlay.Draw(target, gridpos, 0, 0, w, h, text1, text2);
            var rgb = new byte[3 * w * h];
            for (int row = 0; row < h; row++) {
                for (int x = 0; x < w; x++) {
                    int o = 4 * ((h - 1 - row) * w + x), p = 3 * (row * w + x);
                    rgb[p] = target.Color![o];
                    rgb[p + 1] = target.Color[o + 1];
                    rgb[p + 2] = target.Color[o + 2];
                }
            }

            return new Frame(h, w, rgb);
        }

        /// <summary>MuJoCo's scene of the Python env's current state, with the given decoration options.</summary>
        /// <param name="env">The unwrapped Gymnasium env; borrowed.</param>
        /// <param name="point">Contact points.</param>
        /// <param name="force">Contact forces.</param>
        /// <param name="split">Forces split into normal and friction.</param>
        /// <param name="frame">The frames drawn (<c>mjtFrame</c>).</param>
        /// <returns>Each geom's type and category, then its 23 floats.</returns>
        /// <exception cref="PythonException">MuJoCo raised.</exception>
        private static (int[] Kinds, float[] Floats) PythonScene(PyObject env, bool point, bool force, bool split, int frame) {
            return PythonRuntime.Run(() => {
                using PyObject p = PyConvert.ToPython(point), f = PyConvert.ToPython(force), s = PyConvert.ToPython(split), fr = PyConvert.ToPython(frame);
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(env);
                using PyObject result = Python.evalStmts(SceneStatements, "result", ("env", unwrapped), ("point", p), ("force", f), ("split", s), ("frame", fr));
                ValueFacts facts = ValueFacts.FromPython(result);
                return (MemoryMarshal.Cast<byte, int>(facts.Items[0].Bytes).ToArray(), MemoryMarshal.Cast<byte, float>(facts.Items[1].Bytes).ToArray());
            });
        }

        /// <summary>The first difference between MuJoCo's geoms and Gym.NET's scene, or <see langword="null"/> when they are equal.</summary>
        /// <param name="kinds">MuJoCo's types and categories.</param>
        /// <param name="floats">MuJoCo's 23 floats per geom.</param>
        /// <param name="scene">Gym.NET's scene.</param>
        /// <returns>The difference, described.</returns>
        private static string? FirstDifference(int[] kinds, float[] floats, MujocoScene scene) {
            int count = kinds.Length / 2;
            if (count != scene.Geoms.Count)
                return $"MuJoCo has {count} geoms ({Count(kinds, 4)} decorations), Gym.NET {scene.Geoms.Count} ({scene.Geoms.Count(g => g.Category == MujocoScene.CategoryDecor)})";
            string[] names = ["size0", "size1", "size2", "pos0", "pos1", "pos2", "mat0", "mat1", "mat2", "mat3", "mat4", "mat5", "mat6", "mat7", "mat8",
                              "rgba0", "rgba1", "rgba2", "rgba3", "emission", "specular", "shininess", "reflectance"];
            for (int i = 0; i < count; i++) {
                SceneGeom g = scene.Geoms[i];
                if (kinds[2 * i] != g.Type || kinds[2 * i + 1] != g.Category)
                    return $"geom {i}: MuJoCo's type {kinds[2 * i]} category {kinds[2 * i + 1]}, Gym.NET's {g.Type} {g.Category}";
                float[] mine = [.. g.Size, .. g.Pos, .. g.Mat, .. g.Rgba, g.Emission, g.Specular, g.Shininess, g.Reflectance];
                for (int k = 0; k < 23; k++) {
                    float theirs = floats[23 * i + k];
                    if (BitConverter.SingleToInt32Bits(theirs) != BitConverter.SingleToInt32Bits(mine[k]))
                        return $"geom {i} (type {g.Type}, category {g.Category}) {names[k]}: MuJoCo's {theirs.ToString("R", CultureInfo.InvariantCulture)}, Gym.NET's {mine[k].ToString("R", CultureInfo.InvariantCulture)}";
                }
            }

            return null;

            static int Count(int[] kinds, int category) => Enumerable.Range(0, kinds.Length / 2).Count(i => kinds[2 * i + 1] == category);
        }

        /// <summary>A double's bits as 16 hex digits, as the snippet's <c>bits</c> writes them.</summary>
        /// <param name="x">The value.</param>
        /// <returns>The digits.</returns>
        private static string Bits(double x) => ((ulong)BitConverter.DoubleToInt64Bits(x)).ToString("x16", CultureInfo.InvariantCulture);

        /// <summary>A grid position's MuJoCo name, for labels.</summary>
        /// <param name="gridpos">The position.</param>
        /// <returns>For example <c>TOPLEFT</c>.</returns>
        private static string GridName(int gridpos) => new[] { "TOPLEFT", "TOPRIGHT", "BOTTOMLEFT", "BOTTOMRIGHT", "TOP", "BOTTOM", "LEFT", "RIGHT" }[gridpos];

        /// <summary>One scripted input: its kind as the snippet names it, a GLFW key or button name, and two numbers.</summary>
        /// <param name="Kind"><c>key_down</c>, <c>key_up</c>, <c>button_down</c>, <c>button_up</c>, <c>move</c>, <c>scroll</c> or <c>resize</c>.</param>
        /// <param name="Name">The GLFW constant's name for keys and buttons, else empty.</param>
        /// <param name="X">The pointer's x, the wheel's horizontal notches, or the new width.</param>
        /// <param name="Y">The pointer's y, the wheel's vertical notches, or the new height.</param>
        private sealed record Event(string Kind, string Name, double X, double Y);

        /// <summary>A key's press and release.</summary>
        /// <param name="name">The GLFW key's name.</param>
        /// <returns>The two events.</returns>
        private static Event[] Key(string name) => [Down(name), Up(name)];

        /// <summary>A key's press.</summary>
        /// <param name="name">The GLFW key's name.</param>
        /// <returns>The event.</returns>
        private static Event Down(string name) => new("key_down", name, 0, 0);

        /// <summary>A key's release.</summary>
        /// <param name="name">The GLFW key's name.</param>
        /// <returns>The event.</returns>
        private static Event Up(string name) => new("key_up", name, 0, 0);

        /// <summary>A mouse button's press or release at the pointer.</summary>
        /// <param name="kind"><c>button_down</c> or <c>button_up</c>.</param>
        /// <param name="name">The GLFW button's name.</param>
        /// <param name="x">The pointer's x.</param>
        /// <param name="y">The pointer's y.</param>
        /// <returns>The event.</returns>
        private static Event Button(string kind, string name, double x, double y) => new(kind, name, x, y);

        /// <summary>A pointer move.</summary>
        /// <param name="x">The new x.</param>
        /// <param name="y">The new y.</param>
        /// <returns>The event.</returns>
        private static Event Move(double x, double y) => new("move", "", x, y);

        /// <summary>A turn of the wheel.</summary>
        /// <param name="dx">Horizontal notches.</param>
        /// <param name="dy">Vertical notches.</param>
        /// <returns>The event.</returns>
        private static Event Scroll(double dx, double dy) => new("scroll", "", dx, dy);

        /// <summary>A new framebuffer size.</summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns>The event.</returns>
        private static Event Resize(int width, int height) => new("resize", "", width, height);

        /// <summary>A scripted event as Gym.NET's viewers report it.</summary>
        /// <param name="e">The event.</param>
        /// <returns>The viewer input.</returns>
        /// <exception cref="ArgumentException">An event kind or name the script does not use.</exception>
        private static ViewerInput ToInput(Event e) => e.Kind switch {
            "key_down" => ViewerInput.KeyDown(SdlKey(e.Name)),
            "key_up" => ViewerInput.KeyUp(SdlKey(e.Name)),
            "button_down" => ViewerInput.MouseDown(GlfwButton(e.Name), e.X, e.Y),
            "button_up" => ViewerInput.MouseUp(GlfwButton(e.Name), e.X, e.Y),
            "move" => ViewerInput.MouseMove(e.X, e.Y),
            "scroll" => ViewerInput.Scroll(e.X, e.Y, 0, 0),
            "resize" => ViewerInput.Resized((int)e.X, (int)e.Y),
            _ => throw new ArgumentException($"no such event kind '{e.Kind}'", nameof(e)),
        };

        /// <summary>The SDL keycode Gym.NET's viewers report for a GLFW key.</summary>
        /// <param name="name">The GLFW key's name.</param>
        /// <returns>The keycode.</returns>
        /// <exception cref="ArgumentException">A key the script does not use.</exception>
        private static int SdlKey(string name) => name switch {
            "KEY_TAB" => KeyCode.Tab, "KEY_SPACE" => KeyCode.Space, "KEY_RIGHT" => KeyCode.Right, "KEY_S" => KeyCode.S, "KEY_F" => KeyCode.F,
            "KEY_D" => KeyCode.D, "KEY_T" => KeyCode.T, "KEY_C" => KeyCode.C, "KEY_E" => KeyCode.E, "KEY_H" => KeyCode.H, "KEY_R" => KeyCode.R,
            "KEY_0" => KeyCode.Digit0, "KEY_1" => KeyCode.Digit0 + 1, "KEY_2" => KeyCode.Digit0 + 2, "KEY_3" => KeyCode.Digit0 + 3, "KEY_4" => KeyCode.Digit4,
            "KEY_LEFT_SHIFT" => KeyCode.LeftShift, "KEY_RIGHT_SHIFT" => KeyCode.RightShift,
            _ => throw new ArgumentException($"no SDL keycode mapped for {name}", nameof(name)),
        };

        /// <summary>GLFW's number of a mouse button, which Gym.NET's viewers use too.</summary>
        /// <param name="name">The GLFW button's name.</param>
        /// <returns>The number.</returns>
        /// <exception cref="ArgumentException">A button the script does not use.</exception>
        private static int GlfwButton(string name) => name switch {
            "MOUSE_BUTTON_LEFT" => ViewerInput.LeftButton,
            "MOUSE_BUTTON_RIGHT" => ViewerInput.RightButton,
            "MOUSE_BUTTON_MIDDLE" => ViewerInput.MiddleButton,
            _ => throw new ArgumentException($"no such button {name}", nameof(name)),
        };

        /// <summary>
        ///     A viewer that hands the window one scripted batch per frame and records each frame's menu (read from the window
        ///     while the frame is shown, before its event poll).
        /// </summary>
        private sealed class ScriptedViewer : IInteractiveEnvViewer {
            /// <summary>The current render call's batches still to hand out.</summary>
            private readonly Queue<ViewerInput[]> _batches = new();

            /// <summary>Reads the window's menu of the frame being shown.</summary>
            public Func<IReadOnlyList<(int Gridpos, string Text1, string Text2)>>? Capture { get; set; }

            /// <summary>The current render call's frames' menus.</summary>
            public List<IReadOnlyList<(int Gridpos, string Text1, string Text2)>> Menus { get; } = new();

            /// <summary>The factory handing out this viewer.</summary>
            /// <param name="width">The width.</param>
            /// <param name="height">The height.</param>
            /// <param name="title">The title.</param>
            /// <returns>This viewer.</returns>
            public Task<IEnvViewer> Factory(int width, int height, string? title = null) => Task.FromResult<IEnvViewer>(this);

            /// <summary>Starts a render call: its batches, and no frames yet; batches a call leaves unread are dropped with the next.</summary>
            /// <param name="batches">One batch per expected frame.</param>
            public void Load(IEnumerable<ViewerInput[]> batches) {
                _batches.Clear();
                foreach (ViewerInput[] batch in batches)
                    _batches.Enqueue(batch);
                Menus.Clear();
            }

            /// <summary>Records the frame's menu.</summary>
            /// <param name="img">The frame; not kept.</param>
            public void Render(Image img) => Menus.Add(Capture!());

            /// <summary>Nothing to close.</summary>
            public void CloseEnvironment() { }

            /// <summary>Nothing to start.</summary>
            public void BeginInput() { }

            /// <summary>Hands out the next batch.</summary>
            /// <returns>The batch, or nothing.</returns>
            public IReadOnlyList<ViewerInput> TakeInput() => _batches.Count > 0 ? _batches.Dequeue() : [];

            /// <summary>Nothing to release.</summary>
            public void Dispose() { }
        }
    }
}
