using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Gym.Envs;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Envs {
    /// <summary>
    ///     The MuJoCo frames (DR-71): each id's <c>rgb_array</c> and <c>depth_array</c> frames, drawn by Gym.NET (DR-62),
    ///     against Gymnasium's of the same state after a seeded reset and after ten seeded sampled steps (fewer when the episode
    ///     ends first). Each frame is measured (differing pixels or depth values, the largest delta or ulp distance, a
    ///     histogram, PNGs of both sides and of their difference) and gated on its checked-in baseline: it fails when it
    ///     differs in more pixels than recorded plus Gymnasium's own render-to-render noise at that state.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Equality with Gymnasium is never required (DR-67): part of Gymnasium's pixels are its GPU's own choices, which
    ///     Gym.NET reproduces where they were measured (<c>tools/parity/mujoco/gpu_probe.py</c>) and approximates elsewhere.
    ///     So the cells are pinned twice: to the reference platform, whose arithmetic the engine's trajectory is bit for bit
    ///     with, and to the GPU the baseline was recorded on (<see cref="MujocoFrameBaseline"/>); on another GPU or driver they
    ///     are Inconclusive, and so where Gymnasium cannot draw MuJoCo frames at all (a machine without a GPU, such as CI's
    ///     hosted runner), which <see cref="MujocoGpuProbe"/> finds before Gymnasium's first render. A deliberate improvement
    ///     re-records the baseline with <c>GYMNET_PARITY_RECORD_BASELINE=1</c>, which also measures the noise from five
    ///     renders of each state.
    ///     </para>
    ///     <para>
    ///     The trajectory itself runs through <see cref="GymPair"/>, so every reset and step is still compared bit for bit
    ///     before a frame is measured: a frame never differs because the states did.
    ///     </para>
    /// </remarks>
    [TestClass]
    public sealed class MujocoFrameLiveParityTests : ParityTestBase {
        /// <summary>What pins the cells to the reference platform: the engine's trajectory, as in the lockstep cells.</summary>
        private const string Platform = "MuJoCo 4.0.0's win-x64 arithmetic (the C runtime's sin/cos/pow, its AVX kernels)";

        /// <summary>The reset's and the action space's seed.</summary>
        private const long Seed = 0;

        /// <summary>The steps before the second frame.</summary>
        private const int Steps = 10;

        /// <summary>Gymnasium's renders of each state a recording measures the noise from.</summary>
        private const int NoiseRenders = 5;

        /// <summary>Every id that builds, in both array modes; <c>rgbd_tuple</c> is the two together (the unit tests pin that).</summary>
        public static IEnumerable<object[]> IdsAndModes =>
            from id in MujocoLiveParityTests.Ids from mode in new[] { "rgb_array", "depth_array" } select new object[] { id, mode };

        /// <summary>
        ///     An id's frames in a mode, after the reset and after the steps, each within its baseline; with
        ///     <c>GYMNET_PARITY_RECORD_BASELINE=1</c>, the measurements and Gymnasium's noise become the baseline instead.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="mode"><c>rgb_array</c> or <c>depth_array</c>.</param>
        /// <exception cref="AssertFailedException">A frame worse than its baseline, a frame with no baseline, or frames of different shapes.</exception>
        /// <exception cref="AssertInconclusiveException">Off the reference platform, on another GPU than the baseline's, or where Gymnasium cannot draw MuJoCo frames (no GPU).</exception>
        /// <exception cref="ParityException">The trajectory diverged before a frame was taken.</exception>
        [TestMethod]
        [DynamicData(nameof(IdsAndModes))]
        public void GymNet_Frames_StayWithinTheirBaseline(string id, string mode) {
            SkipOffReferencePlatform(Platform);
            List<string> failures = Run(id, mode, new Dictionary<string, object?> { ["render_mode"] = mode }, MujocoFrameBaseline.RecordRequested);
            if (failures.Count > 0)
                Assert.Fail(string.Join(Environment.NewLine, failures));
        }

        /// <summary>
        ///     The gate's negative control for colour: HalfCheetah-v5 drawn without textures (<c>visual_options</c> turning
        ///     <c>mjVIS_TEXTURE</c> off on Gym.NET's side only), whose checker floor then differs far beyond the baseline at
        ///     both states.
        /// </summary>
        /// <exception cref="AssertFailedException">The planted defect passed the gate.</exception>
        /// <exception cref="AssertInconclusiveException">Off the reference platform or GPU, where Gymnasium cannot draw MuJoCo frames (no GPU), or while recording (the control would record its defect).</exception>
        [TestMethod]
        public void PlantedTexturelessFloor_FailsTheGate() {
            SkipOffReferencePlatform(Platform);
            if (MujocoFrameBaseline.RecordRequested)
                Assert.Inconclusive("Recording the baseline: a planted defect is never recorded.");
            var kwargs = new Dictionary<string, object?> { ["render_mode"] = "rgb_array", ["visual_options"] = new Dictionary<int, bool> { [1] = false } };
            List<string> failures = Run("HalfCheetah-v5", "rgb_array", kwargs, record: false);
            Assert.AreEqual(2, failures.Count, string.Join(Environment.NewLine, failures));
            foreach (string failure in failures)
                StringAssert.Contains(failure, "worse than the baseline");
        }

        /// <summary>
        ///     The gate's negative control for depth: Reacher-v5's free camera set farther on Gym.NET's side only
        ///     (<c>default_camera_config</c>), so every depth value moves.
        /// </summary>
        /// <exception cref="AssertFailedException">The planted defect passed the gate.</exception>
        /// <exception cref="AssertInconclusiveException">Off the reference platform or GPU, where Gymnasium cannot draw MuJoCo frames (no GPU), or while recording.</exception>
        [TestMethod]
        public void PlantedFartherCamera_FailsTheDepthGate() {
            SkipOffReferencePlatform(Platform);
            if (MujocoFrameBaseline.RecordRequested)
                Assert.Inconclusive("Recording the baseline: a planted defect is never recorded.");
            var camera = new Dictionary<string, object?> { ["trackbodyid"] = 0L, ["distance"] = 1.0 };
            var kwargs = new Dictionary<string, object?> { ["render_mode"] = "depth_array", ["default_camera_config"] = camera };
            List<string> failures = Run("Reacher-v5", "depth_array", kwargs, record: false);
            Assert.AreEqual(2, failures.Count, string.Join(Environment.NewLine, failures));
            foreach (string failure in failures)
                StringAssert.Contains(failure, "worse than the baseline");
        }

        /// <summary>
        ///     Runs one id's frames in a mode: the seeded reset and the steps in lockstep, each state's frames measured, and
        ///     either gated on the baseline or, recording, written into it with Gymnasium's noise.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="mode"><c>rgb_array</c> or <c>depth_array</c>.</param>
        /// <param name="dotnetKwargs">Gym.NET's side's keyword arguments (a negative control plants its defect here).</param>
        /// <param name="record">Whether to record instead of gating.</param>
        /// <returns>The states that failed the gate, each with its measurement; empty when all passed or recorded.</returns>
        /// <exception cref="AssertInconclusiveException">
        ///     Gymnasium cannot draw MuJoCo frames here (a machine without a GPU, found by <see cref="MujocoGpuProbe"/> before
        ///     its first render), or draws them on another GPU than the baseline's.
        /// </exception>
        /// <exception cref="ParityException">The trajectory diverged before a frame was taken.</exception>
        private List<string> Run(string id, string mode, Dictionary<string, object?> dotnetKwargs, bool record) {
            MujocoFrameBaseline baseline = MujocoFrameBaseline.Load();
            // before Gymnasium's first render, which raises where it cannot draw MuJoCo frames (no GPU): such a machine, or
            // another GPU than the baseline's, is Inconclusive (DR-71); recording needs only that Gymnasium can draw them,
            // since the GPU it draws on becomes the baseline's
            if (record)
                MujocoGpuProbe.RequireRendering("Recording the MuJoCo frame baseline needs Gymnasium's frames");
            else
                MujocoGpuProbe.RequireGpu(baseline.Gpu, "The MuJoCo frame baseline was recorded");
            using var pair = new GymPair(PyGymEnv.Make(id, renderMode: mode),
                                         new GymNetEnv((IEnv)PyKwargs.Construct((Type)Gymnasium.Spec(id).EntryPoint!, dotnetKwargs)));
            var failures = new List<string>();
            string? gpu = null;

            pair.Reset(Seed);
            Measure("after reset");
            IActionSource actions = ActionSources.Sampled(Seed);
            actions.Start(pair);
            while (pair.StepsTaken < Steps && !pair.Ended) {
                using ParityAction action = actions.Next(pair);
                pair.Step(action);
            }

            Measure($"after step {pair.StepsTaken.ToString(CultureInfo.InvariantCulture)}");
            if (record) {
                baseline.Save();
                TestContext.WriteLine($"recorded into {MujocoFrameBaseline.FilePath}");
            }

            return failures;

            // one state: Gymnasium's frame (and, recording, its noise) against Gym.NET's, then the gate or the record
            void Measure(string label) {
                string key = $"{id} {mode} {label}";
                Rendered python = RenderPython(pair);
                // the GPU is read right after Gymnasium's render, while its offscreen context is current on this thread
                gpu ??= PythonRuntime.Run(() =>
                    $"{Python.OpenGL.GL.glGetString(Python.OpenGL.GL.GL_RENDERER)} | {Python.OpenGL.GL.glGetString(Python.OpenGL.GL.GL_VERSION)}");
                if (!record && baseline.Gpu != gpu)
                    Assert.Inconclusive($"The MuJoCo frame baseline was recorded on '{baseline.Gpu}'; Gymnasium draws on '{gpu}' here, whose own choices decide part of its pixels (DR-71).");
                Rendered dotnet = RenderDotNet(pair);
                if (python.Height != dotnet.Height || python.Width != dotnet.Width || (python.Rgb is null) != (dotnet.Rgb is null)) {
                    failures.Add($"{key}: Gymnasium's frame is {python.Describe()}, Gym.NET's {dotnet.Describe()}");
                    return;
                }

                (long differing, long largest, string described) = Compare(python, dotnet, key);
                TestContext.WriteLine($"{key}: {described}");
                if (record) {
                    long noise = 0;
                    var renders = new List<Rendered> { python };
                    for (int i = 1; i < NoiseRenders; i++)
                        renders.Add(RenderPython(pair));
                    for (int a = 0; a < renders.Count; a++) {
                        for (int b = a + 1; b < renders.Count; b++)
                            noise = Math.Max(noise, Differing(renders[a], renders[b]));
                    }

                    baseline.Set(key, new MujocoFrameBaseline.Entry(differing, largest, noise), gpu);
                    TestContext.WriteLine($"{key}: Gymnasium's own renders differ in up to {noise}");
                    return;
                }

                if (!baseline.TryGet(key, out MujocoFrameBaseline.Entry entry)) {
                    failures.Add($"{key}: no baseline; record it with {MujocoFrameBaseline.RecordVariable}=1 (-f net10.0)");
                    return;
                }

                if (differing > entry.Differing + entry.Noise)
                    failures.Add($"{key}: {differing} differ, worse than the baseline's {entry.Differing} plus Gymnasium's noise {entry.Noise} ({described})");
            }
        }

        /// <summary>A frame of either kind: RGB bytes or depth values, with its size.</summary>
        /// <param name="Height">Rows.</param>
        /// <param name="Width">Columns.</param>
        /// <param name="Rgb">The RGB frame, for <c>rgb_array</c>.</param>
        /// <param name="Depth">The depths, row-major, for <c>depth_array</c>.</param>
        private sealed record Rendered(int Height, int Width, Frame? Rgb, float[]? Depth) {
            /// <summary>The frame's kind and size, for a report.</summary>
            /// <returns>For example <c>rgb 480x480</c>.</returns>
            public string Describe() => $"{(Rgb is null ? "depth" : "rgb")} {Height}x{Width}";
        }

        /// <summary>Gymnasium's frame: <c>env.render()</c> of the unwrapped env, read as uint8 RGB or float32 depth.</summary>
        /// <param name="pair">The pair.</param>
        /// <returns>The frame.</returns>
        /// <exception cref="InvalidDataException">The render was neither a uint8 (H, W, 3) nor a float32 (H, W) array.</exception>
        private static Rendered RenderPython(GymPair pair) {
            return PythonRuntime.Run(() => {
                using PyObject result = Python.gymnasium.Env.render(pair.Python.Stepped);
                ValueFacts facts = ValueFacts.FromPython(result);
                if (facts.Kind == FactKind.Array && facts.DType == "|u1" && facts.Shape.Length == 3) {
                    Frame frame = Frame.FromFacts(facts);
                    return new Rendered(frame.Height, frame.Width, frame, null);
                }

                if (facts.Kind == FactKind.Array && facts.DType == "<f4" && facts.Shape.Length == 2) {
                    var depth = new float[facts.Bytes.Length / 4];
                    Buffer.BlockCopy(facts.Bytes, 0, depth, 0, facts.Bytes.Length);
                    return new Rendered(checked((int)facts.Shape[0]), checked((int)facts.Shape[1]), null, depth);
                }

                throw new InvalidDataException($"Gymnasium's MuJoCo render was {facts.Render()}, neither rgb_array's nor depth_array's.");
            });
        }

        /// <summary>Gym.NET's frame: <c>Render()</c> of the .NET env.</summary>
        /// <param name="pair">The pair.</param>
        /// <returns>The frame.</returns>
        /// <exception cref="InvalidDataException">The render was neither an RGB nor a depth result.</exception>
        private static Rendered RenderDotNet(GymPair pair) {
            using RenderResult result = ((GymNetEnv)pair.DotNet).Env.Render();
            switch (result) {
                case RenderResult.Rgb rgb: {
                    // the result owns its array (disposed with it above): only copies leave this method
                    int h = (int)rgb.Frame.shape[0], w = (int)rgb.Frame.shape[1];
                    return new Rendered(h, w, new Frame(h, w, rgb.Frame.ToArray<byte>()), null);
                }
                case RenderResult.Depth d: {
                    int h = (int)d.Frame.shape[0], w = (int)d.Frame.shape[1];
                    return new Rendered(h, w, null, d.Frame.ToArray<float>());
                }
                default:
                    throw new InvalidDataException($"Gym.NET's MuJoCo render was {result}, neither rgb_array's nor depth_array's.");
            }
        }

        /// <summary>Measures two frames of one kind and writes their artifacts (both frames and the difference, as PNGs).</summary>
        /// <param name="python">Gymnasium's frame.</param>
        /// <param name="dotnet">Gym.NET's frame, of the same kind and size.</param>
        /// <param name="key">The state, naming the artifacts' directory.</param>
        /// <returns>The differing count, the largest delta (RGB) or ulp distance (depth), and the description.</returns>
        private static (long Differing, long Largest, string Described) Compare(Rendered python, Rendered dotnet, string key) {
            if (python.Rgb is not null) {
                FrameComparison comparison = FrameDiff.Compare(python.Rgb, dotnet.Rgb!);
                FrameDiff.WriteArtifacts(python.Rgb, dotnet.Rgb!, comparison, key);
                return (comparison.DifferingPixels, comparison.MaxChannelDelta, comparison.Describe());
            }

            float[] p = python.Depth!, d = dotnet.Depth!;
            long differing = 0, largest = 0;
            for (int i = 0; i < p.Length; i++) {
                long ulps = Math.Abs((long)BitConverter.SingleToInt32Bits(p[i]) - BitConverter.SingleToInt32Bits(d[i]));
                if (ulps == 0)
                    continue;
                differing++;
                largest = Math.Max(largest, ulps);
            }

            string described = $"{differing} of {p.Length} depth values differ, by at most {largest} ulps";
            string directory = ParityArtifacts.DirectoryFor(key);
            Grey(python, p, d).SavePng(Path.Combine(directory, "python.png"));
            Grey(dotnet, d, p).SavePng(Path.Combine(directory, "dotnet.png"));
            File.WriteAllText(Path.Combine(directory, "stats.txt"), described + Environment.NewLine);
            return (differing, largest, described);
        }

        /// <summary>The pixels or depth values in which two frames of one kind differ.</summary>
        /// <param name="a">One frame.</param>
        /// <param name="b">The other, of the same kind and size.</param>
        /// <returns>The count.</returns>
        private static long Differing(Rendered a, Rendered b) {
            if (a.Rgb is not null)
                return FrameDiff.Compare(a.Rgb, b.Rgb!).DifferingPixels;
            long n = 0;
            for (int i = 0; i < a.Depth!.Length; i++)
                n += BitConverter.SingleToInt32Bits(a.Depth[i]) != BitConverter.SingleToInt32Bits(b.Depth![i]) ? 1 : 0;
            return n;
        }

        /// <summary>
        ///     A depth frame as a grey image for the artifacts, stretched over the range both frames share (MuJoCo's depths
        ///     crowd near 1, so the full [0, 1] range would show one shade).
        /// </summary>
        /// <param name="frame">The frame's size.</param>
        /// <param name="depth">The depths to draw.</param>
        /// <param name="other">The other side's depths, for the shared range.</param>
        /// <returns>The image.</returns>
        private static Frame Grey(Rendered frame, float[] depth, float[] other) {
            float lo = Math.Min(depth.Min(), other.Min()), hi = Math.Max(depth.Max(), other.Max());
            float span = hi > lo ? hi - lo : 1;
            var rgb = new byte[3 * depth.Length];
            for (int i = 0; i < depth.Length; i++)
                rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = (byte)Math.Clamp((int)((depth[i] - lo) / span * 255), 0, 255);
            return new Frame(frame.Height, frame.Width, rgb);
        }
    }
}
