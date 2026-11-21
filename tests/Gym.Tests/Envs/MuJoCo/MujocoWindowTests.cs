#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Gym.Environments;
using Gym.Environments.Envs.MuJoCo;
using Gym.Environments.Envs.MuJoCo.Rendering;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Rendering;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Tests.Envs.MuJoCo {
    /// <summary>
    ///     A MuJoCo env's <c>render_mode="human"</c> window (DR-70): Gymnasium's <c>WindowViewer</c> inside
    ///     <see cref="MujocoRenderer"/>, driven through a scripted viewer that reports keys, the mouse, resizes and the close
    ///     button between frames, as GLFW's event poll delivers them.
    /// </summary>
    /// <remarks>
    ///     The expectations are Gymnasium's: the menu's texts written out as <c>_create_overlay</c> builds them, the keys'
    ///     effects as <c>_key_callback</c> applies them, and the camera moved through <c>mjv_moveCamera</c>'s actions with the
    ///     deltas <c>_cursor_pos_callback</c> and <c>_scroll_callback</c> compute. The parity suite runs Gymnasium's own
    ///     callbacks and <c>mjr_overlay</c> against these (<c>MujocoWindowLiveParityTests</c>).
    /// </remarks>
    [TestClass]
    public class MujocoWindowTests {
        /// <summary>A small window keeps the software renderer quick where the menu need not be seen whole.</summary>
        private const int Small = 64;

        /// <summary>
        ///     Gymnasium's top-left menu of a running window with every toggle off and nothing captured, on a model with one
        ///     camera fewer than <c>#cams</c> says (the free one counts): the first column.
        /// </summary>
        /// <param name="cams"><c>model.ncam + 1</c>.</param>
        /// <returns>The column's text, each line ending with <c>'\n'</c> as <c>add_overlay</c> leaves it.</returns>
        private static string MenuLabels(int cams) =>
            $"\nRen[d]er every frame\nSwitch camera (#cams = {cams})\n[C]ontact forces\nT[r]ansparent\nStop\nReferenc[e] frames\n[H]ide Menu\nCap[t]ure frame\nToggle geomgroup visibility\n";

        /// <summary>The second column of <see cref="MenuLabels"/>'s menu with the free camera (camera ID -1).</summary>
        private const string MenuValues = "\nOn\n[Tab] (camera ID = -1)\nOff\nOff\n[Space]\nOff\n\n\n0-4\n";

        /// <summary>
        ///     The first human render opens a viewer titled <c>mujoco</c> at the renderer's size, starts reading its input, and
        ///     shows one frame per reset, step and render; <c>close</c> shuts it, and a render after that opens a new one.
        /// </summary>
        [TestMethod]
        public void Human_OpensAMujocoWindowAndShowsAFramePerRender() {
            var viewer = new ScriptedWindow();
            using var env = new InvertedPendulumEnvV5(renderMode: "human", width: Small, height: 48, viewerFactory: viewer.Factory);
            using (env.Reset(seed: 0)) { }
            Assert.AreEqual(MujocoWindow.Title, viewer.Title);
            Assert.AreEqual((Small, 48), (viewer.Width, viewer.Height));
            Assert.IsTrue(viewer.InputStarted, "the window reads the viewer's input");
            Assert.AreEqual(1, viewer.Frames.Count, "the reset's frame");
            Assert.AreEqual((Small, 48), (viewer.Frames[0].Width, viewer.Frames[0].Height));

            using (NDArray action = np.zeros(new Shape(1), np.float32))
            using (env.Step(action)) { }
            Assert.IsTrue(env.Render().IsNone, "human mode returns nothing");
            Assert.AreEqual(3, viewer.Frames.Count, "reset, step and render each show one frame");

            env.Close();
            Assert.IsTrue(viewer.Closed, "close shuts the window");
            Assert.IsNull(env.MujocoRenderer.Window);
            env.Render();
            Assert.AreEqual(2, viewer.Opened, "a render after close opens a new window");
        }

        /// <summary>
        ///     The window's camera is not the offscreen modes': Gymnasium's <c>WindowViewer</c> keeps MuJoCo's default free camera
        ///     (looking at the origin from 2 m at azimuth 90° and elevation -45°) and then applies the env's
        ///     <c>default_camera_config</c> (InvertedPendulum-v5's distance 2.04 and tracked body 0, unused by a free camera).
        /// </summary>
        [TestMethod]
        public void Camera_IsMujocosDefaultThenTheEnvsConfiguration() {
            var viewer = new ScriptedWindow();
            using var env = new InvertedPendulumEnvV5(renderMode: "human", width: Small, height: Small, viewerFactory: viewer.Factory);
            using (env.Reset(seed: 0)) { }
            MjvCamera cam = env.MujocoRenderer.Window!.Camera;
            Assert.AreEqual(MjvCamera.Free, cam.Type);
            Assert.AreEqual(-1, cam.FixedCamId);
            Assert.AreEqual(0, cam.TrackBodyId);
            CollectionAssert.AreEqual(new double[] { 0, 0, 0 }, cam.Lookat);
            Assert.AreEqual(2.04, cam.Distance);
            Assert.AreEqual(90.0, cam.Azimuth);
            Assert.AreEqual(-45.0, cam.Elevation);
        }

        /// <summary>
        ///     The first frame's menu is Gymnasium's, text for text (the frame rate is the initial 1/60 s's 60, the solver's
        ///     iterations, step and timestep of the reset state), and the H key hides it: the next frame, of the same state, is
        ///     the scene alone, and drawing the expected menu over it with MuJoCo's overlay gives the first frame byte for byte.
        /// </summary>
        [TestMethod]
        public void Menu_IsGymnasiumsAndHHidesIt() {
            var viewer = new ScriptedWindow();
            viewer.Queue(KeyPress(KeyCode.H));
            using var env = new InvertedPendulumEnvV5(renderMode: "human", width: 420, height: 400, viewerFactory: viewer.Factory);
            using (env.Reset(seed: 0)) { }
            MujocoWindow window = env.MujocoRenderer.Window!;
            var expected = new List<(int, string, string)> {
                (MujocoOverlay.TopLeft, MenuLabels(1), MenuValues),
                (MujocoOverlay.BottomLeft, "FPS\nSolver iterations\nStep\ntimestep\n", "60\n1\n0\n0.02000\n"),
            };
            Assert.IsTrue(window.HideMenu, "H was released after the first frame");
            env.Render();
            Assert.AreEqual(0, window.LastMenu.Count, "a hidden menu draws no overlay");
            Assert.AreEqual(2, viewer.Frames.Count);

            // the scene alone, the menu drawn over it as MuJoCo's overlay draws it, then read back as the window shows it
            (int w, int h, byte[] scene) = viewer.Frames[1];
            var target = new GlTarget(w, h, 1, color: true, stencil: false);
            for (int row = 0; row < h; row++) {
                for (int x = 0; x < w; x++) {
                    int o = 4 * ((h - 1 - row) * w + x), p = 3 * (row * w + x);
                    (target.Color![o], target.Color[o + 1], target.Color[o + 2], target.Color[o + 3]) = (scene[p], scene[p + 1], scene[p + 2], 255);
                }
            }

            foreach ((int gridpos, string text1, string text2) in expected)
                MujocoOverlay.Draw(target, gridpos, 0, 0, w, h, text1, text2);
            byte[] first = viewer.Frames[0].Rgb;
            int differing = 0;
            for (int row = 0; row < h; row++) {
                for (int x = 0; x < w; x++) {
                    int o = 4 * ((h - 1 - row) * w + x), p = 3 * (row * w + x);
                    if (first[p] != target.Color![o] || first[p + 1] != target.Color[o + 1] || first[p + 2] != target.Color[o + 2])
                        differing++;
                }
            }

            Assert.AreEqual(0, differing, "the first frame is the scene with Gymnasium's menu drawn by mjr_overlay");
            Assert.IsFalse(viewer.Frames[0].Rgb.SequenceEqual(scene), "the menu shows");
        }

        /// <summary>
        ///     The keys act on release, as <c>_key_callback</c> does: Tab cycles the model's camera and returns to the free one,
        ///     S and F halve and double the run speed, D turns off rendering every frame, C switches contact points and forces
        ///     on, E the bodies' frames, 0 to 4 flip the geom groups, H hides the menu; a key only pressed does nothing.
        /// </summary>
        [TestMethod]
        public void Keys_ActOnReleaseAsGymnasiumsCallback() {
            var viewer = new ScriptedWindow();
            using var env = new HopperEnvV5(renderMode: "human", width: Small, height: Small, viewerFactory: viewer.Factory);
            viewer.Queue(ViewerInput.KeyDown(KeyCode.Tab));
            using (env.Reset(seed: 0)) { }
            MujocoWindow window = env.MujocoRenderer.Window!;
            Assert.AreEqual(MjvCamera.Free, window.Camera.Type, "a pressed key does nothing until released");

            viewer.Queue(ViewerInput.KeyUp(KeyCode.Tab));
            env.Render();
            Assert.AreEqual((MjvCamera.Fixed, 0), (window.Camera.Type, window.Camera.FixedCamId), "Hopper's one camera");
            viewer.Queue(KeyPress(KeyCode.Tab));
            env.Render();
            Assert.AreEqual((MjvCamera.Free, -1), (window.Camera.Type, window.Camera.FixedCamId), "past the last camera, the free one");

            viewer.Queue([.. KeyPress(KeyCode.S), .. KeyPress(KeyCode.S), .. KeyPress(KeyCode.F)]);
            env.Render();
            Assert.AreEqual(0.5, window.RunSpeed);

            viewer.Queue([.. KeyPress(KeyCode.C), .. KeyPress(KeyCode.E), .. KeyPress(KeyCode.Digit0), .. KeyPress(KeyCode.Digit3), .. KeyPress(KeyCode.H)]);
            env.Render();
            Assert.IsTrue(window.Contacts);
            Assert.IsTrue(window.Scene.VisFlags[MujocoScene.VisContactPoint] && window.Scene.VisFlags[MujocoScene.VisContactForce]);
            Assert.AreEqual(MujocoScene.FrameBody, window.Scene.Frame);
            CollectionAssert.AreEqual(new[] { false, true, true, true, false, false }, window.Scene.GeomGroup);
            Assert.IsTrue(window.HideMenu);

            viewer.Queue([.. KeyPress(KeyCode.C), .. KeyPress(KeyCode.E), .. KeyPress(KeyCode.D)]);
            env.Render();
            Assert.IsFalse(window.Scene.VisFlags[MujocoScene.VisContactPoint] || window.Scene.VisFlags[MujocoScene.VisContactForce]);
            Assert.AreEqual(MujocoScene.FrameNone, window.Scene.Frame);
            Assert.IsFalse(window.RenderEveryFrame);
        }

        /// <summary>
        ///     R divides every geom's alpha by 5 in the env's own model, in float32 as NumPy's in-place division does, and a
        ///     second R multiplies them back by 5, so the offscreen modes see the transparency too.
        /// </summary>
        [TestMethod]
        public void R_MakesTheModelsGeomsTransparentInFloat32() {
            var viewer = new ScriptedWindow();
            using var env = new AntEnvV5(renderMode: "human", width: Small, height: Small, viewerFactory: viewer.Factory);
            float[] alphas = Alphas(env);
            viewer.Queue(KeyPress(KeyCode.R));
            using (env.Reset(seed: 0)) { }
            Assert.IsTrue(env.MujocoRenderer.Window!.Transparent);
            CollectionAssert.AreEqual(alphas.Select(a => a / 5f).ToArray(), Alphas(env));
            viewer.Queue(KeyPress(KeyCode.R));
            env.Render();
            Assert.IsFalse(env.MujocoRenderer.Window!.Transparent);
            CollectionAssert.AreEqual(alphas.Select(a => a / 5f * 5f).ToArray(), Alphas(env));

            static float[] Alphas(MujocoEnv e) => Enumerable.Range(0, e.Model.ngeom).Select(i => e.Model.geom_rgba[4 * i + 3]).ToArray();
        }

        /// <summary>
        ///     The mouse drives MuJoCo's camera as <c>_cursor_pos_callback</c> does: a left drag orbits by the move over the
        ///     frame's size (<c>mjMOUSE_ROTATE_V</c>), a right drag pans (<c>MOVE_V</c>, with Shift <c>MOVE_H</c>), the wheel
        ///     zooms by <c>-0.05</c> a notch (<c>ZOOM</c>), a move without a button does nothing, and the positions are truncated
        ///     to whole pixels first.
        /// </summary>
        [TestMethod]
        public void Mouse_DrivesTheCameraThroughMjvMoveCamera() {
            var viewer = new ScriptedWindow();
            using var env = new HalfCheetahEnvV5(renderMode: "human", width: 80, height: Small, viewerFactory: viewer.Factory);
            using (env.Reset(seed: 0)) { }
            MujocoWindow window = env.MujocoRenderer.Window!;
            MjvCamera expected = Clone(window.Camera);

            viewer.Queue(ViewerInput.MouseMove(5, 5), ViewerInput.MouseDown(ViewerInput.LeftButton, 10.7, 20.2), ViewerInput.MouseMove(30.9, 12.5),
                         ViewerInput.MouseUp(ViewerInput.LeftButton, 30.9, 12.5), ViewerInput.MouseMove(60, 60));
            env.Render();
            expected.Move(env.Model, MjvCamera.MouseRotateV, 20 / 80.0, -8 / 64.0);
            AssertSameCamera(expected, window.Camera, "left drag");
            Assert.AreEqual(90 - 45.0, window.Camera.Azimuth, "a quarter of the width turns the camera by 45°");
            Assert.IsFalse(window.ButtonLeftPressed);
            Assert.AreEqual((30, 12), (window.LastMouseX, window.LastMouseY));

            viewer.Queue(ViewerInput.MouseDown(ViewerInput.RightButton, 40, 40), ViewerInput.MouseMove(36, 47), ViewerInput.KeyDown(KeyCode.LeftShift),
                         ViewerInput.MouseMove(30, 50), ViewerInput.KeyUp(KeyCode.LeftShift), ViewerInput.MouseUp(ViewerInput.RightButton, 30, 50),
                         ViewerInput.Scroll(0, 2, 30, 50));
            env.Render();
            expected.Move(env.Model, MjvCamera.MouseMoveV, -4 / 80.0, 7 / 64.0);
            expected.Move(env.Model, MjvCamera.MouseMoveH, -6 / 80.0, 3 / 64.0);
            expected.Move(env.Model, MjvCamera.MouseZoom, 0, -0.05 * 2);
            AssertSameCamera(expected, window.Camera, "right drags, then the wheel");
        }

        /// <summary>
        ///     Space pauses: a paused render keeps drawing frames until → (one frame, still paused) or Space (running again) is
        ///     released, and the menu's lines pile up over those frames as Gymnasium's do (<c>_overlays</c> is cleared only when
        ///     the render returns), cut at MuJoCo's 500 characters when drawn.
        /// </summary>
        [TestMethod]
        public void Pause_WaitsForSpaceOrRightAndThePilesUpTheMenu() {
            var viewer = new ScriptedWindow();
            using var env = new InvertedPendulumEnvV5(renderMode: "human", width: Small, height: Small, viewerFactory: viewer.Factory);
            viewer.Queue(KeyPress(KeyCode.Space));
            using (env.Reset(seed: 0)) { }
            MujocoWindow window = env.MujocoRenderer.Window!;
            Assert.IsTrue(window.Paused);

            // two frames without input, then → on the third
            viewer.Queue();
            viewer.Queue();
            viewer.Queue(KeyPress(KeyCode.Right));
            env.Render();
            Assert.AreEqual(4, viewer.Frames.Count, "the paused render drew until → was released");
            Assert.IsTrue(window.Paused, "→ advances one step and stays paused");
            string paused = "\nRen[d]er every frame\nSwitch camera (#cams = 1)\n[C]ontact forces\nT[r]ansparent\nStart\nAdvance simulation by one step\n" +
                            "Referenc[e] frames\n[H]ide Menu\nCap[t]ure frame\nToggle geomgroup visibility\n";
            Assert.AreEqual(string.Concat(Enumerable.Repeat(paused, 3)), window.LastMenu[0].Text1, "three frames' lines, piled up");

            viewer.Queue(KeyPress(KeyCode.Space));
            env.Render();
            Assert.AreEqual(5, viewer.Frames.Count);
            Assert.IsFalse(window.Paused);
            Assert.AreEqual(paused, window.LastMenu[0].Text1, "a new render starts the menu afresh");

            // → while running pauses after the next frame
            viewer.Queue(KeyPress(KeyCode.Right));
            env.Render();
            Assert.IsTrue(window.Paused);
            viewer.Queue(KeyPress(KeyCode.Space));
            env.Render();
            Assert.AreEqual(7, viewer.Frames.Count);
        }

        /// <summary>
        ///     With rendering every frame off, a render draws only the frames real time owes (<c>_loop_count</c>, the model's
        ///     timestep over the measured frame time and the run speed): any positive debt draws a frame and costs a whole one,
        ///     so at a run speed of 2^30 the first render draws one and the next ones none for a long while.
        /// </summary>
        [TestMethod]
        public void RenderEveryFrameOff_DrawsOnlyTheFramesRealTimeOwes() {
            var viewer = new ScriptedWindow();
            using var env = new InvertedPendulumEnvV5(renderMode: "human", width: Small, height: Small, viewerFactory: viewer.Factory);
            viewer.Queue([.. KeyPress(KeyCode.D), .. Enumerable.Range(0, 30).SelectMany(_ => KeyPress(KeyCode.F))]);
            using (env.Reset(seed: 0)) { }
            MujocoWindow window = env.MujocoRenderer.Window!;
            Assert.AreEqual(Math.Pow(2, 30), window.RunSpeed);
            Assert.AreEqual(0.0, window.LoopCount, "rendering every frame set the debt to one frame, which the reset's frame paid");
            for (int i = 0; i < 5; i++)
                env.Render();
            Assert.AreEqual(2, viewer.Frames.Count, "the first render's tiny debt drew one frame; the next four owe none");
            Assert.IsTrue(window.LoopCount > -1 && window.LoopCount < -1 + 1e-6, $"one frame in advance of real time, got {window.LoopCount}");
        }

        /// <summary>
        ///     T saves the last frame shown, menu included, as an RGB PNG at the image path with the next index, and the next
        ///     frame's menu names the file (<c>Saved as ...</c>).
        /// </summary>
        [TestMethod]
        public void T_CapturesTheLastFrameAndTheMenuNamesIt() {
            string folder = Path.Combine(Path.GetTempPath(), "gymnet-mujoco-window-" + Guid.NewGuid().ToString("N"));
            try {
                var viewer = new ScriptedWindow();
                using var env = new InvertedPendulumEnvV5(renderMode: "human", width: Small, height: 48, viewerFactory: viewer.Factory);
                using (env.Reset(seed: 0)) { }
                MujocoWindow window = env.MujocoRenderer.Window!;
                window.ImagePath = Path.Combine(folder, "frame_{0:D7}.png");
                viewer.Queue(KeyPress(KeyCode.T));
                env.Render();
                string saved = Path.Combine(folder, "frame_0000000.png");
                Assert.AreEqual(1, window.ImageIdx);
                using (Image<Rgb24> png = Image.Load<Rgb24>(saved)) {
                    var bytes = new byte[3 * png.Width * png.Height];
                    png.CopyPixelDataTo(bytes);
                    CollectionAssert.AreEqual(viewer.Frames[1].Rgb, bytes, "the frame shown when T was released");
                }

                env.Render();
                Assert.IsTrue(window.LastMenu[0].Text2.Contains("\n\nSaved as " + saved + "\n"), window.LastMenu[0].Text2);
            } finally {
                if (Directory.Exists(folder))
                    Directory.Delete(folder, recursive: true);
            }
        }

        /// <summary>
        ///     Esc prints Gymnasium's two lines and closes the window; later renders draw nothing, and a paused window's render
        ///     returns instead of waiting for keys that can no longer come.
        /// </summary>
        [TestMethod]
        public void Esc_PrintsGymnasiumsLinesAndClosesTheWindow() {
            var viewer = new ScriptedWindow();
            using var env = new InvertedPendulumEnvV5(renderMode: "human", width: Small, height: Small, viewerFactory: viewer.Factory);
            var printed = new StringWriter();
            using (env.Reset(seed: 0)) { }
            MujocoWindow window = env.MujocoRenderer.Window!;
            window.Out = printed;
            viewer.Queue([.. KeyPress(KeyCode.Space), .. KeyPress(KeyCode.Escape), .. KeyPress(KeyCode.F)]);
            env.Render();
            Assert.AreEqual("Pressed ESC" + Environment.NewLine + "Quitting." + Environment.NewLine, printed.ToString());
            Assert.IsTrue(viewer.Closed);
            Assert.IsFalse(window.IsOpen);
            Assert.AreEqual(1.0, window.RunSpeed, "input after Esc is dropped with the window");
            Assert.IsTrue(window.Paused);
            env.Render();
            Assert.AreEqual(2, viewer.Frames.Count, "a closed window draws nothing, and the paused render returned");
        }

        /// <summary>The close button closes the window at the next frame, which is then not drawn (GLFW's <c>window_should_close</c>).</summary>
        [TestMethod]
        public void CloseButton_ClosesAtTheNextFrame() {
            var viewer = new ScriptedWindow();
            using var env = new InvertedPendulumEnvV5(renderMode: "human", width: Small, height: Small, viewerFactory: viewer.Factory);
            viewer.Queue(ViewerInput.Close());
            using (env.Reset(seed: 0)) { }
            Assert.IsFalse(viewer.Closed, "the close takes effect at the next frame");
            env.Render();
            Assert.IsTrue(viewer.Closed);
            Assert.AreEqual(1, viewer.Frames.Count);
        }

        /// <summary>A resize sets the next frame's size (the framebuffer's), and an empty size, as a minimized window reports, is skipped.</summary>
        [TestMethod]
        public void Resize_SetsTheNextFramesSize() {
            var viewer = new ScriptedWindow();
            using var env = new InvertedPendulumEnvV5(renderMode: "human", width: Small, height: Small, viewerFactory: viewer.Factory);
            viewer.Queue(ViewerInput.Resized(40, 30), ViewerInput.Resized(0, 0));
            using (env.Reset(seed: 0)) { }
            env.Render();
            Assert.AreEqual((40, 30), (viewer.Frames[1].Width, viewer.Frames[1].Height));
        }

        /// <summary>
        ///     The C key's decorations reach the frame's scene as MuJoCo's: on Ant-v5 standing on the floor, a flat cylinder per
        ///     contact and an arrow per contact force; E adds each body's three axes (but the world's).
        /// </summary>
        [TestMethod]
        public void Decorations_ContactsAndFramesJoinTheScene() {
            var viewer = new ScriptedWindow();
            using var env = new AntEnvV5(renderMode: "human", width: Small, height: Small, viewerFactory: viewer.Factory);
            using (env.Reset(seed: 0)) { }
            for (int i = 0; i < 20; i++) {
                using NDArray action = np.zeros(new Shape(8), np.float32);
                using (env.Step(action)) { }
            }

            MujocoWindow window = env.MujocoRenderer.Window!;
            int plain = window.Scene.Geoms.Count;
            Assert.IsTrue(env.Data.ncon > 0, "the ant touches the floor");
            viewer.Queue(KeyPress(KeyCode.C));
            env.Render();
            env.Render();
            List<SceneGeom> decor = window.Scene.Geoms.Where(g => g.Category == MujocoScene.CategoryDecor).ToList();
            Assert.AreEqual(env.Data.ncon, decor.Count(g => g.Type == 5), "a contact point per contact");
            Assert.IsTrue(decor.Any(g => g.Type is MujocoScene.GeomArrow or MujocoScene.GeomArrow2), "the contact forces' arrows");
            Assert.AreEqual(plain + decor.Count, window.Scene.Geoms.Count);

            viewer.Queue([.. KeyPress(KeyCode.C), .. KeyPress(KeyCode.E)]);
            env.Render();
            env.Render();
            Assert.AreEqual(3 * (env.Model.nbody - 1), window.Scene.Geoms.Count(g => g.Category == MujocoScene.CategoryDecor), "three axes per body");
        }

        /// <summary>
        ///     A human render needs a viewer: with neither a <c>viewer_factory</c> nor <c>Gymnasium.HumanViewerFactory</c>,
        ///     Gymnasium's missing-dependency error (DR-20); and a visual option outside <c>mjtVisFlag</c> fails before any window
        ///     opens.
        /// </summary>
        [TestMethod]
        public void Human_WithoutAViewerOrWithABadOption_OpensNothing() {
            EnvViewerFactory? previous = HumanViewerFactory;
            HumanViewerFactory = null;
            try {
                using var env = new ReacherEnvV5(renderMode: "human");
                Assert.ThrowsExactly<DependencyNotInstalled>(() => env.Reset(seed: 0));
            } finally {
                HumanViewerFactory = previous;
            }

            var viewer = new ScriptedWindow();
            using var bad = new ReacherEnvV5(renderMode: "human", visualOptions: new Dictionary<int, bool> { [31] = true }, viewerFactory: viewer.Factory);
            Assert.ThrowsExactly<IndexError>(() => bad.Reset(seed: 0));
            Assert.AreEqual(0, viewer.Opened);
        }

        /// <summary>
        ///     Python's <c>"%.Nf"</c>, which the menu formats the run speed and the timestep with: the exact binary value rounded
        ///     half to even (0.015625 is <c>0.01562</c>, where .NET's <c>"F5"</c> gives <c>0.01563</c>), the sign of a negative
        ///     zero kept, every digit of a large value, and Python's spellings of NaN and the infinities.
        /// </summary>
        [TestMethod]
        public void FormatFixed_IsPythonsPercentF() {
            Assert.AreEqual("0.02000", MujocoWindow.FormatFixed(0.02, 5));
            Assert.AreEqual("0.00300", MujocoWindow.FormatFixed(0.003, 5));
            Assert.AreEqual("1.000", MujocoWindow.FormatFixed(1, 3));
            Assert.AreEqual("0.01562", MujocoWindow.FormatFixed(0.015625, 5));
            Assert.AreEqual("0.03125", MujocoWindow.FormatFixed(0.03125, 5));
            Assert.AreEqual("0.12", MujocoWindow.FormatFixed(0.125, 2));
            Assert.AreEqual("0.38", MujocoWindow.FormatFixed(0.375, 2));
            Assert.AreEqual("2", MujocoWindow.FormatFixed(2.5, 0));
            Assert.AreEqual("4", MujocoWindow.FormatFixed(3.5, 0));
            Assert.AreEqual("-0.000", MujocoWindow.FormatFixed(-0.0, 3));
            Assert.AreEqual("-0.000", MujocoWindow.FormatFixed(-1e-9, 3));
            Assert.AreEqual("0.000", MujocoWindow.FormatFixed(5e-324, 3));
            Assert.AreEqual("9.313", MujocoWindow.FormatFixed(Math.Pow(2, -30) * 1e10, 3));
            Assert.AreEqual("1073741824.000", MujocoWindow.FormatFixed(Math.Pow(2, 30), 3));
            Assert.AreEqual("100000000000000000000.000", MujocoWindow.FormatFixed(1e20, 3));
            Assert.AreEqual("nan", MujocoWindow.FormatFixed(double.NaN, 3));
            Assert.AreEqual("inf", MujocoWindow.FormatFixed(double.PositiveInfinity, 3));
            Assert.AreEqual("-inf", MujocoWindow.FormatFixed(double.NegativeInfinity, 3));
        }

        /// <summary>A key's press and release, as a viewer reports a tap.</summary>
        /// <param name="key">The SDL keycode.</param>
        /// <returns>The two inputs.</returns>
        private static ViewerInput[] KeyPress(int key) => [ViewerInput.KeyDown(key), ViewerInput.KeyUp(key)];

        /// <summary>A copy of a camera's fields.</summary>
        /// <param name="cam">The camera.</param>
        /// <returns>The copy.</returns>
        private static MjvCamera Clone(MjvCamera cam) {
            var copy = new MjvCamera {
                Type = cam.Type, FixedCamId = cam.FixedCamId, TrackBodyId = cam.TrackBodyId, Distance = cam.Distance, Azimuth = cam.Azimuth,
                Elevation = cam.Elevation, Orthographic = cam.Orthographic,
            };
            Array.Copy(cam.Lookat, copy.Lookat, 3);
            return copy;
        }

        /// <summary>Requires two cameras' fields to be equal, bit for bit.</summary>
        /// <param name="expected">The expected camera.</param>
        /// <param name="actual">The window's camera.</param>
        /// <param name="what">What moved it, for the message.</param>
        /// <exception cref="AssertFailedException">A field differs.</exception>
        private static void AssertSameCamera(MjvCamera expected, MjvCamera actual, string what) {
            CollectionAssert.AreEqual(expected.Lookat, actual.Lookat, $"{what}: lookat");
            Assert.AreEqual(expected.Distance, actual.Distance, $"{what}: distance");
            Assert.AreEqual(expected.Azimuth, actual.Azimuth, $"{what}: azimuth");
            Assert.AreEqual(expected.Elevation, actual.Elevation, $"{what}: elevation");
        }

        /// <summary>
        ///     A viewer that records what the window shows and hands out scripted input: one queued batch per frame, read by
        ///     the window's event poll after the frame (an empty one when nothing is queued).
        /// </summary>
        private sealed class ScriptedWindow : IInteractiveEnvViewer {
            /// <summary>The batches still to hand out, oldest first.</summary>
            private readonly ConcurrentQueue<ViewerInput[]> _batches = new();

            /// <summary>Every frame shown: its size and its RGB bytes, top row first.</summary>
            public List<(int Width, int Height, byte[] Rgb)> Frames { get; } = new();

            /// <summary>The width the factory was last asked for.</summary>
            public int Width { get; private set; }

            /// <summary>The height the factory was last asked for.</summary>
            public int Height { get; private set; }

            /// <summary>The title the factory was last given.</summary>
            public string? Title { get; private set; }

            /// <summary>How many windows the factory opened.</summary>
            public int Opened { get; private set; }

            /// <summary>Whether <see cref="BeginInput"/> ran.</summary>
            public bool InputStarted { get; private set; }

            /// <summary>Whether <see cref="Close"/> ran since the last opening.</summary>
            public bool Closed { get; private set; }

            /// <summary>A factory handing out this viewer, recording the size and title.</summary>
            /// <param name="width">The width.</param>
            /// <param name="height">The height.</param>
            /// <param name="title">The title.</param>
            /// <returns>This viewer.</returns>
            public IEnvViewer Factory(int width, int height, string? title = null) {
                (Width, Height, Title, Closed) = (width, height, title, false);
                Opened++;
                return this;
            }

            /// <summary>Queues the input the window's next event poll reads (none for an empty call).</summary>
            /// <param name="batch">The inputs, in order.</param>
            public void Queue(params ViewerInput[] batch) => _batches.Enqueue(batch);

            /// <summary>Records the frame: its size and a copy of its bytes (the window reuses nothing, but the frame is the caller's).</summary>
            /// <param name="frame">The frame, <c>(H, W, 3)</c> uint8.</param>
            public void Render(NDArray frame) => Frames.Add(((int)frame.shape[1], (int)frame.shape[0], frame.ToArray<byte>()));

            /// <summary>Marks the viewer closed.</summary>
            public void Close() => Closed = true;

            /// <summary>Marks input started.</summary>
            public void BeginInput() => InputStarted = true;

            /// <summary>Hands out the next queued batch.</summary>
            /// <returns>The batch, or nothing.</returns>
            public IReadOnlyList<ViewerInput> TakeInput() => _batches.TryDequeue(out ViewerInput[]? batch) ? batch : [];

            /// <summary>Nothing to release.</summary>
            public void Dispose() { }
        }
    }
}
