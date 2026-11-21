using System;
using System.Collections.Generic;
using Gym.Environments.Envs.MuJoCo.Rendering;
using Gym.Envs;
using Gym.Exceptions;
using Gym.MuJoCo;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Draws a MuJoCo env's frames (<c>MujocoRenderer</c>): the camera resolved as Gymnasium resolves it (a named camera,
    ///     an id, or the free camera with the env's default configuration), and the render modes <c>rgb_array</c>,
    ///     <c>depth_array</c>, <c>rgbd_tuple</c> and <c>human</c>. Gym.NET draws the scene itself (DR-62); its frames are
    ///     compared with Gymnasium's and the differences measured, not required equal (DR-67).
    /// </summary>
    /// <remarks>
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/mujoco_rendering.py#L701">gymnasium/envs/mujoco/mujoco_rendering.py</see>.
    /// </remarks>
    [PyName("MujocoRenderer")]
    public sealed class MujocoRenderer {
        /// <summary>
        ///     Resolves the camera and keeps the settings (Gymnasium's <c>MujocoRenderer.__init__</c>); nothing is drawn until
        ///     <see cref="Render"/>.
        /// </summary>
        /// <param name="model">The model to draw.</param>
        /// <param name="data">The simulation to draw.</param>
        /// <param name="defaultCamConfig">The free camera's settings (<c>default_cam_config</c>), or <see langword="null"/>.</param>
        /// <param name="width">The frame's width, or <see langword="null"/> (only <c>human</c> mode allows it).</param>
        /// <param name="height">The frame's height, or <see langword="null"/>.</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="cameraId">The camera id (<c>camera_id</c>), or <see langword="null"/>.</param>
        /// <param name="cameraName">The camera name (<c>camera_name</c>), or <see langword="null"/>.</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>), or <see langword="null"/>.</param>
        /// <param name="viewerFactory">
        ///     Gym.NET's <c>viewer_factory</c> (DR-20): the window <c>human</c> mode shows frames in, or <see langword="null"/>
        ///     for <c>Gymnasium.HumanViewerFactory</c>.
        /// </param>
        /// <exception cref="ArgumentNullException">A null model or data.</exception>
        /// <exception cref="ValueError">Both a camera id and a camera name, with Gymnasium's message.</exception>
        /// <remarks>
        ///     With neither camera given, Gymnasium looks up a camera named <c>track</c> and then uses the free camera anyway
        ///     (id -1), so the env's default camera configuration takes effect; a named or numbered camera is used as given.
        /// </remarks>
        public MujocoRenderer(MjModel model, MjData data, IReadOnlyDictionary<string, object?>? defaultCamConfig = null, int? width = null,
                              int? height = null, int maxGeom = 1000, int? cameraId = null, string? cameraName = null,
                              IReadOnlyDictionary<int, bool>? visualOptions = null, IEnvironmentViewerFactoryDelegate? viewerFactory = null) {
            Model = model ?? throw new ArgumentNullException(nameof(model));
            Data = data ?? throw new ArgumentNullException(nameof(data));
            DefaultCamConfig = defaultCamConfig;
            Width = width;
            Height = height;
            MaxGeom = maxGeom;
            ViewerFactory = viewerFactory;
            VisualOptions = visualOptions ?? new Dictionary<int, bool>();
            if (cameraId != null && cameraName != null)
                throw new ValueError("Both `camera_id` and `camera_name` cannot be specified at the same time.");
            bool noCameraSpecified = cameraName == null && cameraId == null;
            if (noCameraSpecified)
                cameraName = "track";
            if (cameraId == null) {
                CameraId = Mujoco.MjName2Id(Model, MjtObj.Camera, cameraName!);
                // a model's own "track" camera would be used as a fixed camera and ignore the env's default camera
                // configuration, so Gymnasium forces the free camera when none was asked for
                if (noCameraSpecified && CameraId != -1)
                    CameraId = -1;
            } else {
                CameraId = cameraId.Value;
            }
        }

        /// <summary>The model drawn (<c>model</c>), the env's: borrowed, never disposed here.</summary>
        [NDBorrowed]
        public MjModel Model { get; }

        /// <summary>The simulation drawn (<c>data</c>), the env's: borrowed, never disposed here.</summary>
        [NDBorrowed]
        public MjData Data { get; }

        /// <summary>The free camera's settings (<c>default_cam_config</c>), applied when a mode's first frame is drawn.</summary>
        public IReadOnlyDictionary<string, object?>? DefaultCamConfig { get; }

        /// <summary>The frame's width in pixels (<c>width</c>).</summary>
        public int? Width { get; }

        /// <summary>The frame's height in pixels (<c>height</c>).</summary>
        public int? Height { get; }

        /// <summary>The most geoms a frame draws (<c>max_geom</c>).</summary>
        public int MaxGeom { get; }

        /// <summary>The camera the frames are drawn from (<c>camera_id</c>): -1 for the free camera, else a model camera.</summary>
        public int CameraId { get; }

        /// <summary>The visual flags switched for the frames (<c>_vopt</c>), by <c>mjtVisFlag</c> value.</summary>
        internal IReadOnlyDictionary<int, bool> VisualOptions { get; }

        /// <summary>The window factory <c>human</c> mode uses (DR-20), or <see langword="null"/> for the app-wide one.</summary>
        internal IEnvironmentViewerFactoryDelegate? ViewerFactory { get; }

        /// <summary>
        ///     One render mode's viewer (Gymnasium's <c>OffScreenViewer</c>, inlined per DR-69): its own scene, camera and
        ///     drawing context, created with the first frame of that mode.
        /// </summary>
        private sealed class Viewer {
            /// <summary>The scene, kept between frames as MuJoCo's <c>MjvScene</c> is.</summary>
            internal readonly MujocoScene Scene = new();

            /// <summary>The abstract camera.</summary>
            internal readonly MjvCamera Camera = new();

            /// <summary>The drawing context.</summary>
            internal required MujocoFrame Frame;
        }

        /// <summary>The viewers by render mode (<c>_viewers</c>), but for <c>human</c>'s.</summary>
        private readonly Dictionary<string, Viewer> _viewers = new(StringComparer.Ordinal);

        /// <summary>The <c>human</c> mode's window (<c>_viewers["human"]</c>, Gymnasium's <c>WindowViewer</c>), opened by its first frame.</summary>
        private MujocoWindow? _window;

        /// <summary>
        ///     Draws one frame in <paramref name="renderMode"/> (<c>render</c>): the scene from the camera (the free camera
        ///     with the env's default configuration, or the chosen model camera), drawn by Gym.NET's own rasterizer (DR-62);
        ///     in <c>human</c> mode, into the window with Gymnasium's menu and controls (DR-70).
        /// </summary>
        /// <param name="renderMode">The mode; <see langword="null"/> is refused, as in Gymnasium.</param>
        /// <returns>
        ///     <c>rgb_array</c>: an <c>(height, width, 3)</c> uint8 frame; <c>depth_array</c>: an <c>(height, width)</c> float32
        ///     depth (0 at the near plane, 1 at the far one, as MuJoCo reads it back); <c>rgbd_tuple</c>: both;
        ///     <c>human</c>: <see cref="RenderResult.None"/> after showing the frame. The caller owns the arrays.
        /// </returns>
        /// <exception cref="AssertionError">A mode other than <c>human</c> without a width or height, with Gymnasium's message.</exception>
        /// <exception cref="AttributeError">An unknown mode, or none, with Gymnasium's message.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The camera id names no model camera; or a human window's width or height is not positive.</exception>
        /// <exception cref="DependencyNotInstalled"><c>human</c> mode with neither a <c>viewer_factory</c> nor <c>Gymnasium.HumanViewerFactory</c> (DR-20).</exception>
        /// <exception cref="ValueError"><c>human</c> mode with a NaN simulation time or timestep, which the menu's step count cannot round.</exception>
        /// <exception cref="OverflowError"><c>human</c> mode with an infinite step count in the menu (Python's <c>OverflowError</c>).</exception>
        /// <exception cref="System.IO.IOException">The human window's T key could not write its PNG.</exception>
        /// <exception cref="UnauthorizedAccessException">The human window's T key may not write its PNG there.</exception>
        /// <exception cref="InvalidCastException">A camera configuration value of a kind the camera field cannot take.</exception>
        /// <exception cref="IndexError">A visual option whose flag index is not an <c>mjtVisFlag</c>, with NumPy's message.</exception>
        /// <remarks>
        ///     <para>
        ///     Frames are measured against Gymnasium's GPU frames, not required equal (DR-67, DR-71): lighting, textures,
        ///     the floor's reflection, shadows and multisampling follow MuJoCo's classic renderer in software, with the
        ///     reference GPU's measured rounding, and the rest of the GPU's choices still decide some pixels at edges and on
        ///     distant textures (on 2026-11-12, 0.07% of InvertedPendulum-v5's pixels to 16% of Humanoid-v5's, nearly all by
        ///     one level; depths bit for bit but for a few ulps). A 480×480 frame costs about 0.05 to 0.5 s of one desktop
        ///     core's time spread over all cores, the shadow map (4096² depths per shadow light) a good part of it, so an
        ///     <c>rgb_array</c> loop renders far slower than it steps.
        ///     </para>
        ///     <para>
        ///     <c>human</c> mode opens a viewer from the env's <c>viewer_factory</c> or <c>Gymnasium.HumanViewerFactory</c>
        ///     titled <c>mujoco</c>, at the renderer's width and height, and keeps Gymnasium's controls: Space pauses, and
        ///     while paused this call returns only when Space or → is released in the window (see the window's controls in
        ///     the MuJoCo guide). Its camera starts at MuJoCo's default free camera with the env's
        ///     <see cref="DefaultCamConfig"/>; <see cref="CameraId"/> is not used, as in Gymnasium.
        ///     </para>
        /// </remarks>
        public RenderResult Render(string? renderMode) {
            if (renderMode != "human" && (Width == null || Height == null))
                throw new AssertionError(
                    $"The width: {PyBuiltins.Repr(Width)} and height: {PyBuiltins.Repr(Height)} cannot be `None` when the render_mode is not `human`.");
            if (renderMode is not ("human" or "rgb_array" or "depth_array" or "rgbd_tuple"))
                throw new AttributeError(
                    $"Unexpected mode: {renderMode ?? "None"}, expected modes: human, rgb_array, depth_array, or rgbd_tuple");
            if (renderMode == "human") {
                // WindowViewer.render: no camera id, the window's own camera as the user moved it
                GetWindow().Render();
                return RenderResult.None;
            }

            Viewer viewer = GetViewer(renderMode);
            int width = Width!.Value, height = Height!.Value;

            // OffScreenViewer.render: the camera id is applied on every frame
            if (CameraId == -1) {
                viewer.Camera.Type = MjvCamera.Free;
            } else {
                viewer.Camera.Type = MjvCamera.Fixed;
            }

            viewer.Camera.FixedCamId = CameraId;
            viewer.Scene.Update(Model, Data, viewer.Camera);
            viewer.Frame.Render(viewer.Scene, width, height);

            bool wantRgb = renderMode is "rgb_array" or "rgbd_tuple", wantDepth = renderMode is "depth_array" or "rgbd_tuple";
            byte[]? rgb = wantRgb ? new byte[3 * width * height] : null;
            float[]? depth = wantDepth ? new float[width * height] : null;
            viewer.Frame.ReadPixels(rgb, depth);
            NDArray? rgbFrame = rgb == null ? null : np.array(rgb).reshape(height, width, 3);
            NDArray? depthFrame = depth == null ? null : np.array(depth).reshape(height, width);
            return renderMode switch {
                "rgb_array" => new RenderResult.Rgb(rgbFrame!),
                "depth_array" => new RenderResult.Depth(depthFrame!),
                _ => new RenderResult.RgbDepth(rgbFrame!, depthFrame!),
            };
        }

        /// <summary>
        ///     The viewer of a mode, created on its first frame: the free camera aimed at the geoms' median position from the
        ///     model's extent (<c>_init_camera</c>), then the env's default camera configuration (<c>_set_cam_config</c>).
        /// </summary>
        /// <param name="renderMode">The mode.</param>
        /// <returns>The viewer.</returns>
        /// <exception cref="InvalidCastException">A camera configuration value of a kind the camera field cannot take.</exception>
        /// <exception cref="IndexError">A visual option whose flag index is not an <c>mjtVisFlag</c>, with NumPy's message.</exception>
        private Viewer GetViewer(string renderMode) {
            if (_viewers.TryGetValue(renderMode, out Viewer? viewer))
                return viewer;
            viewer = new Viewer { Frame = new MujocoFrame(Model) };
            viewer.Scene.MaxGeom = MaxGeom;
            // BaseRender.__init__: the visual options switch the viewer's own flags once, over MuJoCo's defaults; a flag
            // index outside mjtVisFlag fails as MjvOption's flags array refuses it
            foreach ((int flag, bool on) in VisualOptions) {
                if ((uint)flag >= (uint)viewer.Scene.VisFlags.Length)
                    throw new IndexError($"index {flag} is out of bounds for axis 0 with size {viewer.Scene.VisFlags.Length}");
                viewer.Scene.VisFlags[flag] = on;
            }

            MjvCamera cam = viewer.Camera;
            cam.Type = MjvCamera.Free;
            cam.FixedCamId = -1;
            int ngeom = Model.ngeom;
            var column = new double[ngeom];
            for (int i = 0; i < 3; i++) {
                for (int g = 0; g < ngeom; g++)
                    column[g] = Data.geom_xpos[3 * g + i];
                cam.Lookat[i] = Median(column);
            }

            cam.Distance = Model.extent;
            SetCamConfig(cam);

            _viewers[renderMode] = viewer;
            return viewer;

            // NumPy's median: the middle value, or the mean of the two middle values
            static double Median(double[] values) {
                var sorted = (double[])values.Clone();
                Array.Sort(sorted);
                int n = sorted.Length;
                return n == 0 ? double.NaN : n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
            }
        }

        /// <summary>
        ///     The <c>human</c> mode's window, opened on its first frame (<c>_get_viewer("human")</c>): Gymnasium's
        ///     <c>WindowViewer</c> leaves MuJoCo's default free camera as it is (no <c>_init_camera</c>), then the env's default
        ///     camera configuration applies (<c>_set_cam_config</c>).
        /// </summary>
        /// <returns>The window.</returns>
        /// <exception cref="IndexError">A visual option whose flag index is not an <c>mjtVisFlag</c>; no window opens.</exception>
        /// <exception cref="DependencyNotInstalled">No viewer factory is set; a later frame tries again.</exception>
        /// <exception cref="ArgumentOutOfRangeException">A width or height that is not positive.</exception>
        /// <exception cref="InvalidCastException">A camera configuration value of a kind the camera field cannot take.</exception>
        private MujocoWindow GetWindow() {
            if (_window != null)
                return _window;
            var window = new MujocoWindow(Model, Data, Width, Height, MaxGeom, VisualOptions, ViewerFactory);
            if (WindowClock != null)
                window.Time = WindowClock;
            try {
                SetCamConfig(window.Camera);
            } catch {
                // a configuration the camera refuses leaves no window open behind the exception
                window.Close();
                throw;
            }

            return _window = window;
        }

        /// <summary>
        ///     Applies the env's <see cref="DefaultCamConfig"/> to a camera (<c>_set_cam_config</c>): each known key sets the
        ///     camera field of its name; other keys are ignored.
        /// </summary>
        /// <param name="cam">The camera.</param>
        /// <exception cref="InvalidCastException">A value of a kind the camera field cannot take.</exception>
        private void SetCamConfig(MjvCamera cam) {
            if (DefaultCamConfig == null)
                return;
            foreach ((string key, object? value) in DefaultCamConfig) {
                switch (key) {
                    case "lookat":
                        double[] point = value as double[] ?? throw new InvalidCastException($"default_cam_config's lookat must be three numbers, got {value}.");
                        Array.Copy(point, cam.Lookat, 3);
                        break;
                    case "distance":
                        cam.Distance = Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
                        break;
                    case "azimuth":
                        cam.Azimuth = Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
                        break;
                    case "elevation":
                        cam.Elevation = Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
                        break;
                    case "trackbodyid":
                        cam.TrackBodyId = Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture);
                        break;
                    case "fixedcamid":
                        cam.FixedCamId = Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture);
                        break;
                    case "type":
                        cam.Type = Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture);
                        break;
                    case "orthographic":
                        cam.Orthographic = Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture) != 0;
                        break;
                }
            }
        }

        /// <summary>
        ///     Closes every window and buffer the modes opened (<c>close</c>): the human window shuts (a paused frame loop
        ///     cannot be running, since it holds the env's thread); drawing again opens them again.
        /// </summary>
        public void Close() {
            _window?.Close();
            _window = null;
            _viewers.Clear();
        }

        /// <summary>The <c>human</c> window, or <see langword="null"/> before its first frame and after <see cref="Close"/>; for tests.</summary>
        internal MujocoWindow? Window => _window;

        /// <summary>
        ///     The clock the <c>human</c> window measures its frames with, given to it when it opens; <see langword="null"/> for
        ///     the stopwatch. A parity test sets it before the first frame, so its frame times and counts are scripted like
        ///     Gymnasium's (<see cref="MujocoWindow.Time"/>).
        /// </summary>
        internal Func<double>? WindowClock { get; set; }
    }
}
