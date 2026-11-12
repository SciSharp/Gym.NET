using System;
using System.Collections.Generic;
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
        ///     Draws one frame in <paramref name="renderMode"/> (<c>render</c>).
        /// </summary>
        /// <param name="renderMode">The mode; <see langword="null"/> is refused, as in Gymnasium.</param>
        /// <returns>
        ///     <c>rgb_array</c>: an <c>(height, width, 3)</c> uint8 frame; <c>depth_array</c>: an <c>(height, width)</c> float32
        ///     depth; <c>rgbd_tuple</c>: both; <c>human</c>: <see cref="RenderResult.None"/> after showing the frame.
        /// </returns>
        /// <exception cref="AssertionError">A mode other than <c>human</c> without a width or height, with Gymnasium's message.</exception>
        /// <exception cref="AttributeError">An unknown mode, or none, with Gymnasium's message.</exception>
        /// <exception cref="NotImplementedError">The frame itself: Gym.NET's MuJoCo rasterizer (DR-62) is not written yet.</exception>
        public RenderResult Render(string? renderMode) {
            if (renderMode != "human" && (Width == null || Height == null))
                throw new AssertionError(
                    $"The width: {PyBuiltins.Repr(Width)} and height: {PyBuiltins.Repr(Height)} cannot be `None` when the render_mode is not `human`.");
            if (renderMode is not ("human" or "rgb_array" or "depth_array" or "rgbd_tuple"))
                throw new AttributeError(
                    $"Unexpected mode: {renderMode ?? "None"}, expected modes: human, rgb_array, depth_array, or rgbd_tuple");
            throw new NotImplementedError("Gym.NET's MuJoCo frames (DR-62) are not drawn yet");
        }

        /// <summary>Closes every window and buffer the modes opened (<c>close</c>); drawing again opens them again.</summary>
        public void Close() { }
    }
}
