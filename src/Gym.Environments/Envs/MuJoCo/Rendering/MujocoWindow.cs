using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text;
using Gym.Exceptions;
using Gym.MuJoCo;
using Gym.Rendering;
using Gym.Utils;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Environments.Envs.MuJoCo.Rendering {
    /// <summary>
    ///     A MuJoCo env's <c>render_mode="human"</c> window (Gymnasium's <c>WindowViewer</c>, inlined into
    ///     <see cref="MujocoRenderer"/> per DR-69, with all of its controls per DR-70): each <see cref="Render"/> draws the
    ///     scene from the window's own free camera with the menu on top, shows it in a viewer from the viewer factory, and
    ///     reads the viewer's mouse and keys between frames.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The controls are Gymnasium's, acting when a key is released: Tab cycles the model's cameras and back to the free
    ///     one, Space pauses (a paused <see cref="Render"/> keeps redrawing and returns only when Space or → is released, so
    ///     the env waits), → advances one step and pauses, S and F halve and double the run speed, D toggles rendering every
    ///     frame, C contact points and forces, E the bodies' frames, R transparency (the model's geom alphas divided or
    ///     multiplied by 5, in float32 as NumPy does, so the offscreen frames see it too), 0 to 4 the geom groups, T saves the
    ///     last frame as a PNG, H hides the menu, Esc closes the window (printing <c>Pressed ESC</c> and <c>Quitting.</c>).
    ///     The left button drag orbits the camera, the right one pans it (Shift: horizontally instead of vertically), the
    ///     wheel zooms, all through MuJoCo's <c>mjv_moveCamera</c> (<see cref="MjvCamera.Move"/>).
    ///     </para>
    ///     <para>
    ///     Gymnasium's quirks are kept, since the frames show them: the menu's lines accumulate over the frames of one
    ///     <see cref="Render"/> (a paused window repeats the menu until MuJoCo's 500-character cap cuts it), the run-speed
    ///     arithmetic uses the model's timestep rather than the env's <c>dt</c>, and the camera does not start where the
    ///     offscreen modes' does (MuJoCo's default free camera, then the env's <c>default_cam_config</c>; <c>camera_id</c>
    ///     and <c>camera_name</c> are ignored).
    ///     </para>
    ///     <para>
    ///     Where Gymnasium leaves GLFW's behaviour, Gym.NET reads it as follows (recorded with DR-70): the frame is drawn at
    ///     the viewer's client size in the viewer's own unit, so the pointer needs no framebuffer scale; the window's buffer
    ///     is single-sampled with 24-bit depths, as GLFW's default framebuffer; the swap's wait for the display
    ///     (<c>swap_interval(1)</c>) is a 60 Hz <see cref="Clock"/>; a missing width or height opens a 960 × 540 window,
    ///     half a 1920 × 1080 monitor, since a viewer factory cannot report the monitor's size; T creates the folder of
    ///     Gymnasium's <c>/tmp/frame_%07d.png</c> when it is missing (on Windows, <c>\tmp</c> of the current drive), where
    ///     imageio would fail; and after Esc or the close button the window draws nothing more and a paused
    ///     <see cref="Render"/> returns, where Gymnasium's viewer keeps calling GLFW on the destroyed window.
    ///     </para>
    ///     <para>
    ///     Frames are drawn by Gym.NET's software renderer (DR-62), so a human window shows a few to a few dozen frames a
    ///     second depending on the model and the size, and a paused window keeps a core busy as Gymnasium's keeps the GPU.
    ///     Not thread-safe: <see cref="Render"/> and <see cref="Close"/> belong to the env's thread; the viewer's input is
    ///     queued by its UI thread and read here between frames.
    ///     </para>
    ///     <para>
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/mujoco_rendering.py#L335">gymnasium/envs/mujoco/mujoco_rendering.py</see>.
    ///     </para>
    /// </remarks>
    internal sealed class MujocoWindow {
        /// <summary>The window's title, GLFW's <c>create_window(..., "mujoco", ...)</c>.</summary>
        internal const string Title = "mujoco";

        /// <summary>The size of a window opened without a width or height: half a 1920 × 1080 monitor (Gymnasium halves the monitor's).</summary>
        internal const int FallbackWidth = 960, FallbackHeight = 540;

        /// <summary>The refresh rate the swap waits for (<c>swap_interval(1)</c>), as the common 60 Hz display.</summary>
        private const double RefreshRate = 60;

        /// <summary>The model drawn: the env's, borrowed. Its geom alphas are changed by the R key, as Gymnasium changes them.</summary>
        [NDBorrowed]
        private readonly MjModel _model;

        /// <summary>The simulation drawn: the env's, borrowed.</summary>
        [NDBorrowed]
        private readonly MjData _data;

        /// <summary>The window's drawing context.</summary>
        private readonly MujocoFrame _frame;

        /// <summary>Paces the frames as the swap's wait for the display would.</summary>
        private readonly Clock _clock = new();

        /// <summary>The overlays of the frames since the last <see cref="Render"/> ended, in the order their position was first added (<c>_overlays</c>).</summary>
        private readonly List<(int Gridpos, StringBuilder Text1, StringBuilder Text2)> _overlays = new();

        /// <summary>The open viewer, or <see langword="null"/> once closed.</summary>
        private IEnvViewer? _viewer;

        /// <summary>Whether the user's close button asked the window to close (GLFW's <c>window_should_close</c>), acted on by the next frame.</summary>
        private bool _shouldClose;

        /// <summary>The framebuffer's size: the opening size, then each size the viewer reports.</summary>
        private int _fbWidth, _fbHeight;

        /// <summary>Whether each Shift key is down, from the viewer's key reports (GLFW's <c>get_key</c>).</summary>
        private bool _leftShift, _rightShift;

        /// <summary>The last frame shown, top row first, for the T key's capture; <see langword="null"/> before the first.</summary>
        private byte[]? _lastRgb;

        /// <summary>The last frame's size.</summary>
        private int _lastWidth, _lastHeight;

        /// <summary>
        ///     Opens the window (Gymnasium's <c>WindowViewer.__init__</c>): the visual options switched on its own scene, then a
        ///     viewer of <paramref name="width"/> × <paramref name="height"/> titled <c>mujoco</c>, whose input it starts reading.
        /// </summary>
        /// <param name="model">The model; borrowed.</param>
        /// <param name="data">The simulation; borrowed.</param>
        /// <param name="width">The window's width, or <see langword="null"/> for <see cref="FallbackWidth"/>.</param>
        /// <param name="height">The window's height, or <see langword="null"/> for <see cref="FallbackHeight"/>.</param>
        /// <param name="maxGeom">The most geoms a frame draws.</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch, by <c>mjtVisFlag</c> value.</param>
        /// <param name="viewerFactory">The env's <c>viewer_factory</c>, or <see langword="null"/> for <c>Gymnasium.HumanViewerFactory</c>.</param>
        /// <exception cref="IndexError">A visual option whose index is not an <c>mjtVisFlag</c>, with NumPy's message; no window opens.</exception>
        /// <exception cref="DependencyNotInstalled">No viewer factory is set, where Gymnasium would lack GLFW.</exception>
        /// <exception cref="ArgumentOutOfRangeException">A width or height that is not positive.</exception>
        internal MujocoWindow(MjModel model, MjData data, int? width, int? height, int maxGeom, IReadOnlyDictionary<int, bool> visualOptions,
                              IEnvironmentViewerFactoryDelegate? viewerFactory) {
            _model = model;
            _data = data;
            _fbWidth = width ?? FallbackWidth;
            _fbHeight = height ?? FallbackHeight;
            if (_fbWidth <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), width, "The window's width must be positive.");
            if (_fbHeight <= 0)
                throw new ArgumentOutOfRangeException(nameof(height), height, "The window's height must be positive.");
            Scene.MaxGeom = maxGeom;
            // checked before the window opens, so a bad flag leaves no window behind (Gymnasium's opens first and leaks it)
            foreach ((int flag, bool on) in visualOptions) {
                if ((uint)flag >= (uint)Scene.VisFlags.Length)
                    throw new IndexError($"index {flag} is out of bounds for axis 0 with size {Scene.VisFlags.Length}");
                Scene.VisFlags[flag] = on;
            }

            _frame = new MujocoFrame(model);
            IEnvironmentViewerFactoryDelegate factory = global::Gym.Gymnasium.ResolveHumanViewerFactory(viewerFactory);
            // The factory may start a UI thread and complete asynchronously; render calls are synchronous, as GLFW's are.
            // TODO(async viewer path, DR-53): blocks on the factory; see IEnvironmentViewerFactoryDelegate.
            _viewer = factory(_fbWidth, _fbHeight, Title).GetAwaiter().GetResult();
            if (_viewer is IInteractiveEnvViewer interactive)
                interactive.BeginInput();
        }

        /// <summary>The window's camera (<c>cam</c>): MuJoCo's default free camera until the env's <c>default_cam_config</c> is applied.</summary>
        internal MjvCamera Camera { get; } = new();

        /// <summary>The window's scene and its options (<c>scn</c> and <c>vopt</c>): the C, E and 0–4 keys switch its flags.</summary>
        internal MujocoScene Scene { get; } = new();

        /// <summary>Whether the window is still open: false after Esc, the close button, or <see cref="Close"/>.</summary>
        internal bool IsOpen => _viewer != null;

        /// <summary>Whether the left mouse button is held (<c>_button_left_pressed</c>).</summary>
        internal bool ButtonLeftPressed { get; private set; }

        /// <summary>Whether the right mouse button is held (<c>_button_right_pressed</c>).</summary>
        internal bool ButtonRightPressed { get; private set; }

        /// <summary>The pointer's last whole-pixel x (<c>_last_mouse_x</c>).</summary>
        internal int LastMouseX { get; private set; }

        /// <summary>The pointer's last whole-pixel y (<c>_last_mouse_y</c>).</summary>
        internal int LastMouseY { get; private set; }

        /// <summary>Whether the simulation is paused (<c>_paused</c>): <see cref="Render"/> then waits for Space or →.</summary>
        internal bool Paused { get; private set; }

        /// <summary>Whether the geoms were made transparent by R (<c>_transparent</c>).</summary>
        internal bool Transparent { get; private set; }

        /// <summary>Whether contact points and forces are drawn (<c>_contacts</c>).</summary>
        internal bool Contacts { get; private set; }

        /// <summary>Whether every <see cref="Render"/> draws exactly one frame (<c>_render_every_frame</c>), else as many as real time asks.</summary>
        internal bool RenderEveryFrame { get; private set; } = true;

        /// <summary>Whether → asked for one more frame before pausing again (<c>_advance_by_one_step</c>).</summary>
        internal bool AdvanceByOneStep { get; private set; }

        /// <summary>Whether the menu is hidden (<c>_hide_menu</c>).</summary>
        internal bool HideMenu { get; private set; }

        /// <summary>The number of frames T saved (<c>_image_idx</c>), the next one's index.</summary>
        internal int ImageIdx { get; private set; }

        /// <summary>
        ///     Where T saves frames, a composite format of the index: Gymnasium's <c>"/tmp/frame_%07d.png"</c>. The menu shows
        ///     the path, so changing it changes the frames; tests point it into their own folder.
        /// </summary>
        internal string ImagePath { get; set; } = "/tmp/frame_{0:D7}.png";

        /// <summary>The frames' measured time, an exponential average over the last frames (<c>_time_per_render</c>), from 1/60 s.</summary>
        internal double TimePerRender { get; private set; } = 1 / 60.0;

        /// <summary>The simulation's speed relative to real time (<c>_run_speed</c>), halved by S and doubled by F.</summary>
        internal double RunSpeed { get; private set; } = 1.0;

        /// <summary>The frames still owed to real time (<c>_loop_count</c>), when not every frame is rendered.</summary>
        internal double LoopCount { get; private set; }

        /// <summary>Where Esc's two lines are printed; <see langword="null"/> for the console's standard output, as Python's <c>print</c>.</summary>
        internal TextWriter? Out { get; set; }

        /// <summary>
        ///     The clock a frame's time is measured with (<c>time.time()</c>), in seconds: read when a frame starts and when it
        ///     ends, as Gymnasium reads it, so a test that scripts both clocks gets Gymnasium's frame rates and frame counts.
        /// </summary>
        internal Func<double> Time { get; set; } = () => Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;

        /// <summary>
        ///     The overlays the last frame drew, in drawing order, as the grid position and the two columns' texts (lines
        ///     accumulated over the frames of one <see cref="Render"/> included); empty while the menu is hidden and before the
        ///     first frame. What tests compare with Gymnasium's <c>_create_overlay</c>.
        /// </summary>
        internal IReadOnlyList<(int Gridpos, string Text1, string Text2)> LastMenu { get; private set; } = [];

        /// <summary>
        ///     Draws the simulation's current state (<c>WindowViewer.render</c>): one frame, or while paused frames until Space
        ///     or → is released, or with <see cref="RenderEveryFrame"/> off as many frames as the model's timestep at the run
        ///     speed asks of the measured frame time. Each frame reads the viewer's input after it is shown.
        /// </summary>
        /// <exception cref="ValueError">The simulation's time or timestep is NaN, which the menu's step count cannot round.</exception>
        /// <exception cref="OverflowError">An infinite step count or frame rate in the menu (Python's <c>OverflowError</c>).</exception>
        /// <exception cref="IOException">T could not write its PNG.</exception>
        /// <exception cref="UnauthorizedAccessException">T may not write where <see cref="ImagePath"/> points.</exception>
        /// <exception cref="InvalidOperationException">The camera's near plane is degenerate, or the embedded font is missing.</exception>
        /// <exception cref="ArgumentOutOfRangeException">A camera id or tracked body outside the model (from the env's camera configuration).</exception>
        internal void Render() {
            if (Paused) {
                while (Paused) {
                    Update();
                    if (AdvanceByOneStep) {
                        AdvanceByOneStep = false;
                        break;
                    }

                    // a closed window can never report Space or →, so the wait ends with it
                    if (_viewer == null)
                        break;
                }
            } else {
                LoopCount += _model.timestep / (TimePerRender * RunSpeed);
                if (RenderEveryFrame)
                    LoopCount = 1;
                while (LoopCount > 0) {
                    Update();
                    LoopCount -= 1;
                }
            }

            // the overlays of this call's frames are dropped together, which is why they pile up within one call
            _overlays.Clear();
        }

        /// <summary>
        ///     Closes the window (<c>WindowViewer.close</c>); safe to call twice. The window draws nothing afterwards, and a
        ///     paused <see cref="Render"/> no longer waits.
        /// </summary>
        internal void Close() => CloseViewer();

        /// <summary>
        ///     One frame (<c>update</c>): the menu's lines added, the scene updated and drawn into the window's buffer with the
        ///     overlays on top, shown, paced to the display's rate, then the viewer's input read; the frame's time feeds
        ///     <see cref="TimePerRender"/>.
        /// </summary>
        /// <exception cref="ValueError">A NaN step count.</exception>
        /// <exception cref="OverflowError">An infinite step count or frame rate.</exception>
        /// <exception cref="IOException">T could not write its PNG.</exception>
        /// <exception cref="UnauthorizedAccessException">T may not write its PNG there.</exception>
        /// <exception cref="InvalidOperationException">The camera's near plane is degenerate, or the font is missing.</exception>
        /// <exception cref="ArgumentOutOfRangeException">A camera id or tracked body outside the model.</exception>
        private void Update() {
            // the menu is filled before the closed-window check, as Gymnasium's _create_overlay runs first
            CreateOverlay();
            double start = Time();
            if (_viewer == null)
                return;
            if (_shouldClose) {
                // the close button: Gymnasium destroys the window at the next frame
                CloseViewer();
                return;
            }

            int width = _fbWidth, height = _fbHeight;
            Scene.Update(_model, _data, Camera);
            _frame.Render(Scene, width, height, window: true);
            var drawn = new List<(int Gridpos, string Text1, string Text2)>();
            if (!HideMenu) {
                foreach ((int gridpos, StringBuilder text1, StringBuilder text2) in _overlays) {
                    string first = text1.ToString(), second = text2.ToString();
                    _frame.DrawOverlay(gridpos, first, second);
                    drawn.Add((gridpos, first, second));
                }
            }

            LastMenu = drawn;

            // swap_buffers: the frame as the window shows it, top row first
            var rgb = new byte[3 * width * height];
            _frame.ReadPixels(rgb, null);
            _lastRgb = rgb;
            _lastWidth = width;
            _lastHeight = height;
            using (Image<Rgb24> image = Image.LoadPixelData<Rgb24>(rgb, width, height))
                _viewer.Render(image);
            _clock.Tick(RefreshRate);

            PollEvents();
            TimePerRender = 0.9 * TimePerRender + 0.1 * (Time() - start);
        }

        /// <summary>
        ///     <c>glfw.poll_events()</c>: the viewer's queued input, in its order, through Gymnasium's callbacks (keys act on
        ///     release; Shift is tracked for the pointer; buttons, moves and the wheel drive the camera; a resize sets the next
        ///     frame's size; the close button closes at the next frame). Input after Esc is dropped with the window.
        /// </summary>
        /// <exception cref="IOException">T could not write its PNG.</exception>
        /// <exception cref="UnauthorizedAccessException">T may not write its PNG there.</exception>
        private void PollEvents() {
            if (_viewer is not IInteractiveEnvViewer interactive)
                return;
            foreach (ViewerInput input in interactive.TakeInput()) {
                // GLFW delivers nothing more for a destroyed window
                if (_viewer == null)
                    break;
                switch (input.Kind) {
                    case ViewerInputKind.KeyDown:
                        TrackShift(input.Key, true);
                        break;
                    case ViewerInputKind.KeyUp:
                        TrackShift(input.Key, false);
                        KeyReleased(input.Key);
                        break;
                    case ViewerInputKind.MouseDown:
                    case ViewerInputKind.MouseUp:
                        MouseButton(input);
                        break;
                    case ViewerInputKind.MouseMove:
                        CursorPos(input.X, input.Y);
                        break;
                    case ViewerInputKind.Scroll:
                        Camera.Move(_model, MjvCamera.MouseZoom, 0, -0.05 * input.ScrollY);
                        break;
                    case ViewerInputKind.Resized when input.Width > 0 && input.Height > 0:
                        // a minimized window's empty size is skipped: the next frame keeps the last real one
                        _fbWidth = input.Width;
                        _fbHeight = input.Height;
                        break;
                    case ViewerInputKind.Close:
                        _shouldClose = true;
                        break;
                }
            }
        }

        /// <summary>Records a Shift key's state, which turns the pointer's vertical moves horizontal.</summary>
        /// <param name="key">The SDL keycode.</param>
        /// <param name="down">Whether it went down.</param>
        private void TrackShift(int key, bool down) {
            if (key == KeyCode.LeftShift)
                _leftShift = down;
            else if (key == KeyCode.RightShift)
                _rightShift = down;
        }

        /// <summary>Gymnasium's <c>_key_callback</c> for a released key.</summary>
        /// <param name="key">The SDL keycode; GLFW's key codes of the same keys are the same characters.</param>
        /// <exception cref="IOException">T could not write its PNG.</exception>
        /// <exception cref="UnauthorizedAccessException">T may not write its PNG there.</exception>
        private void KeyReleased(int key) {
            switch (key) {
                case KeyCode.Tab:
                    // the next model camera, then back to the free one
                    Camera.FixedCamId += 1;
                    Camera.Type = MjvCamera.Fixed;
                    if (Camera.FixedCamId >= _model.ncam) {
                        Camera.FixedCamId = -1;
                        Camera.Type = MjvCamera.Free;
                    }

                    break;
                case KeyCode.Space:
                    Paused = !Paused;
                    break;
                case KeyCode.Right:
                    AdvanceByOneStep = true;
                    Paused = true;
                    break;
                case KeyCode.S:
                    RunSpeed /= 2.0;
                    break;
                case KeyCode.F:
                    RunSpeed *= 2.0;
                    break;
                case KeyCode.D:
                    RenderEveryFrame = !RenderEveryFrame;
                    break;
                case KeyCode.T:
                    Capture();
                    break;
                case KeyCode.C:
                    Contacts = !Contacts;
                    Scene.VisFlags[MujocoScene.VisContactPoint] = Contacts;
                    Scene.VisFlags[MujocoScene.VisContactForce] = Contacts;
                    break;
                case KeyCode.E:
                    Scene.Frame = 1 - Scene.Frame;
                    break;
                case KeyCode.H:
                    HideMenu = !HideMenu;
                    break;
                case KeyCode.R:
                    Transparent = !Transparent;
                    // model.geom_rgba[:, 3] /= 5.0 (or *= 5.0): float32 arithmetic in place, so the round trip may not restore
                    // every alpha to its last bit, exactly as in Gymnasium
                    for (int i = 0; i < _model.ngeom; i++) {
                        if (Transparent)
                            _model.geom_rgba[4 * i + 3] /= 5f;
                        else
                            _model.geom_rgba[4 * i + 3] *= 5f;
                    }

                    break;
                case >= KeyCode.Digit0 and <= KeyCode.Digit4:
                    Scene.GeomGroup[key - KeyCode.Digit0] ^= true;
                    break;
            }

            // outside the chain above, as in Gymnasium's callback
            if (key == KeyCode.Escape) {
                TextWriter output = Out ?? Console.Out;
                output.WriteLine("Pressed ESC");
                output.WriteLine("Quitting.");
                CloseViewer();
            }
        }

        /// <summary>
        ///     T (<c>imageio.imwrite(self._image_path % self._image_idx, ...)</c>): the last frame shown, menu included, saved
        ///     as an RGB PNG at <see cref="ImagePath"/> with the next index; the folder is created when missing.
        /// </summary>
        /// <exception cref="IOException">The PNG could not be written.</exception>
        /// <exception cref="UnauthorizedAccessException">The folder or file may not be written.</exception>
        private void Capture() {
            if (_lastRgb == null)
                return;
            string path = string.Format(CultureInfo.InvariantCulture, ImagePath, ImageIdx);
            string? folder = Path.GetDirectoryName(Path.GetFullPath(path));
            if (folder != null)
                Directory.CreateDirectory(folder);
            using (Image<Rgb24> image = Image.LoadPixelData<Rgb24>(_lastRgb, _lastWidth, _lastHeight))
                image.SaveAsPng(path);
            ImageIdx += 1;
        }

        /// <summary>
        ///     Gymnasium's <c>_mouse_button_callback</c>: the left and right buttons' states, and the pointer's position as the
        ///     drag's starting point (whole pixels, truncated as Python's <c>int</c>).
        /// </summary>
        /// <param name="input">The button report.</param>
        private void MouseButton(ViewerInput input) {
            bool down = input.Kind == ViewerInputKind.MouseDown;
            if (input.Button == ViewerInput.LeftButton)
                ButtonLeftPressed = down;
            else if (input.Button == ViewerInput.RightButton)
                ButtonRightPressed = down;
            LastMouseX = (int)input.X;
            LastMouseY = (int)input.Y;
        }

        /// <summary>
        ///     Gymnasium's <c>_cursor_pos_callback</c>: with a button held, the move since the last position as fractions of
        ///     the frame turns the camera (the right button pans, the left orbits; Shift makes either horizontal).
        /// </summary>
        /// <param name="xpos">The pointer's x.</param>
        /// <param name="ypos">The pointer's y.</param>
        private void CursorPos(double xpos, double ypos) {
            if (!(ButtonLeftPressed || ButtonRightPressed))
                return;
            bool shift = _leftShift || _rightShift;
            int action = ButtonRightPressed
                ? shift ? MjvCamera.MouseMoveH : MjvCamera.MouseMoveV
                : shift ? MjvCamera.MouseRotateH : MjvCamera.MouseRotateV;
            int x = (int)xpos, y = (int)ypos;
            int dx = x - LastMouseX, dy = y - LastMouseY;
            Camera.Move(_model, action, dx / (double)_fbWidth, dy / (double)_fbHeight);
            LastMouseX = x;
            LastMouseY = y;
        }

        /// <summary>
        ///     Gymnasium's <c>_create_overlay</c>: the menu's lines at the top left (labels and their keys or states) and the
        ///     statistics at the bottom left (frame rate, solver iterations, step, timestep), added to what earlier frames of
        ///     the same <see cref="Render"/> added.
        /// </summary>
        /// <exception cref="ValueError">A NaN step count.</exception>
        /// <exception cref="OverflowError">An infinite step count or frame rate.</exception>
        private void CreateOverlay() {
            const int topleft = MujocoOverlay.TopLeft, bottomleft = MujocoOverlay.BottomLeft;
            if (RenderEveryFrame)
                AddOverlay(topleft, "", "");
            else
                AddOverlay(topleft, $"Run speed = {FormatFixed(RunSpeed, 3)} x real time", "[S]lower, [F]aster");
            AddOverlay(topleft, "Ren[d]er every frame", RenderEveryFrame ? "On" : "Off");
            AddOverlay(topleft, $"Switch camera (#cams = {Int(_model.ncam + 1)})", $"[Tab] (camera ID = {Int(Camera.FixedCamId)})");
            AddOverlay(topleft, "[C]ontact forces", Contacts ? "On" : "Off");
            AddOverlay(topleft, "T[r]ansparent", Transparent ? "On" : "Off");
            if (!Paused) {
                AddOverlay(topleft, "Stop", "[Space]");
            } else {
                AddOverlay(topleft, "Start", "[Space]");
                AddOverlay(topleft, "Advance simulation by one step", "[right arrow]");
            }

            AddOverlay(topleft, "Referenc[e] frames", Scene.Frame == 1 ? "On" : "Off");
            AddOverlay(topleft, "[H]ide Menu", "");
            if (ImageIdx > 0)
                AddOverlay(topleft, "Cap[t]ure frame", "Saved as " + string.Format(CultureInfo.InvariantCulture, ImagePath, ImageIdx - 1));
            else
                AddOverlay(topleft, "Cap[t]ure frame", "");
            AddOverlay(topleft, "Toggle geomgroup visibility", "0-4");

            // "%d" of a float truncates toward zero, as Python's int()
            AddOverlay(bottomleft, "FPS", PyBuiltins.Int(1 / TimePerRender).ToString(CultureInfo.InvariantCulture));
            AddOverlay(bottomleft, "Solver iterations", Int(_data.solver_niter_island[0] + 1));
            AddOverlay(bottomleft, "Step", PyMath.Round(_data.time / _model.timestep).ToString(CultureInfo.InvariantCulture));
            AddOverlay(bottomleft, "timestep", FormatFixed(_model.timestep, 5));

            static string Int(int value) => value.ToString(CultureInfo.InvariantCulture);
        }

        /// <summary>Gymnasium's <c>add_overlay</c>: appends a line to each column of a grid position, creating the position on first use.</summary>
        /// <param name="gridpos">The grid position.</param>
        /// <param name="text1">The first column's line.</param>
        /// <param name="text2">The second column's line.</param>
        private void AddOverlay(int gridpos, string text1, string text2) {
            int i = _overlays.FindIndex(o => o.Gridpos == gridpos);
            if (i < 0) {
                _overlays.Add((gridpos, new StringBuilder(), new StringBuilder()));
                i = _overlays.Count - 1;
            }

            _overlays[i].Text1.Append(text1).Append('\n');
            _overlays[i].Text2.Append(text2).Append('\n');
        }

        /// <summary>Closes the viewer once; later frames draw nothing.</summary>
        private void CloseViewer() {
            IEnvViewer? viewer = _viewer;
            _viewer = null;
            viewer?.CloseEnvironment();
        }

        /// <summary>
        ///     Python's <c>"%.{digits}f" % x</c>: the exact binary value rounded half to even at <paramref name="digits"/>
        ///     decimals, the sign kept for a negative zero, <c>nan</c>, <c>inf</c> and <c>-inf</c> for the others. .NET's
        ///     <c>"F"</c> format rounds an exact half away from zero instead.
        /// </summary>
        /// <param name="x">The value.</param>
        /// <param name="digits">The decimals, 0 or more.</param>
        /// <returns>The text, for example <c>0.01000</c> or <c>1.000</c>.</returns>
        internal static string FormatFixed(double x, int digits) {
            if (double.IsNaN(x))
                return "nan";
            if (double.IsInfinity(x))
                return x > 0 ? "inf" : "-inf";
            long bits = BitConverter.DoubleToInt64Bits(x);
            int exponent = (int)((bits >> 52) & 0x7FF);
            long fraction = bits & 0xFFFFFFFFFFFFFL;
            // the value is mantissa * 2^power exactly; a subnormal has no hidden bit and the smallest exponent
            BigInteger mantissa = exponent == 0 ? fraction : fraction | (1L << 52);
            int power = (exponent == 0 ? 1 : exponent) - 1075;
            BigInteger scaled = mantissa * BigInteger.Pow(10, digits), units;
            if (power >= 0) {
                units = scaled << power;
            } else {
                BigInteger denominator = BigInteger.One << -power;
                units = BigInteger.DivRem(scaled, denominator, out BigInteger remainder);
                int half = (remainder * 2).CompareTo(denominator);
                if (half > 0 || (half == 0 && !units.IsEven))
                    units += 1;
            }

            string text = units.ToString(CultureInfo.InvariantCulture).PadLeft(digits + 1, '0');
            string sign = bits < 0 ? "-" : "";
            return digits == 0 ? sign + text : sign + text[..^digits] + "." + text[^digits..];
        }
    }
}
