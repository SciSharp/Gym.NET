using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Gym.Environments;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Image = SixLabors.ImageSharp.Image;

namespace Gym.Rendering.WinForm {
    /// <summary>
    ///     A Windows Forms window that shows an env's frames in a picture box, on a UI thread of its own: envs open it through
    ///     <see cref="Factory"/> (<see cref="Open"/>), an app that awaits through <see cref="Run"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Once <see cref="BeginInput"/> runs it is an <see cref="IInteractiveEnvViewer"/>: it reports keys as SDL keycodes
    ///     (<see cref="KeyCode"/>), its close button (which then no longer closes it) and the user's resizes, so
    ///     <c>Gym.Utils.Play</c> can play an env in it. A key is read from Windows' virtual key (<see cref="Keys"/>), which
    ///     follows the keyboard layout for letters, as pygame's keycodes do (DR-47): the key labeled A gives <c>K_a</c> on any
    ///     layout. Where a layout types another character unshifted than its virtual key names (AZERTY's digit row types
    ///     <c>&amp;</c>, <c>é</c>, ...), SDL reports that character and this viewer the virtual key (<c>K_1</c>). A held key
    ///     reports one press, not the system's auto-repeat, as pygame's default. WinForms reports Shift, Ctrl and Alt without
    ///     their side, so they are reported as the left ones. The mouse is reported too (DR-70: buttons, moves and the
    ///     wheel, in the client area's pixels, for a MuJoCo env's camera controls). Before <see cref="BeginInput"/> nothing
    ///     changes.
    ///     </para>
    ///     <para>
    ///     It renders both ways the viewer contract offers (DR-85): <see cref="Render"/> waits for the form's thread to show
    ///     the frame, <see cref="RenderAsync"/> awaits it. Closing through the contract (<see cref="IEnvViewer.Close"/>)
    ///     closes and disposes the form and ends its thread; footgun: <see cref="Form.Close"/>, which a variable typed
    ///     <c>WinFormEnvViewer</c> calls as <c>viewer.Close()</c>, is WinForms' own: it leaves the thread running and, with
    ///     input on, is reported as the user's close.
    ///     </para>
    /// </remarks>
    public partial class WinFormEnvViewer : Form, IInteractiveEnvViewer, IAsyncEnvViewer {
        /// <summary>The size of the last frame's BMP stream, to size the next one's buffer.</summary>
        private int _lastSize = 0;

        /// <summary>Set once the form is shown (<see cref="OnShown"/>); released as the form closes.</summary>
        private readonly ManualResetEventSlim _ready = new ManualResetEventSlim();

        /// <summary>The input reported since the last <see cref="TakeInput"/>, filled on the form's thread.</summary>
        private readonly ConcurrentQueue<ViewerInput> _input = new ConcurrentQueue<ViewerInput>();

        /// <summary>The keys held now, so the system's auto-repeat reports no second press (form's thread only).</summary>
        private readonly HashSet<Keys> _held = new HashSet<Keys>();

        /// <summary>Whether input is reported (<see cref="BeginInput"/> ran); read on the form's thread, set on any.</summary>
        private volatile bool _inputEnabled;

        /// <summary>Set by <see cref="IEnvViewer.Close"/>, whose <see cref="Form.Close"/> WinForms reports as a user's close too.</summary>
        private volatile bool _closingByCode;

        /// <summary>Set once the form is shown (<see cref="OnShown"/>), so the sizes of its first layout are not reported as resizes.</summary>
        private volatile bool _shown;

        /// <summary>
        ///     The factory envs open this window with (DR-20, DR-85): <see cref="Open"/>, which starts the form on its own UI
        ///     thread and returns once it exists.
        /// </summary>
        public static EnvViewerFactory Factory => Open;

        /// <summary>
        ///     Opens a <see cref="WinFormEnvViewer"/> on a UI thread of its own and returns it once the form exists (the
        ///     synchronous factory envs call, DR-85), its client area the frame's size, so a frame of that size shows whole at
        ///     one screen pixel per pixel.
        /// </summary>
        /// <param name="width">The width of the frames, the form's client width.</param>
        /// <param name="height">The height of the frames, the form's client height.</param>
        /// <param name="title">The title of the form, also mentioned in the thread name.</param>
        /// <returns>The viewer; its STA thread then runs the form's message loop until the form closes.</returns>
        /// <exception cref="TimeoutException">Thrown when the form did not come up within 10 seconds.</exception>
        /// <remarks>It waits on an event the form's thread sets, never on a task, and the form's thread never waits for the caller (R8).</remarks>
        public static IEnvViewer Open(int width, int height, string title = null) {
            WinFormEnvViewer created = null;
            using var ready = new ManualResetEventSlim();
            Start(width, height, title, viewer => {
                created = viewer;
                ready.Set();
            });
            if (!ready.Wait(10_000))
                throw new TimeoutException("Starting viewer timed out.");
            return created;
        }

        /// <summary>
        ///     Starts a <see cref="WinFormEnvViewer"/> in seperate thread, its client area the frame's size, so a frame of that
        ///     size shows whole at one screen pixel per pixel: the asynchronous way to open one for an app that awaits. Envs
        ///     use the synchronous <see cref="Open"/> (DR-85).
        /// </summary>
        /// <param name="height">The height of the frames, the form's client height</param>
        /// <param name="width">The width of the frames, the form's client width</param>
        /// <param name="title">The title of the form, also mentioned in the thread name.</param>
        /// <returns>
        ///     A task of the viewer, completed once the form exists on its own STA thread (which then runs its message loop
        ///     until the form closes).
        /// </returns>
        /// <exception cref="TimeoutException">Thrown (through the task) when the form did not come up within 10 seconds.</exception>
        public static async Task<IEnvViewer> Run(int width, int height, string title = null) {
            var taskResult = new TaskCompletionSource<IEnvViewer>(TaskCreationOptions.RunContinuationsAsynchronously);
            Start(width, height, title, viewer => taskResult.SetResult(viewer));

            var timeout = Task.Delay(10_000);
            if (await Task.WhenAny(taskResult.Task, timeout) == timeout)
                throw new TimeoutException("Starting viewer timed out.");

            Debug.Assert(taskResult.Task.Result != null, "At this point viewer shouldn't be null.");

            return taskResult.Task.Result;
        }

        /// <summary>
        ///     Creates the form on a new STA thread, hands it to <paramref name="created"/> from that thread, then runs the
        ///     form's message loop there until the form closes: what <see cref="Open"/> and <see cref="Run"/> share.
        /// </summary>
        /// <param name="width">The frames' width, the form's client width.</param>
        /// <param name="height">The frames' height, the form's client height.</param>
        /// <param name="title">The form's title, also in the thread's name.</param>
        /// <param name="created">Called on the form's thread once the form exists, before its message loop starts.</param>
        private static void Start(int width, int height, string title, Action<WinFormEnvViewer> created) {
            var thread = new Thread(() => {
                var v = new WinFormEnvViewer(width + 12, height + 12, title);
                // The picture shows a frame at its own size from the top-left corner (it does not scale), so the client
                // area takes the frame's size; the outer size above only seeds the form. A title bar and borders take more
                // than 12 pixels on Windows 10 and 11, which cropped about 27 rows off every frame's bottom before.
                v.ClientSize = new System.Drawing.Size(width, height);
                created(v);
                if (SystemInformation.UserInteractive) {
                    Application.Run(v);
                } else {
                    // Non-interactive / headless environment
                    v.CreateControl();
                    Application.Run(new ApplicationContext());
                }
            });
            thread.SetApartmentState(ApartmentState.STA);
            thread.IsBackground = true;
            thread.Start();
            thread.Name = $"Viewer{(string.IsNullOrEmpty(title) ? "" : $"-{title}")}";
        }

        /// <summary>A form of the given outer size; create it on the thread that will run it, as <see cref="Run"/> does.</summary>
        /// <param name="width">The form's width in pixels, borders included (<see cref="Run"/> adds 12 to the frame's).</param>
        /// <param name="height">The form's height in pixels, borders included.</param>
        /// <param name="title">The title, or <see langword="null"/> to keep the designer's.</param>
        public WinFormEnvViewer(int width, int height, string title = null) {
            InitializeComponent();
            Height = height;
            Width = width;
            if (title != null)
                Text = title;
            // the picture fills the form, so it is the picture that receives the mouse (DR-70); its coordinates are the
            // client area's. The wheel goes to the focused control, the form, through OnMouseWheel.
            PictureFrame.MouseDown += (_, e) => ReportButton(e, down: true);
            PictureFrame.MouseUp += (_, e) => ReportButton(e, down: false);
            PictureFrame.MouseMove += (_, e) => {
                if (_inputEnabled)
                    _input.Enqueue(ViewerInput.MouseMove(e.X, e.Y));
            };
        }

        /// <summary>
        ///     Shows a frame in <see cref="PictureFrame"/>, replacing the previous one; from another thread it waits for the
        ///     form's thread to show it. The frame stays the caller's: it is copied into an image first.
        /// </summary>
        /// <param name="frame">The frame, <c>(H, W, 3)</c>; another dtype is cast to uint8.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c>.</exception>
        public void Render(NDArray frame) {
            // Copied on the caller's thread, so the caller may reuse the frame however the form's thread is scheduled.
            Image<Rgb24> image = ViewerFrames.ToImage(frame);
            if (InvokeRequired) {
                Invoke(new Action(() => Show(image)));
                return;
            }

            Show(image);
        }

        /// <summary>
        ///     Shows a frame without blocking (DR-85): the frame is copied on the caller's thread, then handed to the form's
        ///     thread, and the task completes once it is shown. On the form's thread itself it completes at once.
        /// </summary>
        /// <param name="frame">The frame, <c>(H, W, 3)</c>; it stays the caller's, copied before this returns its task.</param>
        /// <param name="cancellationToken">Cancels the wait for the form's thread; the frame may still be shown.</param>
        /// <returns>The pending render.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c>.</exception>
        /// <exception cref="OperationCanceledException">Thrown, or through the task, when <paramref name="cancellationToken"/> is cancelled first.</exception>
        public ValueTask RenderAsync(NDArray frame, CancellationToken cancellationToken = default) {
            cancellationToken.ThrowIfCancellationRequested();
            Image<Rgb24> image = ViewerFrames.ToImage(frame);
            if (!InvokeRequired) {
                Show(image);
                return default;
            }

            var shown = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            BeginInvoke(new Action(() => {
                try {
                    Show(image);
                    shown.SetResult();
                } catch (Exception e) {
                    shown.SetException(e);
                }
            }));
            return new ValueTask(shown.Task.WaitAsync(cancellationToken));
        }

        /// <summary>Puts an image into <see cref="PictureFrame"/> as a bitmap, disposing the image and the frame it replaces; on the form's thread.</summary>
        /// <param name="image">The frame's image, which this disposes.</param>
        private void Show(Image<Rgb24> image) {
            using (image)
            using (var ms = new MemoryStream(_lastSize)) {
                image.SaveAsBmp(ms);
                _lastSize = (int) ms.Length;
                Clear();
                PictureFrame.Image = new Bitmap(ms);
            }
        }

        /// <summary>Raises the <see cref="E:System.Windows.Forms.Form.Shown" /> event.</summary>
        /// <param name="e">A <see cref="T:System.EventArgs" /> that contains the event data. </param>
        protected override void OnShown(EventArgs e) {
            base.OnShown(e);
            _shown = true;
            _ready.Set();
        }

        /// <summary>Removes and disposes the frame shown; call it on the form's thread.</summary>
        public void Clear() {
            if (PictureFrame.Image != null) {
                var img = PictureFrame.Image;
                PictureFrame.Image = null;
                img.Dispose();
            }
        }

        /// <summary>
        ///     Closes and disposes the form and ends its message loop's thread (the contract's <see cref="IEnvViewer.Close"/>);
        ///     from another thread it waits for the form's thread to do it. A close made here is never turned into a
        ///     <see cref="ViewerInputKind.Close"/> report.
        /// </summary>
        void IEnvViewer.Close() {
            if (InvokeRequired) {
                Invoke(new Action(CloseOnOwnThread));
                return;
            }

            CloseOnOwnThread();
        }

        /// <summary>Closes the form without blocking (DR-85); completes once the form's thread has closed it.</summary>
        /// <returns>The pending close.</returns>
        public ValueTask CloseAsync() {
            if (!InvokeRequired) {
                CloseOnOwnThread();
                return default;
            }

            var closed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            BeginInvoke(new Action(() => {
                try {
                    CloseOnOwnThread();
                    closed.SetResult();
                } catch (Exception e) {
                    closed.SetException(e);
                }
            }));
            return new ValueTask(closed.Task);
        }

        /// <summary>0.2's name for closing: closes the form as <see cref="IEnvViewer.Close"/> does, kept so 0.2 code compiles until 2.0 (DR-58, DR-85).</summary>
        [Obsolete("0.2's name: close through IEnvViewer.Close(). Removed in 2.0.")]
        public void CloseEnvironment() => ((IEnvViewer)this).Close();

        /// <summary>Closes and disposes the form and ends its thread's message loop; on the form's thread.</summary>
        private void CloseOnOwnThread() {
            // Marked first: WinForms reports Form.Close as a user's close too, which input mode would turn into a report.
            _closingByCode = true;
            Close();
            Dispose();
            Application.ExitThread();
        }

        /// <summary>
        ///     Starts reporting input: keys (arrows and Tab included, which a form otherwise keeps for focus), the close
        ///     button (which from now on only reports and no longer closes), the user's resizes and the mouse. Safe from any
        ///     thread; calling it again does nothing more.
        /// </summary>
        public void BeginInput() => _inputEnabled = true;

        /// <summary>Takes the input reported since the last call, oldest first. Safe from any thread.</summary>
        /// <returns>The inputs; empty when there are none.</returns>
        public IReadOnlyList<ViewerInput> TakeInput() {
            var taken = new List<ViewerInput>();
            while (_input.TryDequeue(out ViewerInput input))
                taken.Add(input);
            return taken;
        }

        /// <summary>With input on, every key is the game's: arrows and Tab reach <see cref="OnKeyDown"/> instead of moving the focus.</summary>
        /// <param name="keyData">The key with its modifiers.</param>
        /// <returns>Whether the key is an input key.</returns>
        protected override bool IsInputKey(Keys keyData) => _inputEnabled || base.IsInputKey(keyData);

        /// <summary>Reports a key press once per hold (no auto-repeat), when input is on and the key has an SDL keycode.</summary>
        /// <param name="e">The key event; marked handled when reported.</param>
        protected override void OnKeyDown(KeyEventArgs e) {
            base.OnKeyDown(e);
            if (!_inputEnabled || !SdlKeys.TryGetValue(e.KeyCode, out int code))
                return;
            // A key already held is the system repeating it: pygame reports no repeats unless asked to.
            if (_held.Add(e.KeyCode))
                _input.Enqueue(ViewerInput.KeyDown(code));
            e.Handled = true;
        }

        /// <summary>Reports a key release, when input is on and the key has an SDL keycode.</summary>
        /// <param name="e">The key event; marked handled when reported.</param>
        protected override void OnKeyUp(KeyEventArgs e) {
            base.OnKeyUp(e);
            if (!_inputEnabled || !SdlKeys.TryGetValue(e.KeyCode, out int code))
                return;
            _held.Remove(e.KeyCode);
            _input.Enqueue(ViewerInput.KeyUp(code));
            e.Handled = true;
        }

        /// <summary>
        ///     Reports a mouse button press or release at the pointer (GLFW's numbering of the buttons, DR-70), when input is
        ///     on; other buttons are not reported. WinForms captures the mouse to the picture while a button is held, so the
        ///     release and the moves of a drag arrive even outside it.
        /// </summary>
        /// <param name="e">The mouse event.</param>
        /// <param name="down">Whether the button went down.</param>
        private void ReportButton(MouseEventArgs e, bool down) {
            if (!_inputEnabled)
                return;
            int button = e.Button switch {
                MouseButtons.Left => ViewerInput.LeftButton,
                MouseButtons.Right => ViewerInput.RightButton,
                MouseButtons.Middle => ViewerInput.MiddleButton,
                _ => -1,
            };
            if (button < 0)
                return;
            _input.Enqueue(down ? ViewerInput.MouseDown(button, e.X, e.Y) : ViewerInput.MouseUp(button, e.X, e.Y));
        }

        /// <summary>
        ///     Reports a turn of the wheel in notches (Windows' 120 a notch), positive away from the user, at the pointer in
        ///     the client area, when input is on.
        /// </summary>
        /// <param name="e">The wheel event, passed on.</param>
        protected override void OnMouseWheel(MouseEventArgs e) {
            base.OnMouseWheel(e);
            if (_inputEnabled)
                _input.Enqueue(ViewerInput.Scroll(0, e.Delta / 120.0, e.X, e.Y));
        }

        /// <summary>
        ///     With input on, turns the user's close button into a <see cref="ViewerInputKind.Close"/> report and keeps the form
        ///     open, as a pygame window stays open until the program quits; closes by code and by Windows go ahead.
        /// </summary>
        /// <param name="e">The closing event; cancelled when reported.</param>
        protected override void OnFormClosing(FormClosingEventArgs e) {
            base.OnFormClosing(e);
            if (_inputEnabled && !_closingByCode && e.CloseReason == CloseReason.UserClosing) {
                e.Cancel = true;
                _input.Enqueue(ViewerInput.Close());
            }
        }

        /// <summary>
        ///     Reports every new client size once the form is shown (a drag reports each step of it, as SDL does; maximizing,
        ///     snapping and other programs' resizes too), when input is on and the size is not empty (a minimized form's).
        ///     The sizes the form takes while it is first laid out are left out.
        /// </summary>
        /// <param name="e">The event's arguments, passed on.</param>
        protected override void OnClientSizeChanged(EventArgs e) {
            base.OnClientSizeChanged(e);
            if (_inputEnabled && _shown && ClientSize.Width > 0 && ClientSize.Height > 0)
                _input.Enqueue(ViewerInput.Resized(ClientSize.Width, ClientSize.Height));
        }

        /// <summary>
        ///     The SDL keycode of each of Windows' virtual keys (DR-47): the character a key names for the keys that type one
        ///     (unshifted, as on a US layout for the punctuation keys), <see cref="KeyCode.ScancodeMask"/> codes for the rest;
        ///     Shift, Ctrl and Alt as their left keys.
        /// </summary>
        private static readonly Dictionary<Keys, int> SdlKeys = BuildKeyMap();

        /// <summary>Builds <see cref="SdlKeys"/>.</summary>
        /// <returns>The map.</returns>
        private static Dictionary<Keys, int> BuildKeyMap() {
            var map = new Dictionary<Keys, int> {
                [Keys.Back] = KeyCode.Backspace, [Keys.Tab] = KeyCode.Tab, [Keys.Return] = KeyCode.Return, [Keys.Escape] = KeyCode.Escape,
                [Keys.Space] = KeyCode.Space, [Keys.OemQuotes] = KeyCode.Quote, [Keys.Oemcomma] = KeyCode.Comma, [Keys.OemMinus] = KeyCode.Minus,
                [Keys.OemPeriod] = KeyCode.Period, [Keys.OemQuestion] = KeyCode.Slash, [Keys.OemSemicolon] = KeyCode.Semicolon,
                [Keys.Oemplus] = KeyCode.EqualsSign, [Keys.OemOpenBrackets] = KeyCode.LeftBracket, [Keys.OemPipe] = KeyCode.Backslash,
                [Keys.OemCloseBrackets] = KeyCode.RightBracket, [Keys.Oemtilde] = KeyCode.Backquote, [Keys.Delete] = KeyCode.Delete,
                [Keys.CapsLock] = KeyCode.CapsLock, [Keys.PrintScreen] = KeyCode.PrintScreen, [Keys.Scroll] = KeyCode.ScrollLock,
                [Keys.Pause] = KeyCode.Pause, [Keys.Insert] = KeyCode.Insert, [Keys.Home] = KeyCode.Home, [Keys.PageUp] = KeyCode.PageUp,
                [Keys.End] = KeyCode.End, [Keys.PageDown] = KeyCode.PageDown, [Keys.Right] = KeyCode.Right, [Keys.Left] = KeyCode.Left,
                [Keys.Down] = KeyCode.Down, [Keys.Up] = KeyCode.Up, [Keys.NumLock] = KeyCode.NumLock, [Keys.Divide] = KeyCode.KeypadDivide,
                [Keys.Multiply] = KeyCode.KeypadMultiply, [Keys.Subtract] = KeyCode.KeypadMinus, [Keys.Add] = KeyCode.KeypadPlus,
                [Keys.Decimal] = KeyCode.KeypadPeriod, [Keys.Apps] = KeyCode.Menu, [Keys.ControlKey] = KeyCode.LeftCtrl,
                [Keys.ShiftKey] = KeyCode.LeftShift, [Keys.Menu] = KeyCode.LeftAlt, [Keys.LWin] = KeyCode.LeftGui, [Keys.RWin] = KeyCode.RightGui,
            };
            // Letters, digits, keypad digits and function keys run in order in Keys (A is 65, D0 48, NumPad0 96, F1 112).
            for (int i = 0; i < 26; i++)
                map[Keys.A + i] = KeyCode.A + i;
            for (int i = 0; i < 10; i++) {
                map[Keys.D0 + i] = KeyCode.Digit0 + i;
                // SDL numbers the keypad 1 to 9, then 0.
                map[Keys.NumPad0 + i] = i == 0 ? KeyCode.Keypad0 : KeyCode.Keypad1 + (i - 1);
            }

            // SDL's F1 to F12 run on from F1, F13 to F15 from F13.
            for (int i = 0; i < 15; i++)
                map[Keys.F1 + i] = i < 12 ? KeyCode.F1 + i : KeyCode.F13 + (i - 12);
            return map;
        }

        /// <summary>
        ///     Releases the displayed frame and the ready signal as the window closes. Failures are swallowed
        ///     (<see cref="TryDispose"/>): closing must finish even when a half-released frame throws.
        /// </summary>
        /// <param name="e">The closing event's arguments, passed on to the base form first.</param>
        protected override void OnClosing(CancelEventArgs e) {
            base.OnClosing(e);
            TryDispose(PictureFrame.Image);
            TryDispose(_ready);
        }

        /// <summary>
        ///     Disposes <paramref name="disposable"/> when it is not <see langword="null"/>, swallowing whatever the disposal
        ///     throws: the window is closing, and a half-released frame or event must not keep it open. This was the
        ///     <c>TryDispose</c> extension of <c>Gym.Threading</c>, removed with that namespace (B-34).
        /// </summary>
        /// <param name="disposable">The object to release, or <see langword="null"/>.</param>
        private static void TryDispose(IDisposable disposable) {
            if (disposable == null)
                return;
            try {
                disposable.Dispose();
            } catch {
                // ignored: see the summary.
            }
        }
    }
}