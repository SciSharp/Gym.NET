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
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Image = SixLabors.ImageSharp.Image;

namespace Gym.Rendering.WinForm {
    /// <summary>
    ///     A form with PictureBox that accepts <see cref="IImageCanvas"/> and renders it on it. Start <see cref="Viewer"/> by calling <see cref="Run"/>
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Once <see cref="BeginInput"/> runs it is an <see cref="IInteractiveEnvViewer"/>: it reports keys as SDL keycodes
    ///     (<see cref="KeyCode"/>, by the key's position on a US keyboard), its close button (which then no longer closes it)
    ///     and the user's resizes, so <c>Gym.Utils.Play</c> can play an env in it. A held key reports one press, not the
    ///     system's auto-repeat, as pygame's default. WinForms reports Shift, Ctrl and Alt without their side, so they are
    ///     reported as the left ones. Before <see cref="BeginInput"/> nothing changes.
    ///     </para>
    /// </remarks>
    public partial class WinFormEnvViewer : Form, IInteractiveEnvViewer {
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

        /// <summary>Set by <see cref="CloseEnvironment"/>, whose <see cref="Form.Close"/> WinForms reports as a user's close too.</summary>
        private volatile bool _closingByCode;

        /// <summary>Set once the form is shown (<see cref="OnShown"/>), so the sizes of its first layout are not reported as resizes.</summary>
        private volatile bool _shown;

        /// <summary>
        ///     A delegate that creates a <see cref="WinFormEnvViewer"/> based on given parameters.
        /// </summary>
        public static IEnvironmentViewerFactoryDelegate Factory => Run;

        /// <summary>
        ///     Starts a <see cref="WinFormEnvViewer"/> in seperate thread.
        /// </summary>
        /// <param name="height">The height of the form</param>
        /// <param name="width">The width of the form</param>
        /// <param name="title">The title of the form, also mentioned in the thread name.</param>
        public static async Task<IEnvViewer> Run(int width, int height, string title = null) {
            var taskResult = new TaskCompletionSource<IEnvViewer>();
            var thread = new Thread(() => {
                var v = new WinFormEnvViewer(width + 12, height + 12, title);
                taskResult.SetResult(v);
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

            var timeout = Task.Delay(10_000);
            if (await Task.WhenAny(taskResult.Task, timeout) == timeout)
                throw new Exception("Starting viewer timed out.");

            Debug.Assert(taskResult.Task.Result != null, "At this point viewer shouldn't be null.");

            return taskResult.Task.Result;
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
        }

        /// <summary>
        ///     Renders this canvas onto <see cref="PictureFrame"/>.
        /// </summary>
        /// <param name="canvas">Canvas painted from <see cref="NGraphics"/></param>
        public void Render(Image canvas) {
            if (InvokeRequired) {
                Invoke(new Action(() => Render(canvas)));
                return;
            }

            using (var ms = new MemoryStream(_lastSize)) {
                canvas.SaveAsBmp(ms);
                _lastSize = (int) ms.Length;
                Clear();
                PictureFrame.Image = new Bitmap(ms);
            }
        }

        /// <summary>
        ///     Renders this canvas onto <see cref="PictureFrame"/>.
        /// </summary>
        /// <param name="canvas">Canvas painted from <see cref="NGraphics"/></param>
        public void RenderAsync(Image canvas) {
            if (InvokeRequired) {
                BeginInvoke(new Action(() => RenderAsync(canvas)));
                return;
            }

            using (var ms = new MemoryStream(_lastSize)) {
                canvas.SaveAsBmp(ms);
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
        ///     Closes and disposes the form and ends its message loop's thread; from another thread it waits for the form's
        ///     thread to do it. A close made here is never turned into a <see cref="ViewerInputKind.Close"/> report.
        /// </summary>
        public void CloseEnvironment() {
            if (InvokeRequired) {
                Invoke(new Action(CloseEnvironment));
                return;
            }

            _closingByCode = true;
            Close();
            Dispose();
            Application.ExitThread();
        }

        /// <summary>
        ///     Starts reporting input: keys (arrows and Tab included, which a form otherwise keeps for focus), the close
        ///     button (which from now on only reports and no longer closes) and the user's resizes. Safe from any thread;
        ///     calling it again does nothing more.
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
        ///     The SDL keycode of each key on a US keyboard: characters for the keys that type one (unshifted),
        ///     <see cref="KeyCode.ScancodeMask"/> codes for the rest; Shift, Ctrl and Alt as their left keys.
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