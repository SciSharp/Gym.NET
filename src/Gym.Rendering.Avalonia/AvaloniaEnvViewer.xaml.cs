using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Markup.Xaml;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Gym.Environments;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Image = SixLabors.ImageSharp.Image;
using AVImage = Avalonia.Controls.Image;
using Size = Avalonia.Size;
using Point = Avalonia.Point;

namespace Gym.Rendering.Avalonia {
    /// <summary>
    ///     An Avalonia window that shows an env's frames, and once <see cref="BeginInput"/> runs, reports the keyboard, its
    ///     close button and the user's resizes as <see cref="IInteractiveEnvViewer"/> input, so <c>Gym.Utils.Play</c> can
    ///     play an env in it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Keys are reported by the keyboard layout's key (<see cref="KeyEventArgs.Key"/>), as SDL keycodes
    ///     (<see cref="KeyCode"/>), the way pygame reports them (DR-47): the key labeled A gives <c>K_a</c> on any layout, so
    ///     a <c>keys_to_action</c> written with <c>ord("a")</c> fires on the key bearing A, as in Gymnasium. Avalonia's key is
    ///     the platform's virtual key, which follows the layout for letters; where a layout types another character
    ///     unshifted than its virtual key names (AZERTY's digit row types <c>&amp;</c>, <c>é</c>, ...), SDL reports that
    ///     character and this viewer the virtual key (<c>K_1</c>), and the keypad's Enter is the main Enter (Avalonia's key
    ///     does not tell them apart). A held key reports one press, not the system's auto-repeat, as pygame's default (no
    ///     <c>key.set_repeat</c>). Keys without an SDL keycode in <see cref="KeyCode"/> are not reported. Resizes are
    ///     reported in device-independent pixels, the unit the window was opened in, and so is the mouse (DR-70: its
    ///     buttons, moves and wheel, for a MuJoCo env's camera controls).
    ///     </para>
    ///     <para>
    ///     Before <see cref="BeginInput"/> nothing changes: the close button closes the window, as it always has for the
    ///     human-mode windows of envs.
    ///     </para>
    ///     <para>
    ///     It renders both ways the viewer contract offers (DR-85): <see cref="Render"/> waits for the UI thread to take the
    ///     frame, <see cref="RenderAsync"/> awaits it. Closing through the contract (<see cref="IEnvViewer.Close"/>) hides the
    ///     window; footgun: <see cref="Window.Close()"/>, which a variable typed <c>AvaloniaEnvViewer</c> calls as
    ///     <c>viewer.Close()</c>, is Avalonia's own and destroys it instead.
    ///     </para>
    /// </remarks>
    public class AvaloniaEnvViewer : Window, IInteractiveEnvViewer, IAsyncEnvViewer {

        /// <summary>Set once the window has opened (<see cref="OnOpened"/>).</summary>
        private readonly ManualResetEventSlim ReadyResetEvent = new ManualResetEventSlim();

        /// <summary>Signals that the UI thread has taken a frame <see cref="Render"/> handed to it, so the caller may go on.</summary>
        private readonly ManualResetEventSlim RenderResetEvent = new ManualResetEventSlim();

        /// <summary>The input reported since the last <see cref="TakeInput"/>, filled on the UI thread.</summary>
        private readonly ConcurrentQueue<ViewerInput> _input = new ConcurrentQueue<ViewerInput>();

        /// <summary>The keys held now, so the system's auto-repeat reports no second press (UI thread only).</summary>
        private readonly HashSet<Key> _held = new HashSet<Key>();

        /// <summary>Whether input is reported (<see cref="BeginInput"/> ran); read on the UI thread, set on any.</summary>
        private volatile bool _inputEnabled;

        /// <summary>
        ///     A window with the XAML's defaults (the designer's size, the title <c>Gym.Rendering.Avalonia</c>); create it on
        ///     Avalonia's UI thread. Envs get theirs through <see cref="Factory"/> instead.
        /// </summary>
        public AvaloniaEnvViewer() {
            InitializeComponent();
        }

        /// <summary>A window of a frame's size; create it on Avalonia's UI thread, as <see cref="Factory"/> does.</summary>
        /// <param name="width">The width in device-independent pixels, the frame's width.</param>
        /// <param name="height">The height in device-independent pixels, the frame's height.</param>
        /// <param name="title">The title, or <see langword="null"/> for none.</param>
        public AvaloniaEnvViewer(int width, int height, string title) : this() {
            Width = width;
            Height = height;
            Title = title;
        }

        /// <summary>Loads the XAML that declares the window.</summary>
        private void InitializeComponent() {
            AvaloniaXamlLoader.Load(this);
        }

        /// <summary>
        ///     Closes the window as <see cref="IEnvViewer.Close"/> does (it is hidden, not destroyed: the Avalonia app is
        ///     shared by every viewer of the process). From another thread the close is posted to the UI thread and this
        ///     returns at once.
        /// </summary>
        public void Dispose() {
            //invoke if required
            if (!Dispatcher.UIThread.CheckAccess()) {
                Dispatcher.UIThread.InvokeAsync(Dispose, DispatcherPriority.MaxValue);
                return;
            }

            Hide();
        }

        /// <summary>
        ///     The factory envs open this window with (DR-20, DR-85): <see cref="StaticAvaloniaApp.Open"/>, which starts the
        ///     shared Avalonia application on its own UI thread the first time and returns once the window is shown.
        /// </summary>
        public static EnvViewerFactory Factory => StaticAvaloniaApp.Open;

        /// <summary>
        ///     Shows a frame, replacing the previous one; the frame stays the caller's (it is copied through a BMP stream
        ///     first). From another thread it waits up to 4 seconds for the UI thread to take the frame.
        /// </summary>
        /// <param name="frame">The frame, <c>(H, W, 3)</c>; another dtype is cast to uint8.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c>.</exception>
        /// <exception cref="TimeoutException">Thrown when the UI thread did not take the frame within 4 seconds (<c>Rendering timed out.</c>).</exception>
        public void Render(NDArray frame) {
            Bitmap bitmap = ToBitmap(frame);
            if (Dispatcher.UIThread.CheckAccess()) {
                Content = new AVImage {
                    Source = bitmap
                };
            } else {
                RenderResetEvent.Reset();
                Dispatcher.UIThread.InvokeAsync(() => {
                    Content = new AVImage {
                        Source = bitmap
                    };
                    RenderResetEvent.Set();
                }, DispatcherPriority.MaxValue);
                // The synchronous path (DR-85) waits on an event, not a task: the UI thread never waits on this one, so the
                // two cannot deadlock, and a stalled UI is reported instead of hanging the env.
                if (!RenderResetEvent.Wait(4_000))
                    throw new TimeoutException("Rendering timed out.");
            }
        }

        /// <summary>
        ///     Shows a frame without blocking (DR-85): the bitmap is made on the caller's thread, then handed to the UI thread,
        ///     and the task completes once the UI has taken it. On the UI thread itself it completes at once.
        /// </summary>
        /// <param name="frame">The frame, <c>(H, W, 3)</c>; it stays the caller's, copied before this returns its task.</param>
        /// <param name="cancellationToken">Cancels the wait for the UI thread; the frame may still be shown.</param>
        /// <returns>The pending render.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c>.</exception>
        /// <exception cref="OperationCanceledException">Thrown, or through the task, when <paramref name="cancellationToken"/> is cancelled first.</exception>
        public ValueTask RenderAsync(NDArray frame, CancellationToken cancellationToken = default) {
            cancellationToken.ThrowIfCancellationRequested();
            Bitmap bitmap = ToBitmap(frame);
            if (Dispatcher.UIThread.CheckAccess()) {
                Content = new AVImage {
                    Source = bitmap
                };
                return default;
            }

            Task shown = Dispatcher.UIThread.InvokeAsync(() => {
                Content = new AVImage {
                    Source = bitmap
                };
            }, DispatcherPriority.MaxValue).GetTask();
            return new ValueTask(shown.WaitAsync(cancellationToken));
        }

        /// <summary>A frame as an Avalonia bitmap, through ImageSharp's BMP encoder (the viewers' own conversion, shared by both packages).</summary>
        /// <param name="frame">The frame, <c>(H, W, 3)</c>.</param>
        /// <returns>A new bitmap the window takes over.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c>.</exception>
        private static Bitmap ToBitmap(NDArray frame) {
            using Image<Rgb24> image = ViewerFrames.ToImage(frame);
            // A 24-bit BMP is its 54-byte header and rows padded to 4 bytes: room for that, so the stream does not regrow.
            using var stream = new MemoryStream(image.Height * (image.Width * 3 + 3) + 54);
            image.SaveAsBmp(stream);
            stream.Seek(0, SeekOrigin.Begin);
            return new Bitmap(stream);
        }

        /// <summary>Marks the window as open, after Avalonia's own handling.</summary>
        /// <param name="e">The event's arguments, passed on.</param>
        protected override void OnOpened(EventArgs e) {
            base.OnOpened(e);
            ReadyResetEvent.Set();
        }

        /// <summary>
        ///     Hides the window (the contract's <see cref="IEnvViewer.Close"/>): the Avalonia app is shared by every viewer of
        ///     the process, so the window is not destroyed. From another thread the hide is posted to the UI thread and this
        ///     returns at once.
        /// </summary>
        void IEnvViewer.Close() {
            //invoke if required
            if (!Dispatcher.UIThread.CheckAccess()) {
                Dispatcher.UIThread.InvokeAsync(Hide, DispatcherPriority.MaxValue);
                return;
            }

            Hide();
        }

        /// <summary>Hides the window without blocking (DR-85); completes once the UI thread has hidden it.</summary>
        /// <returns>The pending close.</returns>
        public ValueTask CloseAsync() {
            if (Dispatcher.UIThread.CheckAccess()) {
                Hide();
                return default;
            }

            return new ValueTask(Dispatcher.UIThread.InvokeAsync(Hide, DispatcherPriority.MaxValue).GetTask());
        }

        /// <summary>0.2's name for closing: hides the window as <see cref="IEnvViewer.Close"/> does, kept so 0.2 code compiles until 2.0 (DR-58, DR-85).</summary>
        [Obsolete("0.2's name: close through IEnvViewer.Close() (hides the window). Removed in 2.0.")]
        public void CloseEnvironment() => ((IEnvViewer)this).Close();

        /// <summary>
        ///     Starts reporting input: keys, the close button (which from now on only reports and no longer closes), the
        ///     user's resizes and the mouse. Safe from any thread; calling it again does nothing more.
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

        /// <summary>Reports a key press once per hold (no auto-repeat), when input is on and the key has an SDL keycode.</summary>
        /// <param name="e">The key event; marked handled when reported.</param>
        protected override void OnKeyDown(KeyEventArgs e) {
            base.OnKeyDown(e);
            if (!_inputEnabled || !SdlKeys.TryGetValue(e.Key, out int code))
                return;
            // A key already held is the system repeating it: pygame reports no repeats unless asked to.
            if (_held.Add(e.Key))
                _input.Enqueue(ViewerInput.KeyDown(code));
            e.Handled = true;
        }

        /// <summary>Reports a key release, when input is on and the key has an SDL keycode.</summary>
        /// <param name="e">The key event; marked handled when reported.</param>
        protected override void OnKeyUp(KeyEventArgs e) {
            base.OnKeyUp(e);
            if (!_inputEnabled || !SdlKeys.TryGetValue(e.Key, out int code))
                return;
            _held.Remove(e.Key);
            _input.Enqueue(ViewerInput.KeyUp(code));
            e.Handled = true;
        }

        /// <summary>
        ///     With input on, turns the user's close button into a <see cref="ViewerInputKind.Close"/> report and keeps the
        ///     window open, as a pygame window stays open until the program quits; any other close goes ahead.
        /// </summary>
        /// <param name="e">The closing event; cancelled when reported.</param>
        protected override void OnClosing(WindowClosingEventArgs e) {
            base.OnClosing(e);
            if (_inputEnabled && !e.IsProgrammatic && e.CloseReason == WindowCloseReason.WindowClosing) {
                e.Cancel = true;
                _input.Enqueue(ViewerInput.Close());
            }
        }

        /// <summary>
        ///     Reports a new client size, in device-independent pixels, whatever made it (a drag, maximizing, snapping, another
        ///     program), as SDL reports every size change of a pygame window; only a resize by the window's own code
        ///     (<see cref="WindowResizeReason.Application"/>) is not the user's and is left out.
        /// </summary>
        /// <param name="e">The resize event.</param>
        protected override void OnResized(WindowResizedEventArgs e) {
            base.OnResized(e);
            if (!_inputEnabled || e.Reason == WindowResizeReason.Application)
                return;
            int width = (int)Math.Round(e.ClientSize.Width), height = (int)Math.Round(e.ClientSize.Height);
            if (width > 0 && height > 0)
                _input.Enqueue(ViewerInput.Resized(width, height));
        }

        /// <summary>
        ///     Reports a mouse button press at the pointer, in device-independent pixels from the client area's top-left (DR-70,
        ///     GLFW's numbering of the buttons), when input is on; buttons beyond the three are not reported.
        /// </summary>
        /// <param name="e">The pointer event; marked handled when reported.</param>
        protected override void OnPointerPressed(PointerPressedEventArgs e) {
            base.OnPointerPressed(e);
            if (!_inputEnabled || !MouseButton(e.GetCurrentPoint(this).Properties.PointerUpdateKind, out int button))
                return;
            Point at = e.GetPosition(this);
            _input.Enqueue(ViewerInput.MouseDown(button, at.X, at.Y));
            e.Handled = true;
        }

        /// <summary>Reports a mouse button release at the pointer, when input is on.</summary>
        /// <param name="e">The pointer event; marked handled when reported.</param>
        protected override void OnPointerReleased(PointerReleasedEventArgs e) {
            base.OnPointerReleased(e);
            if (!_inputEnabled || !MouseButton(e.GetCurrentPoint(this).Properties.PointerUpdateKind, out int button))
                return;
            Point at = e.GetPosition(this);
            _input.Enqueue(ViewerInput.MouseUp(button, at.X, at.Y));
            e.Handled = true;
        }

        /// <summary>Reports the pointer's new position, when input is on (Avalonia captures it to the window while a button is held).</summary>
        /// <param name="e">The pointer event.</param>
        protected override void OnPointerMoved(PointerEventArgs e) {
            base.OnPointerMoved(e);
            if (!_inputEnabled)
                return;
            Point at = e.GetPosition(this);
            _input.Enqueue(ViewerInput.MouseMove(at.X, at.Y));
        }

        /// <summary>Reports a turn of the wheel in notches, positive away from the user, when input is on.</summary>
        /// <param name="e">The wheel event; marked handled when reported.</param>
        protected override void OnPointerWheelChanged(PointerWheelEventArgs e) {
            base.OnPointerWheelChanged(e);
            if (!_inputEnabled)
                return;
            Point at = e.GetPosition(this);
            _input.Enqueue(ViewerInput.Scroll(e.Delta.X, e.Delta.Y, at.X, at.Y));
            e.Handled = true;
        }

        /// <summary>GLFW's number of the button an Avalonia pointer update pressed or released.</summary>
        /// <param name="kind">The update.</param>
        /// <param name="button">The button, when it is the left, right or middle one.</param>
        /// <returns>Whether the update pressed or released one of the three.</returns>
        private static bool MouseButton(PointerUpdateKind kind, out int button) {
            button = kind switch {
                PointerUpdateKind.LeftButtonPressed or PointerUpdateKind.LeftButtonReleased => ViewerInput.LeftButton,
                PointerUpdateKind.RightButtonPressed or PointerUpdateKind.RightButtonReleased => ViewerInput.RightButton,
                PointerUpdateKind.MiddleButtonPressed or PointerUpdateKind.MiddleButtonReleased => ViewerInput.MiddleButton,
                _ => -1,
            };
            return button >= 0;
        }

        /// <summary>
        ///     The SDL keycode of each of Avalonia's keys (DR-47): the character a key names for the keys that type one
        ///     (unshifted, as on a US layout for the punctuation keys), <see cref="KeyCode.ScancodeMask"/> codes for the rest.
        ///     Filled through the indexer, so Avalonia's aliases of one value (<c>Enter</c> and <c>Return</c>) cannot clash.
        /// </summary>
        private static readonly Dictionary<Key, int> SdlKeys = BuildKeyMap();

        /// <summary>Builds <see cref="SdlKeys"/>.</summary>
        /// <returns>The map.</returns>
        private static Dictionary<Key, int> BuildKeyMap() {
            var map = new Dictionary<Key, int> {
                [Key.Back] = KeyCode.Backspace, [Key.Tab] = KeyCode.Tab, [Key.Enter] = KeyCode.Return,
                [Key.Escape] = KeyCode.Escape, [Key.Space] = KeyCode.Space, [Key.OemQuotes] = KeyCode.Quote,
                [Key.OemComma] = KeyCode.Comma, [Key.OemMinus] = KeyCode.Minus, [Key.OemPeriod] = KeyCode.Period,
                [Key.OemQuestion] = KeyCode.Slash, [Key.OemSemicolon] = KeyCode.Semicolon, [Key.OemPlus] = KeyCode.EqualsSign,
                [Key.OemOpenBrackets] = KeyCode.LeftBracket, [Key.OemPipe] = KeyCode.Backslash,
                [Key.OemCloseBrackets] = KeyCode.RightBracket, [Key.OemTilde] = KeyCode.Backquote, [Key.Delete] = KeyCode.Delete,
                [Key.CapsLock] = KeyCode.CapsLock, [Key.PrintScreen] = KeyCode.PrintScreen, [Key.Scroll] = KeyCode.ScrollLock,
                [Key.Pause] = KeyCode.Pause, [Key.Insert] = KeyCode.Insert, [Key.Home] = KeyCode.Home,
                [Key.PageUp] = KeyCode.PageUp, [Key.End] = KeyCode.End, [Key.PageDown] = KeyCode.PageDown,
                [Key.Right] = KeyCode.Right, [Key.Left] = KeyCode.Left, [Key.Down] = KeyCode.Down,
                [Key.Up] = KeyCode.Up, [Key.NumLock] = KeyCode.NumLock, [Key.Divide] = KeyCode.KeypadDivide,
                [Key.Multiply] = KeyCode.KeypadMultiply, [Key.Subtract] = KeyCode.KeypadMinus,
                [Key.Add] = KeyCode.KeypadPlus, [Key.Decimal] = KeyCode.KeypadPeriod,
                [Key.Apps] = KeyCode.Menu, [Key.LeftCtrl] = KeyCode.LeftCtrl, [Key.LeftShift] = KeyCode.LeftShift,
                [Key.LeftAlt] = KeyCode.LeftAlt, [Key.LWin] = KeyCode.LeftGui, [Key.RightCtrl] = KeyCode.RightCtrl,
                [Key.RightShift] = KeyCode.RightShift, [Key.RightAlt] = KeyCode.RightAlt, [Key.RWin] = KeyCode.RightGui,
            };
            // The letters, digits, keypad digits and function keys by name, so nothing depends on the enumeration's values.
            for (int i = 0; i < 26; i++)
                map[Enum.Parse<Key>(((char)('A' + i)).ToString())] = KeyCode.A + i;
            for (int i = 0; i < 10; i++) {
                map[Enum.Parse<Key>($"D{i}")] = KeyCode.Digit0 + i;
                // SDL numbers the keypad 1 to 9, then 0.
                map[Enum.Parse<Key>($"NumPad{i}")] = i == 0 ? KeyCode.Keypad0 : KeyCode.Keypad1 + (i - 1);
            }

            // SDL's F1 to F12 run on from F1, F13 to F15 from F13.
            for (int i = 1; i <= 15; i++)
                map[Enum.Parse<Key>($"F{i}")] = i <= 12 ? KeyCode.F1 + (i - 1) : KeyCode.F13 + (i - 13);
            return map;
        }
    }
}
