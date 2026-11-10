using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Markup.Xaml;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Gym.Environments;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Image = SixLabors.ImageSharp.Image;
using AVImage = Avalonia.Controls.Image;
using Size = Avalonia.Size;

namespace Gym.Rendering.Avalonia {
    /// <summary>
    ///     An Avalonia window that shows an env's frames, and once <see cref="BeginInput"/> runs, reports the keyboard, its
    ///     close button and the user's resizes as <see cref="IInteractiveEnvViewer"/> input, so <c>Gym.Utils.Play</c> can
    ///     play an env in it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Keys are reported by their physical position on a US keyboard (<see cref="KeyEventArgs.PhysicalKey"/>), as SDL
    ///     keycodes (<see cref="KeyCode"/>), so a game's keys stay where they are whatever the layout. A held key reports one
    ///     press, not the system's auto-repeat, as pygame's default (no <c>key.set_repeat</c>). Keys without an SDL keycode
    ///     in <see cref="KeyCode"/> are not reported. Resizes are reported in device-independent pixels, the unit the window
    ///     was opened in.
    ///     </para>
    ///     <para>
    ///     Before <see cref="BeginInput"/> nothing changes: the close button closes the window, as it always has for the
    ///     human-mode windows of envs.
    ///     </para>
    /// </remarks>
    public class AvaloniaEnvViewer : Window, IInteractiveEnvViewer {

        /// <summary>Set once the window has opened (<see cref="OnOpened"/>).</summary>
        private readonly ManualResetEventSlim ReadyResetEvent = new ManualResetEventSlim();

        /// <summary>Signals that the UI thread has taken a frame <see cref="Render"/> handed to it, so the caller may go on.</summary>
        private readonly ManualResetEventSlim RenderResetEvent = new ManualResetEventSlim();

        /// <summary>The input reported since the last <see cref="TakeInput"/>, filled on the UI thread.</summary>
        private readonly ConcurrentQueue<ViewerInput> _input = new ConcurrentQueue<ViewerInput>();

        /// <summary>The physical keys held now, so the system's auto-repeat reports no second press (UI thread only).</summary>
        private readonly HashSet<PhysicalKey> _held = new HashSet<PhysicalKey>();

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
        ///     Closes the window as <see cref="CloseEnvironment"/> does (it is hidden, not destroyed: the Avalonia app is
        ///     shared by every viewer of the process). From another thread the close is posted to the UI thread and this
        ///     returns at once.
        /// </summary>
        public void Dispose() {
            //invoke if required
            if (!Dispatcher.UIThread.CheckAccess()) {
                Dispatcher.UIThread.InvokeAsync(Dispose, DispatcherPriority.MaxValue);
                return;
            }

            CloseEnvironment();
        }

        /// <summary>
        ///     A delegate that creates a <see cref="AvaloniaEnvViewer"/> based on given parameters.
        /// </summary>
        public static IEnvironmentViewerFactoryDelegate Factory => StaticAvaloniaApp.Run;

        /// <summary>
        ///     Shows a frame, replacing the previous one; the image stays the caller's (it is copied through a BMP stream
        ///     first). From another thread it waits up to 4 seconds for the UI thread to take the frame.
        /// </summary>
        /// <param name="img">The frame.</param>
        /// <exception cref="Exception">Thrown when the UI thread did not take the frame within 4 seconds (<c>Rendering timed out.</c>).</exception>
        public void Render(Image img) {
            using (var ms = new MemoryStream(img.Height * img.Width * img.PixelType.BitsPerPixel / 8)) {
                img.SaveAsBmp(ms);
                ms.Seek(0, SeekOrigin.Begin);
                var bitmap = new Bitmap(ms);
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
                    if (!RenderResetEvent.Wait(4_000))
                        throw new Exception("Rendering timed out.");
                }
            }
        }

        /// <summary>Marks the window as open, after Avalonia's own handling.</summary>
        /// <param name="e">The event's arguments, passed on.</param>
        protected override void OnOpened(EventArgs e) {
            base.OnOpened(e);
            ReadyResetEvent.Set();
        }

        /// <summary>
        ///     Hides the window: the Avalonia app is shared by every viewer of the process, so the window is not destroyed.
        ///     From another thread the hide is posted to the UI thread and this returns at once.
        /// </summary>
        public void CloseEnvironment() {
            //invoke if required
            if (!Dispatcher.UIThread.CheckAccess()) {
                Dispatcher.UIThread.InvokeAsync(CloseEnvironment, DispatcherPriority.MaxValue);
                return;
            }

            Hide();
            //Dispose();
        }

        /// <summary>
        ///     Starts reporting input: keys, the close button (which from now on only reports and no longer closes) and the
        ///     user's resizes. Safe from any thread; calling it again does nothing more.
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
            if (!_inputEnabled || !SdlKeys.TryGetValue(e.PhysicalKey, out int code))
                return;
            // A key already held is the system repeating it: pygame reports no repeats unless asked to.
            if (_held.Add(e.PhysicalKey))
                _input.Enqueue(ViewerInput.KeyDown(code));
            e.Handled = true;
        }

        /// <summary>Reports a key release, when input is on and the key has an SDL keycode.</summary>
        /// <param name="e">The key event; marked handled when reported.</param>
        protected override void OnKeyUp(KeyEventArgs e) {
            base.OnKeyUp(e);
            if (!_inputEnabled || !SdlKeys.TryGetValue(e.PhysicalKey, out int code))
                return;
            _held.Remove(e.PhysicalKey);
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
        ///     The SDL keycode of each physical key on a US keyboard: characters for the keys that type one (unshifted),
        ///     <see cref="KeyCode.ScancodeMask"/> codes for the rest.
        /// </summary>
        private static readonly Dictionary<PhysicalKey, int> SdlKeys = BuildKeyMap();

        /// <summary>Builds <see cref="SdlKeys"/>.</summary>
        /// <returns>The map.</returns>
        private static Dictionary<PhysicalKey, int> BuildKeyMap() {
            var map = new Dictionary<PhysicalKey, int> {
                [PhysicalKey.Backspace] = KeyCode.Backspace, [PhysicalKey.Tab] = KeyCode.Tab, [PhysicalKey.Enter] = KeyCode.Return,
                [PhysicalKey.Escape] = KeyCode.Escape, [PhysicalKey.Space] = KeyCode.Space, [PhysicalKey.Quote] = KeyCode.Quote,
                [PhysicalKey.Comma] = KeyCode.Comma, [PhysicalKey.Minus] = KeyCode.Minus, [PhysicalKey.Period] = KeyCode.Period,
                [PhysicalKey.Slash] = KeyCode.Slash, [PhysicalKey.Semicolon] = KeyCode.Semicolon, [PhysicalKey.Equal] = KeyCode.EqualsSign,
                [PhysicalKey.BracketLeft] = KeyCode.LeftBracket, [PhysicalKey.Backslash] = KeyCode.Backslash,
                [PhysicalKey.BracketRight] = KeyCode.RightBracket, [PhysicalKey.Backquote] = KeyCode.Backquote, [PhysicalKey.Delete] = KeyCode.Delete,
                [PhysicalKey.CapsLock] = KeyCode.CapsLock, [PhysicalKey.PrintScreen] = KeyCode.PrintScreen, [PhysicalKey.ScrollLock] = KeyCode.ScrollLock,
                [PhysicalKey.Pause] = KeyCode.Pause, [PhysicalKey.Insert] = KeyCode.Insert, [PhysicalKey.Home] = KeyCode.Home,
                [PhysicalKey.PageUp] = KeyCode.PageUp, [PhysicalKey.End] = KeyCode.End, [PhysicalKey.PageDown] = KeyCode.PageDown,
                [PhysicalKey.ArrowRight] = KeyCode.Right, [PhysicalKey.ArrowLeft] = KeyCode.Left, [PhysicalKey.ArrowDown] = KeyCode.Down,
                [PhysicalKey.ArrowUp] = KeyCode.Up, [PhysicalKey.NumLock] = KeyCode.NumLock, [PhysicalKey.NumPadDivide] = KeyCode.KeypadDivide,
                [PhysicalKey.NumPadMultiply] = KeyCode.KeypadMultiply, [PhysicalKey.NumPadSubtract] = KeyCode.KeypadMinus,
                [PhysicalKey.NumPadAdd] = KeyCode.KeypadPlus, [PhysicalKey.NumPadEnter] = KeyCode.KeypadEnter,
                [PhysicalKey.NumPadDecimal] = KeyCode.KeypadPeriod, [PhysicalKey.NumPadEqual] = KeyCode.KeypadEquals,
                [PhysicalKey.ContextMenu] = KeyCode.Menu, [PhysicalKey.ControlLeft] = KeyCode.LeftCtrl, [PhysicalKey.ShiftLeft] = KeyCode.LeftShift,
                [PhysicalKey.AltLeft] = KeyCode.LeftAlt, [PhysicalKey.MetaLeft] = KeyCode.LeftGui, [PhysicalKey.ControlRight] = KeyCode.RightCtrl,
                [PhysicalKey.ShiftRight] = KeyCode.RightShift, [PhysicalKey.AltRight] = KeyCode.RightAlt, [PhysicalKey.MetaRight] = KeyCode.RightGui,
            };
            // The letters, digits, keypad digits and function keys by name, so nothing depends on the enumeration's values.
            for (int i = 0; i < 26; i++)
                map[Enum.Parse<PhysicalKey>(((char)('A' + i)).ToString())] = KeyCode.A + i;
            for (int i = 0; i < 10; i++) {
                map[Enum.Parse<PhysicalKey>($"Digit{i}")] = KeyCode.Digit0 + i;
                // SDL numbers the keypad 1 to 9, then 0.
                map[Enum.Parse<PhysicalKey>($"NumPad{i}")] = i == 0 ? KeyCode.Keypad0 : KeyCode.Keypad1 + (i - 1);
            }

            // SDL's F1 to F12 run on from F1, F13 to F15 from F13.
            for (int i = 1; i <= 15; i++)
                map[Enum.Parse<PhysicalKey>($"F{i}")] = i <= 12 ? KeyCode.F1 + (i - 1) : KeyCode.F13 + (i - 13);
            return map;
        }
    }
}
