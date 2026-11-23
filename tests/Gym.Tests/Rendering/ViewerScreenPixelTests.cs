#nullable enable
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using Gym.Environments;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using Gym.Rendering.WinForm;
using Gym.Tests.Gymnasium.Envs;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Tests.Rendering {
    /// <summary>
    ///     Every env in every window, byte for byte on screen (DR-99): each registered id runs in <c>human</c> mode in the
    ///     Avalonia and the WinForms viewer, and after the seeded reset and each sampled step the window's client area is
    ///     captured from the screen, as the display shows it, and compared with the frame the env handed the viewer, every
    ///     byte. The frames themselves equal Gymnasium's (the parity suite's <c>human</c> cells); these cells close the last
    ///     link, from the frame to the screen, at whatever display scaling the desktop runs.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The cells open windows and read the screen, so their names end in <c>AvaloniaEnv</c> and <c>WinFormEnv</c> and
    ///     the headless gate leaves them out; they run on a desktop (a disposable one, never a user's screen), and each
    ///     window is raised above the others first, since the screen shows whatever is on top. A capture is retried until
    ///     the window shows the frame or 5 seconds pass, as both toolkits draw a new frame asynchronously; on failure the
    ///     captured and expected images and their difference are written to the test's results.
    ///     </para>
    ///     <para>
    ///     Footgun: the capture reads the screen in physical pixels (the capturing thread is made per-monitor DPI aware
    ///     for the read), so a window that Windows stretches on a scaled display fails here, as DR-99 intends.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ViewerScreenPixelTests {
        /// <summary>The test's context, for the failure images.</summary>
        public TestContext TestContext { get; set; } = null!;

        /// <summary>The ids of every registered env Gym.NET can make whose metadata lists <c>human</c>.</summary>
        public static IEnumerable<object[]> HumanEnvIds => EnvTestingUtils.AllTestingEnvSpecs()
            .Where(spec => {
                using IEnv env = spec.Make();
                return env.Metadata.RenderModes?.Contains("human") == true;
            })
            .Select(spec => new object[] { spec.Id });

        /// <summary>The steps sampled after the reset, each a frame compared on screen.</summary>
        private const int Steps = 4;

        /// <summary>An env's frames in the WinForms window are its very bytes on screen.</summary>
        /// <param name="id">The env's id.</param>
        [TestMethod]
        [DynamicData(nameof(HumanEnvIds))]
        public void EveryEnv_ShowsItsFramesBytesOnScreen_WinFormEnv(string id) =>
            Run(id, WinFormEnvViewer.Factory, viewer => {
                var form = (WinFormEnvViewer)viewer;
                return (IntPtr)form.Invoke(new Func<IntPtr>(() => form.Handle));
            });

        /// <summary>An env's frames in the Avalonia window are its very bytes on screen.</summary>
        /// <param name="id">The env's id.</param>
        [TestMethod]
        [DynamicData(nameof(HumanEnvIds))]
        public void EveryEnv_ShowsItsFramesBytesOnScreen_AvaloniaEnv(string id) =>
            Run(id, AvaloniaEnvViewer.Factory, viewer => Avalonia.Threading.Dispatcher.UIThread.Invoke(
                    () => ((AvaloniaEnvViewer)viewer).TryGetPlatformHandle()?.Handle ?? IntPtr.Zero));

        /// <summary>
        ///     Runs one env in <c>human</c> mode through a recording wrapper around a viewer factory, and after the reset and
        ///     each step compares the window's client area on screen with the frame the env rendered last.
        /// </summary>
        /// <param name="id">The env's id.</param>
        /// <param name="factory">The viewer factory under test.</param>
        /// <param name="handleOf">Reads the window handle of a viewer the factory made.</param>
        /// <exception cref="AssertFailedException">A frame never showed byte for byte on screen within 5 seconds, or the env rendered nothing.</exception>
        private void Run(string id, EnvViewerFactory factory, Func<IEnvViewer, IntPtr> handleOf) {
            RecordingViewer? recorder = null;
            EnvViewerFactory recording = (width, height, title) => recorder = new RecordingViewer(factory(width, height, title));
            using IEnv env = Gym.Gymnasium.Make(id, RenderMode.Human, recording);
            try {
                env.ActionSpace.Seed(0);
                env.Reset(0).Dispose();
                AssertShows("after reset");
                for (int step = 1; step <= Steps; step++) {
                    object? action = env.ActionSpace.Sample();
                    using (StepResult<object?> result = env.Step(action)) {
                        AssertShows($"after step {step}");
                        if (result.Terminated || result.Truncated)
                            break;
                    }

                    (action as NDArray)?.Dispose();
                }
            } finally {
                env.Close();
            }

            // one state: the frame the env rendered last against the window's client area on screen
            void AssertShows(string label) {
                Assert.IsNotNull(recorder, $"{id}: no viewer was opened {label}.");
                (byte[] expected, int width, int height) = recorder!.Last ?? throw new AssertFailedException($"{id}: nothing was rendered {label}.");
                IntPtr hwnd = handleOf(recorder.Inner);
                Assert.AreNotEqual(IntPtr.Zero, hwnd, $"{id}: the viewer has no window handle.");
                ScreenCapture.Raise(hwnd);
                var watch = Stopwatch.StartNew();
                ScreenCapture.Capture capture;
                do {
                    capture = ScreenCapture.CaptureClient(hwnd);
                    if (capture.Width == width && capture.Height == height && capture.Rgb.AsSpan().SequenceEqual(expected))
                        return;
                    Thread.Sleep(50);
                } while (watch.ElapsedMilliseconds < 5_000);

                int differing = capture.Width == width && capture.Height == height ? CountDiffering(capture.Rgb, expected) : -1;
                string stem = Path.Combine(TestContext.TestResultsDirectory ?? Path.GetTempPath(), $"{id}_{recorder.Inner.GetType().Name}_{label.Replace(' ', '_')}");
                SavePng(stem + "_expected.png", expected, width, height);
                SavePng(stem + "_screen.png", capture.Rgb, capture.Width, capture.Height);
                Assert.Fail($"{id} {label}: the window's client area is {capture.Width}x{capture.Height} at {capture.Dpi} DPI, the frame {width}x{height}; " +
                            (differing >= 0 ? $"{differing} pixels differ" : "the sizes differ") + $" after 5 s (images: {stem}_*.png).");
            }
        }

        /// <summary>The pixels whose RGB differs between two images of one size.</summary>
        /// <param name="a">The first image's RGB bytes.</param>
        /// <param name="b">The second's.</param>
        /// <returns>The count.</returns>
        private static int CountDiffering(byte[] a, byte[] b) {
            int count = 0;
            for (int i = 0; i < a.Length; i += 3) {
                if (a[i] != b[i] || a[i + 1] != b[i + 1] || a[i + 2] != b[i + 2])
                    count++;
            }

            return count;
        }

        /// <summary>Writes RGB bytes as a PNG, for a failure's evidence.</summary>
        /// <param name="path">The file.</param>
        /// <param name="rgb">The pixels.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        private static void SavePng(string path, byte[] rgb, int width, int height) {
            if (width <= 0 || height <= 0)
                return;
            using Image<Rgb24> image = SixLabors.ImageSharp.Image.LoadPixelData<Rgb24>(rgb, width, height);
            image.SaveAsPng(path);
        }

        /// <summary>
        ///     A viewer that forwards everything to the viewer under test and keeps a copy of the last frame it was handed:
        ///     what the screen must show. Interactive when the wrapped viewer is, so a MuJoCo env's window keeps its input.
        /// </summary>
        private sealed class RecordingViewer : IInteractiveEnvViewer {
            /// <summary>Wraps a viewer.</summary>
            /// <param name="inner">The viewer under test.</param>
            public RecordingViewer(IEnvViewer inner) => Inner = inner;

            /// <summary>The viewer under test.</summary>
            public IEnvViewer Inner { get; }

            /// <summary>The last frame as RGB bytes with its size; <see langword="null"/> before the first.</summary>
            public (byte[] Rgb, int Width, int Height)? Last { get; private set; }

            /// <summary>Copies the frame, then shows it in the viewer under test.</summary>
            /// <param name="frame">The env's frame.</param>
            public void Render(NDArray frame) {
                using (NDArray bytes = frame.astype(np.uint8))
                    Last = (bytes.ToArray<byte>(), (int)frame.shape[1], (int)frame.shape[0]);
                Inner.Render(frame);
            }

            /// <summary>Closes the viewer under test.</summary>
            public void Close() => Inner.Close();

            /// <summary>Releases the viewer under test.</summary>
            public void Dispose() => Inner.Dispose();

            /// <summary>Starts the wrapped viewer's input, when it has any.</summary>
            public void BeginInput() => (Inner as IInteractiveEnvViewer)?.BeginInput();

            /// <summary>The wrapped viewer's input, or none.</summary>
            /// <returns>The inputs since the last call.</returns>
            public IReadOnlyList<ViewerInput> TakeInput() => (Inner as IInteractiveEnvViewer)?.TakeInput() ?? Array.Empty<ViewerInput>();
        }
    }

    /// <summary>
    ///     Reads a window's client area from the screen as the display shows it, in physical pixels: GDI's copy of the
    ///     desktop at the client area's position, with the reading thread per-monitor DPI aware for the call, so a window
    ///     Windows stretches on a scaled display is read stretched.
    /// </summary>
    internal static class ScreenCapture {
        /// <summary>A capture: the client area's RGB bytes (rows top first), its size and the window's DPI.</summary>
        /// <param name="Rgb">The pixels, <c>3 × Width × Height</c> bytes.</param>
        /// <param name="Width">The client width in physical pixels.</param>
        /// <param name="Height">The client height in physical pixels.</param>
        /// <param name="Dpi">The window's DPI (96 at 100%).</param>
        internal readonly record struct Capture(byte[] Rgb, int Width, int Height, uint Dpi);

        /// <summary>Puts a window above every other at the screen's top-left corner, so the screen shows all of its client area.</summary>
        /// <param name="hwnd">The window.</param>
        public static void Raise(IntPtr hwnd) {
            IntPtr previous = SetThreadDpiAwarenessContext(PerMonitorAwareV2);
            try {
                SetWindowPos(hwnd, HwndTopmost, 24, 24, 0, 0, SwpNoSize | SwpShowWindow);
            } finally {
                SetThreadDpiAwarenessContext(previous);
            }
        }

        /// <summary>Copies a window's client area from the screen.</summary>
        /// <param name="hwnd">The window.</param>
        /// <returns>The capture.</returns>
        /// <exception cref="InvalidOperationException">GDI refused a step of the copy.</exception>
        public static Capture CaptureClient(IntPtr hwnd) {
            // physical pixels for every coordinate below, whatever the test process declared
            IntPtr previous = SetThreadDpiAwarenessContext(PerMonitorAwareV2);
            IntPtr screen = IntPtr.Zero, memory = IntPtr.Zero, section = IntPtr.Zero, old = IntPtr.Zero;
            try {
                if (!GetClientRect(hwnd, out Rect rect))
                    throw new InvalidOperationException("GetClientRect failed.");
                int width = rect.Right - rect.Left, height = rect.Bottom - rect.Top;
                var origin = new PointI();
                ClientToScreen(hwnd, ref origin);
                if (width <= 0 || height <= 0)
                    return new Capture(Array.Empty<byte>(), width, height, GetDpiForWindow(hwnd));
                screen = GetDC(IntPtr.Zero);
                memory = CreateCompatibleDC(screen);
                var info = new BitmapInfo {
                    Size = Marshal.SizeOf<BitmapInfo>(), Width = width, Height = -height, Planes = 1, BitCount = 32, Compression = 0,
                };
                section = CreateDIBSection(screen, ref info, 0, out IntPtr bits, IntPtr.Zero, 0);
                if (section == IntPtr.Zero)
                    throw new InvalidOperationException("CreateDIBSection failed.");
                old = SelectObject(memory, section);
                if (!BitBlt(memory, 0, 0, width, height, screen, origin.X, origin.Y, SrcCopy | CaptureBlt))
                    throw new InvalidOperationException("BitBlt failed.");
                var bgra = new byte[4 * width * height];
                Marshal.Copy(bits, bgra, 0, bgra.Length);
                var rgb = new byte[3 * width * height];
                for (int i = 0, o = 0; o < rgb.Length; i += 4, o += 3) {
                    rgb[o] = bgra[i + 2];
                    rgb[o + 1] = bgra[i + 1];
                    rgb[o + 2] = bgra[i];
                }

                return new Capture(rgb, width, height, GetDpiForWindow(hwnd));
            } finally {
                // GDI objects released in reverse order, whatever step failed
                if (old != IntPtr.Zero)
                    SelectObject(memory, old);
                if (section != IntPtr.Zero)
                    DeleteObject(section);
                if (memory != IntPtr.Zero)
                    DeleteDC(memory);
                if (screen != IntPtr.Zero)
                    ReleaseDC(IntPtr.Zero, screen);
                SetThreadDpiAwarenessContext(previous);
            }
        }

        /// <summary><c>DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2</c>.</summary>
        private static readonly IntPtr PerMonitorAwareV2 = new IntPtr(-4);

        /// <summary><c>HWND_TOPMOST</c>.</summary>
        private static readonly IntPtr HwndTopmost = new IntPtr(-1);

        /// <summary><c>SWP_NOSIZE</c>, <c>SWP_SHOWWINDOW</c>.</summary>
        private const uint SwpNoSize = 0x0001, SwpShowWindow = 0x0040;

        /// <summary><c>SRCCOPY</c>, <c>CAPTUREBLT</c> (layered windows included).</summary>
        private const uint SrcCopy = 0x00CC0020, CaptureBlt = 0x40000000;

        /// <summary>Win32's <c>RECT</c>.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct Rect {
            /// <summary>The coordinates.</summary>
            public int Left, Top, Right, Bottom;
        }

        /// <summary>Win32's <c>POINT</c>.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct PointI {
            /// <summary>The coordinates.</summary>
            public int X, Y;
        }

        /// <summary>GDI's <c>BITMAPINFOHEADER</c> (no palette for 32-bit pixels).</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct BitmapInfo {
            /// <summary>The header's fields, in order.</summary>
            public int Size, Width, Height;
            /// <summary>Planes and bits per pixel.</summary>
            public short Planes, BitCount;
            /// <summary>The rest of the header.</summary>
            public int Compression, SizeImage, XPelsPerMeter, YPelsPerMeter, ClrUsed, ClrImportant;
        }

        /// <summary>Sets the calling thread's DPI awareness; returns the previous context.</summary>
        [DllImport("user32.dll")] private static extern IntPtr SetThreadDpiAwarenessContext(IntPtr dpiContext);
        /// <summary>The client area's size, in the calling thread's DPI awareness.</summary>
        [DllImport("user32.dll")] private static extern bool GetClientRect(IntPtr hWnd, out Rect rect);
        /// <summary>Converts a client point to screen coordinates.</summary>
        [DllImport("user32.dll")] private static extern bool ClientToScreen(IntPtr hWnd, ref PointI point);
        /// <summary>The window's DPI (96 at 100% scaling).</summary>
        [DllImport("user32.dll")] private static extern uint GetDpiForWindow(IntPtr hWnd);
        /// <summary>Moves a window and changes its Z order.</summary>
        [DllImport("user32.dll")] private static extern bool SetWindowPos(IntPtr hWnd, IntPtr after, int x, int y, int cx, int cy, uint flags);
        /// <summary>A device context; of the whole screen for a null window.</summary>
        [DllImport("user32.dll")] private static extern IntPtr GetDC(IntPtr hWnd);
        /// <summary>Releases a device context from <c>GetDC</c>.</summary>
        [DllImport("user32.dll")] private static extern int ReleaseDC(IntPtr hWnd, IntPtr hdc);
        /// <summary>A memory device context compatible with another.</summary>
        [DllImport("gdi32.dll")] private static extern IntPtr CreateCompatibleDC(IntPtr hdc);
        /// <summary>A bitmap whose pixels the caller reads directly.</summary>
        [DllImport("gdi32.dll")] private static extern IntPtr CreateDIBSection(IntPtr hdc, ref BitmapInfo info, uint usage, out IntPtr bits, IntPtr section, uint offset);
        /// <summary>Selects a GDI object into a device context; returns the one replaced.</summary>
        [DllImport("gdi32.dll")] private static extern IntPtr SelectObject(IntPtr hdc, IntPtr gdiObject);
        /// <summary>Copies a rectangle of pixels between device contexts.</summary>
        [DllImport("gdi32.dll")] private static extern bool BitBlt(IntPtr destination, int x, int y, int width, int height, IntPtr source, int sourceX, int sourceY, uint rop);
        /// <summary>Deletes a GDI object.</summary>
        [DllImport("gdi32.dll")] private static extern bool DeleteObject(IntPtr gdiObject);
        /// <summary>Deletes a memory device context.</summary>
        [DllImport("gdi32.dll")] private static extern bool DeleteDC(IntPtr hdc);
    }
}
