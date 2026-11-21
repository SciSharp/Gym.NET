#nullable enable
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Gym.Environments;
using Gym.Rendering.Avalonia;
using Gym.Rendering.WinForm;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SixLabors.ImageSharp.PixelFormats;
using AVImage = Avalonia.Controls.Image;

namespace Gym.Tests.Rendering {
    /// <summary>
    ///     Pins the viewer contract of DR-85: <see cref="IEnvViewer.Render"/> takes the <c>(H, W, 3)</c> uint8 frame and copies
    ///     it, <see cref="IAsyncEnvViewer"/> renders without blocking, <see cref="EnvViewerExtensions"/> falls back to the
    ///     synchronous path, and the built-in windows show the frame's very pixels both ways. The cells that open windows end
    ///     in <c>AvaloniaEnv</c> and <c>WinFormEnv</c>, so the headless gate leaves them out and they run on a desktop.
    /// </summary>
    [TestClass]
    public class ViewerContractTests {
        /// <summary>
        ///     A 3×4 frame (height 3, width 4) whose every pixel differs and whose channels differ, so a swapped axis or a
        ///     swapped red and blue shows: pixel (row r, column c) is <c>(10 r + c, 100 + c, 200 + r)</c>.
        /// </summary>
        /// <returns>A new frame the caller disposes.</returns>
        private static NDArray TestFrame() {
            var bytes = new byte[3 * 4 * 3];
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 4; c++) {
                    int at = (r * 4 + c) * 3;
                    (bytes[at], bytes[at + 1], bytes[at + 2]) = ((byte)(10 * r + c), (byte)(100 + c), (byte)(200 + r));
                }
            }

            return new NDArray(bytes, new Shape(3, 4, 3));
        }

        /// <summary>The pixel <see cref="TestFrame"/> holds at a row and column.</summary>
        /// <param name="row">The row.</param>
        /// <param name="column">The column.</param>
        /// <returns>The pixel.</returns>
        private static Rgb24 Expected(int row, int column) => new((byte)(10 * row + column), (byte)(100 + column), (byte)(200 + row));

        /// <summary>
        ///     The headless viewer checks frames as the windows do, completes its asynchronous calls at once, and is one shared
        ///     instance; the extension's fallback renders a synchronous viewer on the caller's thread and honours a cancelled
        ///     token first.
        /// </summary>
        [TestMethod]
        public async Task NullViewer_AndTheFallback_FollowTheContract() {
            IEnvViewer viewer = NullEnvViewer.Factory(4, 3, "headless");
            Assert.AreSame(viewer, NullEnvViewer.Factory(1, 1), "One shared instance.");
            Assert.IsInstanceOfType<IAsyncEnvViewer>(viewer);
            using NDArray frame = TestFrame();
            viewer.Render(frame);
            await ((IAsyncEnvViewer)viewer).RenderAsync(frame);
            using NDArray flat = np.zeros(new Shape(3, 4), np.uint8);
            Assert.ThrowsExactly<ArgumentException>(() => viewer.Render(flat), "A frame without three channels is refused.");
            Assert.ThrowsExactly<ArgumentNullException>(() => viewer.Render(null!));
            viewer.Close();
            await viewer.CloseAsync();

            var counting = new SyncOnlyViewer();
            ValueTask shown = counting.RenderAsync(frame);
            Assert.IsTrue(shown.IsCompletedSuccessfully, "A synchronous viewer renders on the caller's thread, no task pending.");
            Assert.AreEqual(1, counting.Frames);
            using var cancelled = new CancellationTokenSource();
            cancelled.Cancel();
            Assert.ThrowsExactly<OperationCanceledException>(() => counting.RenderAsync(frame, cancelled.Token));
            Assert.AreEqual(1, counting.Frames, "A cancelled render shows nothing.");
            await counting.CloseAsync();
            Assert.IsTrue(counting.Closed);
#pragma warning disable CS0618 // 0.2's name, kept obsolete until 2.0 (DR-85): it must still close.
            var legacy = new SyncOnlyViewer();
            ((IEnvViewer)legacy).CloseEnvironment();
#pragma warning restore CS0618
            Assert.IsTrue(legacy.Closed, "CloseEnvironment() closes through Close().");
        }

        /// <summary>
        ///     The WinForms window shows the frame's pixels where they belong, rendered synchronously and asynchronously, and
        ///     closes through the contract; opens a window.
        /// </summary>
        [TestMethod]
        public async Task WinFormViewer_ShowsTheFramesPixels_SyncAndAsync_WinFormEnv() {
            var viewer = (WinFormEnvViewer)WinFormEnvViewer.Factory(4, 3, "viewer contract");
            Assert.IsInstanceOfType<IAsyncEnvViewer>(viewer);
            using (NDArray frame = TestFrame()) {
                viewer.Render(frame);
                AssertShows(viewer);
            }

            using (NDArray blank = np.zeros(new Shape(3, 4, 3), np.uint8))
                viewer.Render(blank);
            using (NDArray frame = TestFrame())
                await viewer.RenderAsync(frame);
            AssertShows(viewer);
            await viewer.CloseAsync();

            static void AssertShows(WinFormEnvViewer viewer) {
                // The picture is the form's: read it on the form's thread.
                viewer.Invoke(new Action(() => {
                    var bitmap = (System.Drawing.Bitmap)viewer.PictureFrame.Image!;
                    Assert.AreEqual(4, bitmap.Width);
                    Assert.AreEqual(3, bitmap.Height);
                    for (int r = 0; r < 3; r++) {
                        for (int c = 0; c < 4; c++) {
                            System.Drawing.Color pixel = bitmap.GetPixel(c, r);
                            Assert.AreEqual(Expected(r, c), new Rgb24(pixel.R, pixel.G, pixel.B), $"pixel (row {r}, column {c})");
                        }
                    }
                }));
            }
        }

        /// <summary>
        ///     The Avalonia window shows the frame's pixels where they belong, rendered synchronously and asynchronously, and
        ///     hides through the contract; opens a window.
        /// </summary>
        [TestMethod]
        public async Task AvaloniaViewer_ShowsTheFramesPixels_SyncAndAsync_AvaloniaEnv() {
            var viewer = (AvaloniaEnvViewer)AvaloniaEnvViewer.Factory(4, 3, "viewer contract");
            Assert.IsInstanceOfType<IAsyncEnvViewer>(viewer);
            using (NDArray frame = TestFrame()) {
                viewer.Render(frame);
                AssertShows(viewer);
            }

            using (NDArray blank = np.zeros(new Shape(3, 4, 3), np.uint8))
                viewer.Render(blank);
            using (NDArray frame = TestFrame())
                await viewer.RenderAsync(frame);
            AssertShows(viewer);
            await viewer.CloseAsync();

            static void AssertShows(AvaloniaEnvViewer viewer) {
                // The bitmap is the window's: encode it on the UI thread, decode and compare here. PNG is lossless, so the
                // decoded pixels are the window's own.
                byte[] png = Avalonia.Threading.Dispatcher.UIThread.Invoke(() => {
                    var bitmap = (Avalonia.Media.Imaging.Bitmap)((AVImage)viewer.Content!).Source!;
                    using var stream = new MemoryStream();
                    bitmap.Save(stream, Avalonia.Media.Imaging.PngBitmapEncoderOptions.Default);
                    return stream.ToArray();
                });
                using var image = SixLabors.ImageSharp.Image.Load<Rgb24>(png);
                Assert.AreEqual(4, image.Width);
                Assert.AreEqual(3, image.Height);
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 4; c++)
                        Assert.AreEqual(Expected(r, c), image[c, r], $"pixel (row {r}, column {c})");
                }
            }
        }

        /// <summary>A viewer with only the synchronous path, counting what it was asked to do.</summary>
        private sealed class SyncOnlyViewer : IEnvViewer {
            /// <summary>The frames rendered.</summary>
            public int Frames { get; private set; }

            /// <summary>Whether <see cref="Close"/> ran.</summary>
            public bool Closed { get; private set; }

            /// <summary>Counts the frame.</summary>
            /// <param name="frame">The frame; not kept.</param>
            public void Render(NDArray frame) => Frames++;

            /// <summary>Records the close.</summary>
            public void Close() => Closed = true;

            /// <summary>Nothing to release.</summary>
            public void Dispose() { }
        }
    }
}
