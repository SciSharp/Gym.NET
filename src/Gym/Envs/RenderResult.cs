#nullable enable
using System;
using System.Collections.Generic;
using NumSharp;

namespace Gym.Envs {
    /// <summary>
    ///     What <c>render()</c> returns, Gymnasium's <c>RenderFrame | list[RenderFrame] | None</c> as a closed union: nothing
    ///     (<see cref="None"/>, <c>human</c> mode and envs without a render mode), one RGB frame (<see cref="Rgb"/>,
    ///     <c>rgb_array</c>), one depth frame (<see cref="Depth"/>, <c>depth_array</c>), an RGB-D pair (<see cref="RgbDepth"/>,
    ///     <c>rgbd_tuple</c>), text (<see cref="Ansi"/>, <c>ansi</c>), or the frames collected since the last call
    ///     (<see cref="Frames"/>, the <c>*_list</c> modes). Switch on the case: <c>if (env.Render() is RenderResult.Rgb rgb)</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A frame array is handed over to the caller, who owns it: envs return a fresh array per call (Gymnasium's
    ///     <c>np.transpose(np.array(...))</c> copies), so disposing it never affects the env.
    ///     </para>
    ///     <para>
    ///     Equality is structural: the same case with equal contents, arrays compared by shape, dtype and elements (never
    ///     with <see cref="NDArray"/>'s broadcasting <c>==</c>). Python has no such union, the distinction between an RGB and
    ///     a depth frame (both <c>ndarray</c>) is the render mode's; it is kept here so consumers need not look it up.
    ///     </para>
    /// </remarks>
    public abstract class RenderResult : IEquatable<RenderResult> {
        /// <summary>Only the nested cases derive from this class, so a switch over them is exhaustive.</summary>
        private RenderResult() { }

        /// <summary>The single <see cref="NoneResult"/> value: Python's <c>None</c> from <c>render()</c>.</summary>
        public static RenderResult None { get; } = new NoneResult();

        /// <summary>Whether this is <see cref="None"/>, what <c>render()</c> returns in <c>human</c> mode or without a render mode.</summary>
        public bool IsNone => this is NoneResult;

        /// <summary>Structural equality (see the class remarks).</summary>
        /// <param name="other">The other result; may be <see langword="null"/>.</param>
        /// <returns><see langword="true"/> when both are the same case with equal contents.</returns>
        public abstract bool Equals(RenderResult? other);

        /// <summary>Structural equality with any object (see the class remarks).</summary>
        /// <param name="obj">The other object.</param>
        /// <returns><see langword="true"/> when it is an equal <see cref="RenderResult"/>.</returns>
        public override bool Equals(object? obj) => obj is RenderResult other && Equals(other);

        /// <summary>A hash consistent with <see cref="Equals(RenderResult)"/>: the case and the shapes, never the pixels.</summary>
        /// <returns>The hash.</returns>
        public abstract override int GetHashCode();

        /// <summary>Exact array equality: the same shape, the same dtype, then identical elements.</summary>
        /// <param name="left">The first array.</param>
        /// <param name="right">The second array.</param>
        /// <returns><see langword="true"/> when the arrays are the same values in the same layout.</returns>
        private static bool ArraysEqual(NDArray left, NDArray right) {
            if (ReferenceEquals(left, right))
                return true;
            return left.Shape == right.Shape && left.dtype == right.dtype && np.array_equal(left, right);
        }

        /// <summary>Python's <c>None</c>: nothing was rendered for the caller (<c>human</c> mode draws to a window instead).</summary>
        public sealed class NoneResult : RenderResult {
            /// <summary>Only <see cref="RenderResult.None"/> exists.</summary>
            internal NoneResult() { }

            /// <inheritdoc />
            public override bool Equals(RenderResult? other) => other is NoneResult;

            /// <inheritdoc />
            public override int GetHashCode() => 0;

            /// <summary>Python's <c>repr(None)</c>.</summary>
            /// <returns><c>None</c>.</returns>
            public override string ToString() => "None";
        }

        /// <summary>
        ///     An <c>rgb_array</c> frame: a <c>(height, width, 3)</c> uint8 array, rows top to bottom, as Gymnasium returns it
        ///     after <c>np.transpose(..., axes=(1, 0, 2))</c>.
        /// </summary>
        public sealed class Rgb : RenderResult {
            /// <summary>Wraps a frame.</summary>
            /// <param name="frame">The frame; the result takes it over, and the caller of <c>render()</c> owns it.</param>
            /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
            public Rgb(NDArray frame) {
                Frame = frame ?? throw new ArgumentNullException(nameof(frame));
            }

            /// <summary>The frame.</summary>
            public NDArray Frame { get; }

            /// <inheritdoc />
            public override bool Equals(RenderResult? other) => other is Rgb rgb && ArraysEqual(Frame, rgb.Frame);

            /// <inheritdoc />
            public override int GetHashCode() => HashCode.Combine(1, Frame.Shape);

            /// <summary>Describes the frame by shape and dtype, not by its pixels.</summary>
            /// <returns>For example <c>Rgb((400, 600, 3), uint8)</c>.</returns>
            public override string ToString() => $"Rgb({Frame.Shape}, {Frame.dtype.name})";
        }

        /// <summary>A <c>depth_array</c> frame: a <c>(height, width)</c> float array of depths (MuJoCo).</summary>
        public sealed class Depth : RenderResult {
            /// <summary>Wraps a depth frame.</summary>
            /// <param name="frame">The frame; the caller of <c>render()</c> owns it.</param>
            /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
            public Depth(NDArray frame) {
                Frame = frame ?? throw new ArgumentNullException(nameof(frame));
            }

            /// <summary>The depth frame.</summary>
            public NDArray Frame { get; }

            /// <inheritdoc />
            public override bool Equals(RenderResult? other) => other is Depth depth && ArraysEqual(Frame, depth.Frame);

            /// <inheritdoc />
            public override int GetHashCode() => HashCode.Combine(2, Frame.Shape);

            /// <summary>Describes the frame by shape and dtype.</summary>
            /// <returns>For example <c>Depth((480, 480), float32)</c>.</returns>
            public override string ToString() => $"Depth({Frame.Shape}, {Frame.dtype.name})";
        }

        /// <summary>An <c>rgbd_tuple</c> frame: the RGB frame and the depth frame of the same view (MuJoCo), Python's <c>(rgb, depth)</c>.</summary>
        public sealed class RgbDepth : RenderResult {
            /// <summary>Wraps the two frames.</summary>
            /// <param name="rgb">The <c>(height, width, 3)</c> uint8 frame.</param>
            /// <param name="depth">The <c>(height, width)</c> depth frame.</param>
            /// <exception cref="ArgumentNullException">Thrown when either frame is <see langword="null"/>.</exception>
            public RgbDepth(NDArray rgb, NDArray depth) {
                RgbFrame = rgb ?? throw new ArgumentNullException(nameof(rgb));
                DepthFrame = depth ?? throw new ArgumentNullException(nameof(depth));
            }

            /// <summary>The RGB frame.</summary>
            public NDArray RgbFrame { get; }

            /// <summary>The depth frame.</summary>
            public NDArray DepthFrame { get; }

            /// <inheritdoc />
            public override bool Equals(RenderResult? other) => other is RgbDepth pair && ArraysEqual(RgbFrame, pair.RgbFrame) && ArraysEqual(DepthFrame, pair.DepthFrame);

            /// <inheritdoc />
            public override int GetHashCode() => HashCode.Combine(3, RgbFrame.Shape, DepthFrame.Shape);

            /// <summary>Describes both frames by shape and dtype.</summary>
            /// <returns>The text.</returns>
            public override string ToString() => $"RgbDepth({RgbFrame.Shape}, {RgbFrame.dtype.name}; {DepthFrame.Shape}, {DepthFrame.dtype.name})";
        }

        /// <summary>An <c>ansi</c> frame: text, possibly with newlines and ANSI escape sequences (toy_text).</summary>
        public sealed class Ansi : RenderResult {
            /// <summary>Wraps the text.</summary>
            /// <param name="text">The text.</param>
            /// <exception cref="ArgumentNullException">Thrown when <paramref name="text"/> is <see langword="null"/>.</exception>
            public Ansi(string text) {
                Text = text ?? throw new ArgumentNullException(nameof(text));
            }

            /// <summary>The text.</summary>
            public string Text { get; }

            /// <inheritdoc />
            public override bool Equals(RenderResult? other) => other is Ansi ansi && string.Equals(Text, ansi.Text, StringComparison.Ordinal);

            /// <inheritdoc />
            public override int GetHashCode() => HashCode.Combine(4, StringComparer.Ordinal.GetHashCode(Text));

            /// <summary>The text itself.</summary>
            /// <returns>The text.</returns>
            public override string ToString() => Text;
        }

        /// <summary>
        ///     The frames collected since the previous call, what the <c>*_list</c> render modes return (Gymnasium's
        ///     <c>RenderCollection</c>): a Python list of frames, each itself a <see cref="RenderResult"/> case.
        /// </summary>
        public sealed class Frames : RenderResult {
            /// <summary>Wraps the collected frames.</summary>
            /// <param name="items">The frames, oldest first; copied, so the caller's list can be reused.</param>
            /// <exception cref="ArgumentNullException">Thrown when <paramref name="items"/> or one of them is <see langword="null"/>.</exception>
            public Frames(IEnumerable<RenderResult> items) {
                if (items is null)
                    throw new ArgumentNullException(nameof(items));
                var list = new List<RenderResult>();
                foreach (RenderResult item in items)
                    list.Add(item ?? throw new ArgumentNullException(nameof(items), "A collected frame is null."));
                Items = list;
            }

            /// <summary>The frames, oldest first.</summary>
            public IReadOnlyList<RenderResult> Items { get; }

            /// <inheritdoc />
            public override bool Equals(RenderResult? other) {
                if (other is not Frames frames || frames.Items.Count != Items.Count)
                    return false;
                for (int i = 0; i < Items.Count; i++) {
                    if (!Items[i].Equals(frames.Items[i]))
                        return false;
                }

                return true;
            }

            /// <inheritdoc />
            public override int GetHashCode() => HashCode.Combine(5, Items.Count);

            /// <summary>Describes the list by its length.</summary>
            /// <returns>For example <c>Frames(3)</c>.</returns>
            public override string ToString() => $"Frames({Items.Count})";
        }
    }
}
