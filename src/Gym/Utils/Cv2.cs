#nullable enable
using System;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     OpenCV's <c>cv2.resize</c> for uint8 and float32 images, bit for bit: <see cref="InterArea"/>, the interpolation
    ///     Gymnasium's <c>ResizeObservation</c> uses, and <see cref="InterLinear"/>, the default the vector
    ///     <c>HumanRendering</c> uses. Use it wherever Gymnasium calls <c>cv2.resize</c>; Gym.NET carries no native OpenCV.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A C# translation of the 8-bit paths of OpenCV 4.13.0's <c>cv::resize</c> (decision DR-23): the integer-factor
    ///     area average (<c>resizeAreaFast_</c>, with the 2×2 kernel's own rounding), the general area decimation
    ///     (<c>resizeArea_</c> with <c>computeResizeAreaTab</c>'s weights, in float32), and the fixed-point bilinear kernel
    ///     (<c>resizeGeneric_</c> with 11-bit coefficients), which <c>INTER_AREA</c> also takes, with its own coefficients,
    ///     when either axis grows. The arithmetic follows OpenCV's statement for statement: float32 where OpenCV computes in
    ///     float, round-half-to-even where it calls <c>cvRound</c>, and the integer shifts of its fixed-point kernels, so
    ///     the bytes equal the pinned <c>cv2</c>'s, which the parity suite checks against live OpenCV.
    ///     </para>
    ///     <para>
    ///     uint8 and float32 images are supported (DR-46; Gymnasium's own callers require uint8). The float32 paths are
    ///     OpenCV's float instantiations of the same kernels: the area-fast mean (its 2×2 vector kernel for 1 or 4 channels
    ///     summing pairwise over whole groups of four outputs, the generic sum in groups of four, OpenCV's unrolling), the
    ///     general area decimation copied out unrounded, and the float bilinear kernel. Footgun, float32
    ///     <see cref="InterLinear"/> only: the <c>cv2</c> wheels hand it to Intel IPP when IPP is on (the default, measured
    ///     on 2026-11-10: 255 of 400 random cases differ from OpenCV's own kernel, by up to 4.4e-4), and IPP's algorithm is
    ///     not published; this method gives OpenCV's own result, what <c>cv2</c> computes with <c>cv2.ipp.setUseIPP(False)</c>
    ///     or on a build without IPP. <see cref="InterArea"/> is the same either way.
    ///     </para>
    ///     <para>
    ///     Like <c>cv2</c>, a single-channel image comes back two-dimensional, whether it was given as <c>(H, W)</c> or
    ///     <c>(H, W, 1)</c>. Each call allocates its result and a few tables; it is single-threaded, where OpenCV splits
    ///     rows across threads (which does not change its results).
    ///     </para>
    ///     <para>
    ///     Third-party notice: OpenCV is © Intel Corporation, Willow Garage Inc., Itseez Inc. and contributors, licensed
    ///     under the Apache License 2.0; see THIRD-PARTY-NOTICES.md. Reference:
    ///     <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp">modules/imgproc/src/resize.cpp</see>.
    ///     </para>
    /// </remarks>
    public static class Cv2 {
        /// <summary><c>cv2.INTER_LINEAR</c>: bilinear interpolation, OpenCV's default.</summary>
        public const int InterLinear = 1;

        /// <summary><c>cv2.INTER_AREA</c>: area averaging when shrinking (both axes), a bilinear variant otherwise.</summary>
        public const int InterArea = 3;

        /// <summary>OpenCV's <c>INTER_RESIZE_COEF_BITS</c>: the fixed-point precision of the bilinear coefficients.</summary>
        private const int CoefBits = 11;

        /// <summary>OpenCV's <c>INTER_RESIZE_COEF_SCALE</c>, <c>1 &lt;&lt; 11</c>: a coefficient of 1.0.</summary>
        private const int CoefScale = 1 << CoefBits;

        /// <summary>C's <c>DBL_EPSILON</c> (not .NET's <see cref="double.Epsilon"/>, the smallest subnormal).</summary>
        private const double DblEpsilon = 2.220446049250313E-16;

        /// <summary>
        ///     <c>cv2.resize(src, dsize, interpolation=interpolation)</c> for a uint8 or float32 image: a new
        ///     <c>(height, width)</c> or <c>(height, width, channels)</c> image of the same dtype.
        /// </summary>
        /// <param name="src">The image, <c>(H, W)</c> or <c>(H, W, C)</c> uint8 or float32, of any layout; it is not changed.</param>
        /// <param name="dsize">The target size, OpenCV's order: <c>(width, height)</c>.</param>
        /// <param name="interpolation"><see cref="InterArea"/> or <see cref="InterLinear"/>.</param>
        /// <returns>A new array of <paramref name="src"/>'s dtype the caller owns; two-dimensional for a single channel, as <c>cv2</c> returns it.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="src"/> is <see langword="null"/>.</exception>
        /// <exception cref="NotSupportedException">
        ///     Thrown for a dtype other than uint8 and float32, an interpolation other than the two, or more than 4 channels on
        ///     the general area path (OpenCV's <c>cn &lt;= 4</c> assertion).
        /// </exception>
        /// <exception cref="ValueError">Thrown for an image that is not 2-D or 3-D or is empty, or a target size that is not positive.</exception>
        /// <remarks>Reference: <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L4201">modules/imgproc/src/resize.cpp (cv::resize)</see>.</remarks>
        public static NDArray Resize(NDArray src, (int Width, int Height) dsize, int interpolation = InterLinear) {
            if (src is null)
                throw new ArgumentNullException(nameof(src));
            if (src.typecode is not (NPTypeCode.Byte or NPTypeCode.Single))
                throw new NotSupportedException($"Cv2.Resize supports uint8 and float32 images only, got {src.dtype.name}");
            if (interpolation != InterLinear && interpolation != InterArea)
                throw new NotSupportedException($"Cv2.Resize supports INTER_LINEAR (1) and INTER_AREA (3) only, got {interpolation}");
            if (src.ndim is not (2 or 3))
                throw new ValueError($"Cv2.Resize expects an (H, W) or (H, W, C) image, got {src.ndim} dimensions");

            int srcHeight = checked((int)src.shape[0]), srcWidth = checked((int)src.shape[1]);
            int channels = src.ndim == 3 ? checked((int)src.shape[2]) : 1;
            if (srcHeight == 0 || srcWidth == 0 || channels == 0)
                throw new ValueError("Cv2.Resize expects a non-empty image");
            (int width, int height) = dsize;
            if (width <= 0 || height <= 0)
                throw new ValueError($"Cv2.Resize expects a positive target size, got ({width}, {height})");

            double invScaleX = (double)width / srcWidth, invScaleY = (double)height / srcHeight;
            // cv2 hands a one-channel Mat back as (H, W), even for an (H, W, 1) input.
            Shape shape = channels == 1 ? new Shape(height, width) : new Shape(height, width, channels);
            if (src.typecode == NPTypeCode.Single) {
                // The float32 instantiations of the same kernels (DR-46); a same-size resize is OpenCV's plain copy.
                float[] floats = src.ToArray<float>();
                var floatResult = new float[checked(height * width * channels)];
                if (width == srcWidth && height == srcHeight)
                    Array.Copy(floats, floatResult, floatResult.Length);
                else
                    ResizeCore(floats, srcWidth, srcHeight, channels, floatResult, width, height, invScaleX, invScaleY, interpolation);
                return new NDArray(floatResult, shape);
            }

            // The image's bytes in C order, whatever its layout; OpenCV reads a Mat the same way.
            byte[] source = src.ToArray<byte>();
            var result = new byte[checked(height * width * channels)];
            if (width == srcWidth && height == srcHeight)
                Buffer.BlockCopy(source, 0, result, 0, result.Length);
            else
                ResizeCore(source, srcWidth, srcHeight, channels, result, width, height, invScaleX, invScaleY, interpolation);

            return new NDArray(result, shape);
        }

        /// <summary>OpenCV's <c>hal::resize</c> for 8-bit images: picks the area-fast, area or bilinear kernel by the scales.</summary>
        /// <param name="src">The source bytes, C order.</param>
        /// <param name="srcWidth">The source width in pixels.</param>
        /// <param name="srcHeight">The source height.</param>
        /// <param name="cn">The channels.</param>
        /// <param name="dst">The destination bytes, written.</param>
        /// <param name="dstWidth">The destination width.</param>
        /// <param name="dstHeight">The destination height.</param>
        /// <param name="invScaleX"><c>dstWidth / srcWidth</c>.</param>
        /// <param name="invScaleY"><c>dstHeight / srcHeight</c>.</param>
        /// <param name="interpolation">The interpolation.</param>
        /// <exception cref="NotSupportedException">Thrown for more than 4 channels on the general area path.</exception>
        /// <remarks>Reference: <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L3826">modules/imgproc/src/resize.cpp (hal::resize)</see>.</remarks>
        private static void ResizeCore(byte[] src, int srcWidth, int srcHeight, int cn, byte[] dst, int dstWidth, int dstHeight,
            double invScaleX, double invScaleY, int interpolation) {
            double scaleX = 1.0 / invScaleX, scaleY = 1.0 / invScaleY;
            int iscaleX = CvRound(scaleX), iscaleY = CvRound(scaleY);
            bool isAreaFast = Math.Abs(scaleX - iscaleX) < DblEpsilon && Math.Abs(scaleY - iscaleY) < DblEpsilon;

            // In case of scale_x && scale_y is equal to 2, INTER_AREA (fast) also is equal to INTER_LINEAR.
            if (interpolation == InterLinear && isAreaFast && iscaleX == 2 && iscaleY == 2)
                interpolation = InterArea;

            // True "area" interpolation is only implemented for the case (scale_x >= 1 && scale_y >= 1); in other cases
            // it is emulated using some variant of bilinear interpolation.
            if (interpolation == InterArea && scaleX >= 1 && scaleY >= 1) {
                if (isAreaFast) {
                    AreaFast(src, srcWidth, srcHeight, cn, dst, dstWidth, dstHeight, iscaleX, iscaleY);
                    return;
                }

                if (cn > 4)
                    throw new NotSupportedException("error: (-215:Assertion failed) func != 0 && cn <= 4 in function 'cv::hal::resize'");
                Area(src, srcWidth, srcHeight, cn, dst, dstWidth, dstHeight, scaleX, scaleY);
                return;
            }

            Linear(src, srcWidth, srcHeight, cn, dst, dstWidth, dstHeight, scaleX, scaleY, invScaleX, invScaleY, interpolation == InterArea);
        }

        /// <summary>
        ///     OpenCV's <c>resizeAreaFast_</c> for uint8: the mean of each <c>scaleX × scaleY</c> block, the 2×2 kernel of 1,
        ///     3 or 4 channels rounding half up (<c>(a + b + c + d + 2) &gt;&gt; 2</c>, its SIMD kernel and scalar tail alike),
        ///     the other blocks <c>saturate_cast&lt;uchar&gt;(sum * (1.f / area))</c>, and blocks cut by the image's edge the
        ///     mean of the pixels they hold.
        /// </summary>
        /// <param name="src">The source bytes.</param>
        /// <param name="srcWidth">The source width.</param>
        /// <param name="srcHeight">The source height.</param>
        /// <param name="cn">The channels.</param>
        /// <param name="dst">The destination bytes.</param>
        /// <param name="dstWidth">The destination width.</param>
        /// <param name="dstHeight">The destination height.</param>
        /// <param name="scaleX">The integer horizontal factor.</param>
        /// <param name="scaleY">The integer vertical factor.</param>
        /// <remarks>Reference: <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L2975">modules/imgproc/src/resize.cpp (resizeAreaFast_Invoker)</see>.</remarks>
        private static void AreaFast(byte[] src, int srcWidth, int srcHeight, int cn, byte[] dst, int dstWidth, int dstHeight, int scaleX, int scaleY) {
            int area = scaleX * scaleY;
            float scale = 1f / area;
            int srcStep = srcWidth * cn;
            var ofs = new int[area];
            for (int sy = 0, k = 0; sy < scaleY; sy++) {
                for (int sx = 0; sx < scaleX; sx++)
                    ofs[k++] = sy * srcStep + sx * cn;
            }

            int dwidth = dstWidth * cn, swidth = srcWidth * cn;
            var xofs = new int[dwidth];
            for (int dx = 0; dx < dstWidth; dx++) {
                int j = dx * cn, sx = scaleX * j;
                for (int k = 0; k < cn; k++)
                    xofs[j + k] = sx + k;
            }

            int dwidth1 = srcWidth / scaleX * cn;
            // ResizeAreaFastVec's fast mode: its SIMD kernel and its scalar tail both compute (sum + 2) >> 2.
            bool fastMode = scaleX == 2 && scaleY == 2 && (cn == 1 || cn == 3 || cn == 4);
            for (int dy = 0; dy < dstHeight; dy++) {
                int dRow = dy * dwidth, sy0 = dy * scaleY;
                // A block row cut by the bottom edge is averaged pixel by pixel below.
                int w = sy0 + scaleY <= srcHeight ? dwidth1 : 0;
                if (sy0 >= srcHeight) {
                    Array.Clear(dst, dRow, dwidth);
                    continue;
                }

                int sRow = sy0 * srcStep, dx = 0;
                if (fastMode) {
                    int next = sRow + srcStep;
                    for (; dx < w; dx++) {
                        // The channel c of output pixel p averages input pixels 2p and 2p + 1 of both rows.
                        int index = dx / cn * 2 * cn + dx % cn;
                        dst[dRow + dx] = (byte)((src[sRow + index] + src[sRow + index + cn] + src[next + index] + src[next + index + cn] + 2) >> 2);
                    }
                }

                for (; dx < w; dx++) {
                    int s = sRow + xofs[dx], sum = 0;
                    for (int k = 0; k < area; k++)
                        sum += src[s + ofs[k]];
                    // int * float: the product rounds to float32 before cvRound, as in OpenCV.
                    dst[dRow + dx] = SaturateByte(sum * scale);
                }

                for (; dx < dwidth; dx++) {
                    int sum = 0, count = 0, sx0 = xofs[dx];
                    for (int sy = 0; sy < scaleY; sy++) {
                        if (sy0 + sy >= srcHeight)
                            break;
                        int s = (sy0 + sy) * srcStep + sx0;
                        for (int sx = 0; sx < scaleX * cn; sx += cn) {
                            if (sx0 + sx >= swidth)
                                break;
                            sum += src[s + sx];
                            count++;
                        }
                    }

                    // A block wholly past the edge has no pixels: OpenCV's 0 / 0 is NaN, which saturates to 0.
                    dst[dRow + dx] = SaturateByte((float)sum / count);
                }
            }
        }

        /// <summary>
        ///     OpenCV's <c>resizeArea_&lt;uchar, float&gt;</c>: each destination pixel the area-weighted mean of the source
        ///     pixels its cell covers, weights from <see cref="ComputeResizeAreaTab"/>, accumulated in float32 row by row and
        ///     rounded half to even.
        /// </summary>
        /// <param name="src">The source bytes.</param>
        /// <param name="srcWidth">The source width.</param>
        /// <param name="srcHeight">The source height.</param>
        /// <param name="cn">The channels, at most 4.</param>
        /// <param name="dst">The destination bytes.</param>
        /// <param name="dstWidth">The destination width.</param>
        /// <param name="dstHeight">The destination height.</param>
        /// <param name="scaleX">The horizontal factor, at least 1.</param>
        /// <param name="scaleY">The vertical factor, at least 1.</param>
        /// <exception cref="InvalidOperationException">Thrown when the vertical table skips a destination row (OpenCV's assertion; cannot happen for factors of at least 1).</exception>
        /// <remarks>Reference: <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L3183">modules/imgproc/src/resize.cpp (ResizeArea_Invoker)</see>.</remarks>
        private static void Area(byte[] src, int srcWidth, int srcHeight, int cn, byte[] dst, int dstWidth, int dstHeight, double scaleX, double scaleY) {
            var xtab = new DecimateAlpha[srcWidth * 2];
            var ytab = new DecimateAlpha[srcHeight * 2];
            int xtabSize = ComputeResizeAreaTab(srcWidth, dstWidth, cn, scaleX, xtab);
            int ytabSize = ComputeResizeAreaTab(srcHeight, dstHeight, 1, scaleY, ytab);

            var tabofs = new int[dstHeight + 1];
            int dy = 0;
            for (int k = 0; k < ytabSize; k++) {
                if (k == 0 || ytab[k].Di != ytab[k - 1].Di) {
                    if (ytab[k].Di != dy)
                        throw new InvalidOperationException("Assertion failed: ytab[k].di == dy");
                    tabofs[dy++] = k;
                }
            }

            tabofs[dy] = ytabSize;

            int dwidth = dstWidth * cn, srcStep = srcWidth * cn;
            var buf = new float[dwidth];
            var sum = new float[dwidth];
            // OpenCV splits the rows across threads, each stripe starting from a zero sum; 0 + x == x, so one stripe
            // computes the same bytes.
            int jStart = tabofs[0], jEnd = tabofs[dstHeight], prevDy = ytab[jStart].Di;
            for (int j = jStart; j < jEnd; j++) {
                float beta = ytab[j].Alpha;
                int rowDy = ytab[j].Di, sRow = ytab[j].Si * srcStep;
                Array.Clear(buf, 0, dwidth);
                for (int k = 0; k < xtabSize; k++) {
                    int sxn = xtab[k].Si, dxn = xtab[k].Di;
                    float alpha = xtab[k].Alpha;
                    for (int c = 0; c < cn; c++)
                        buf[dxn + c] = buf[dxn + c] + src[sRow + sxn + c] * alpha;
                }

                if (rowDy != prevDy) {
                    StoreRow(sum, dst, prevDy * dwidth, dwidth);
                    for (int dx = 0; dx < dwidth; dx++)
                        sum[dx] = beta * buf[dx];
                    prevDy = rowDy;
                } else {
                    for (int dx = 0; dx < dwidth; dx++)
                        sum[dx] = sum[dx] + beta * buf[dx];
                }
            }

            StoreRow(sum, dst, prevDy * dwidth, dwidth);
        }

        /// <summary>
        ///     OpenCV's <c>computeResizeAreaTab</c>: for each destination cell, the source pixels it covers and the share of
        ///     the cell each covers (a partial first pixel, whole pixels, a partial last pixel), computed in double and stored
        ///     as float.
        /// </summary>
        /// <param name="ssize">The source length.</param>
        /// <param name="dsize">The destination length.</param>
        /// <param name="cn">The channels, which scale the indices.</param>
        /// <param name="scale">The factor, <c>ssize / dsize</c>.</param>
        /// <param name="tab">The table, at least <c>2 * ssize</c> long; filled.</param>
        /// <returns>The entries written.</returns>
        /// <remarks>Reference: <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L3334">modules/imgproc/src/resize.cpp (computeResizeAreaTab)</see>.</remarks>
        private static int ComputeResizeAreaTab(int ssize, int dsize, int cn, double scale, DecimateAlpha[] tab) {
            int k = 0;
            for (int dx = 0; dx < dsize; dx++) {
                double fsx1 = dx * scale;
                double fsx2 = fsx1 + scale;
                double cellWidth = Math.Min(scale, ssize - fsx1);

                int sx1 = CvCeil(fsx1), sx2 = CvFloor(fsx2);
                sx2 = Math.Min(sx2, ssize - 1);
                sx1 = Math.Min(sx1, sx2);

                if (sx1 - fsx1 > 1e-3)
                    tab[k++] = new DecimateAlpha(dx * cn, (sx1 - 1) * cn, (float)((sx1 - fsx1) / cellWidth));

                for (int sx = sx1; sx < sx2; sx++)
                    tab[k++] = new DecimateAlpha(dx * cn, sx * cn, (float)(1.0 / cellWidth));

                if (fsx2 - sx2 > 1e-3)
                    tab[k++] = new DecimateAlpha(dx * cn, sx2 * cn, (float)(Math.Min(Math.Min(fsx2 - sx2, 1.0), cellWidth) / cellWidth));
            }

            return k;
        }

        /// <summary>
        ///     OpenCV's bilinear <c>resizeGeneric_</c> for uint8, with <c>INTER_AREA</c>'s coefficients when
        ///     <paramref name="areaMode"/>: 11-bit fixed-point horizontal weights into int rows, then the vertical blend
        ///     <c>((b0 * (r0 &gt;&gt; 4)) &gt;&gt; 16) + ((b1 * (r1 &gt;&gt; 4)) &gt;&gt; 16)</c> rounded by <c>(x + 2) &gt;&gt; 2</c>.
        /// </summary>
        /// <param name="src">The source bytes.</param>
        /// <param name="srcWidth">The source width.</param>
        /// <param name="srcHeight">The source height.</param>
        /// <param name="cn">The channels.</param>
        /// <param name="dst">The destination bytes.</param>
        /// <param name="dstWidth">The destination width.</param>
        /// <param name="dstHeight">The destination height.</param>
        /// <param name="scaleX">The horizontal factor, <c>srcWidth / dstWidth</c>.</param>
        /// <param name="scaleY">The vertical factor.</param>
        /// <param name="invScaleX">Its inverse.</param>
        /// <param name="invScaleY">Its inverse.</param>
        /// <param name="areaMode">Whether the coefficients are <c>INTER_AREA</c>'s emulation rather than bilinear's.</param>
        /// <remarks>Reference: <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L4070">modules/imgproc/src/resize.cpp (hal::resize, the bilinear tables)</see>.</remarks>
        private static void Linear(byte[] src, int srcWidth, int srcHeight, int cn, byte[] dst, int dstWidth, int dstHeight,
            double scaleX, double scaleY, double invScaleX, double invScaleY, bool areaMode) {
            const int ksize = 2, ksize2 = ksize / 2;
            int xmax = dstWidth, width = dstWidth * cn;
            var xofs = new int[width];
            var ialpha = new short[width * ksize];
            var yofs = new int[dstHeight];
            var ibeta = new short[dstHeight * ksize];

            for (int dx = 0; dx < dstWidth; dx++) {
                float fx;
                int sx;
                if (!areaMode) {
                    fx = (float)((dx + 0.5) * scaleX - 0.5);
                    sx = CvFloor(fx);
                    fx -= sx;
                } else {
                    sx = CvFloor(dx * scaleX);
                    fx = (float)(dx + 1 - (sx + 1) * invScaleX);
                    fx = fx <= 0 ? 0f : fx - CvFloor(fx);
                }

                // OpenCV also records xmin here, which only the cubic and Lanczos kernels read.
                if (sx < ksize2 - 1 && sx < 0) {
                    fx = 0;
                    sx = 0;
                }

                if (sx + ksize2 >= srcWidth) {
                    xmax = Math.Min(xmax, dx);
                    if (sx >= srcWidth - 1) {
                        fx = 0;
                        sx = srcWidth - 1;
                    }
                }

                for (int k = 0; k < cn; k++)
                    xofs[dx * cn + k] = sx * cn + k;

                int b = dx * cn * ksize;
                ialpha[b] = SaturateShort((1f - fx) * CoefScale);
                ialpha[b + 1] = SaturateShort(fx * CoefScale);
                for (int k = ksize; k < cn * ksize; k++)
                    ialpha[b + k] = ialpha[b + k - ksize];
            }

            for (int dy = 0; dy < dstHeight; dy++) {
                float fy;
                int sy;
                if (!areaMode) {
                    fy = (float)((dy + 0.5) * scaleY - 0.5);
                    sy = CvFloor(fy);
                    fy -= sy;
                } else {
                    sy = CvFloor(dy * scaleY);
                    fy = (float)(dy + 1 - (sy + 1) * invScaleY);
                    fy = fy <= 0 ? 0f : fy - CvFloor(fy);
                }

                // The vertical offsets are not clamped here: the rows are clipped when read.
                yofs[dy] = sy;
                ibeta[dy * ksize] = SaturateShort((1f - fy) * CoefScale);
                ibeta[dy * ksize + 1] = SaturateShort(fy * CoefScale);
            }

            int xmaxCn = xmax * cn, srcStep = srcWidth * cn;
            var row0 = new int[width];
            var row1 = new int[width];
            for (int dy = 0; dy < dstHeight; dy++) {
                int sy0 = yofs[dy];
                HResize(src, Clip(sy0, 0, srcHeight) * srcStep, row0, xofs, ialpha, width, xmaxCn, cn);
                HResize(src, Clip(sy0 + 1, 0, srcHeight) * srcStep, row1, xofs, ialpha, width, xmaxCn, cn);

                int b0 = ibeta[dy * ksize], b1 = ibeta[dy * ksize + 1], dRow = dy * width;
                for (int x = 0; x < width; x++) {
                    // VResizeLinear<uchar, int, short>: OpenCV's SIMD kernel saturates the result where its scalar tail
                    // truncates; the value never leaves [0, 255], so both write the same byte.
                    int value = (((b0 * (row0[x] >> 4)) >> 16) + ((b1 * (row1[x] >> 4)) >> 16) + 2) >> 2;
                    dst[dRow + x] = (byte)Math.Clamp(value, 0, 255);
                }
            }
        }

        /// <summary>
        ///     OpenCV's <c>HResizeLinear&lt;uchar, int, short, 2048&gt;</c>: one source row blended horizontally into int
        ///     values scaled by 2048, two taps up to <paramref name="xmax"/>, the edge pixel alone past it.
        /// </summary>
        /// <param name="src">The source bytes.</param>
        /// <param name="sRow">The row's first byte.</param>
        /// <param name="row">The destination row, written.</param>
        /// <param name="xofs">Each destination element's first source element.</param>
        /// <param name="alpha">The two weights of each destination element.</param>
        /// <param name="width">The destination row length, in elements.</param>
        /// <param name="xmax">The first element whose second tap would be past the edge.</param>
        /// <param name="cn">The channels: the second tap is <paramref name="cn"/> elements on.</param>
        /// <remarks>Reference: <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L1878">modules/imgproc/src/resize.cpp (HResizeLinear)</see>.</remarks>
        private static void HResize(byte[] src, int sRow, int[] row, int[] xofs, short[] alpha, int width, int xmax, int cn) {
            int dx = 0;
            for (; dx < xmax; dx++) {
                int sx = sRow + xofs[dx];
                row[dx] = src[sx] * alpha[dx * 2] + src[sx + cn] * alpha[dx * 2 + 1];
            }

            for (; dx < width; dx++)
                row[dx] = src[sRow + xofs[dx]] * CoefScale;
        }

        /// <summary>OpenCV's <c>hal::resize</c> for float32 images: the kernel choice of the uint8 overload, with the float kernels.</summary>
        /// <param name="src">The source floats, C order.</param>
        /// <param name="srcWidth">The source width in pixels.</param>
        /// <param name="srcHeight">The source height.</param>
        /// <param name="cn">The channels.</param>
        /// <param name="dst">The destination floats, written.</param>
        /// <param name="dstWidth">The destination width.</param>
        /// <param name="dstHeight">The destination height.</param>
        /// <param name="invScaleX"><c>dstWidth / srcWidth</c>.</param>
        /// <param name="invScaleY"><c>dstHeight / srcHeight</c>.</param>
        /// <param name="interpolation">The interpolation.</param>
        /// <exception cref="NotSupportedException">Thrown for more than 4 channels on the general area path.</exception>
        /// <remarks>Reference: <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L3826">modules/imgproc/src/resize.cpp (hal::resize)</see>.</remarks>
        private static void ResizeCore(float[] src, int srcWidth, int srcHeight, int cn, float[] dst, int dstWidth, int dstHeight,
            double invScaleX, double invScaleY, int interpolation) {
            double scaleX = 1.0 / invScaleX, scaleY = 1.0 / invScaleY;
            int iscaleX = CvRound(scaleX), iscaleY = CvRound(scaleY);
            bool isAreaFast = Math.Abs(scaleX - iscaleX) < DblEpsilon && Math.Abs(scaleY - iscaleY) < DblEpsilon;

            // In case of scale_x && scale_y is equal to 2, INTER_AREA (fast) also is equal to INTER_LINEAR.
            if (interpolation == InterLinear && isAreaFast && iscaleX == 2 && iscaleY == 2)
                interpolation = InterArea;

            if (interpolation == InterArea && scaleX >= 1 && scaleY >= 1) {
                if (isAreaFast) {
                    AreaFast(src, srcWidth, srcHeight, cn, dst, dstWidth, dstHeight, iscaleX, iscaleY);
                    return;
                }

                if (cn > 4)
                    throw new NotSupportedException("error: (-215:Assertion failed) func != 0 && cn <= 4 in function 'cv::hal::resize'");
                Area(src, srcWidth, srcHeight, cn, dst, dstWidth, dstHeight, scaleX, scaleY);
                return;
            }

            Linear(src, srcWidth, srcHeight, cn, dst, dstWidth, dstHeight, scaleX, scaleY, invScaleX, invScaleY, interpolation == InterArea);
        }

        /// <summary>
        ///     OpenCV's <c>resizeAreaFast_&lt;float, float, ResizeAreaFastVec_SIMD_32f&gt;</c>: the mean of each
        ///     <c>scaleX × scaleY</c> block in float32. Two orders of summation, as OpenCV's kernels run them: for 2×2 blocks
        ///     of 1 or 4 channels, the 128-bit vector kernel takes whole groups of four outputs and sums each block pairwise,
        ///     <c>((a + b) + (c + d)) * 0.25f</c>; every other output (that kernel's tail, the other blocks and channel counts)
        ///     takes the generic loop, unrolled by four, <c>sum += ((a + b) + c) + d</c>, then <c>sum * (1.f / area)</c>.
        ///     Blocks cut by the image's edge are the mean of the pixels they hold (NaN for none, OpenCV's <c>0 / 0</c>).
        /// </summary>
        /// <param name="src">The source floats.</param>
        /// <param name="srcWidth">The source width.</param>
        /// <param name="srcHeight">The source height.</param>
        /// <param name="cn">The channels.</param>
        /// <param name="dst">The destination floats.</param>
        /// <param name="dstWidth">The destination width.</param>
        /// <param name="dstHeight">The destination height.</param>
        /// <param name="scaleX">The integer horizontal factor.</param>
        /// <param name="scaleY">The integer vertical factor.</param>
        /// <remarks>
        ///     The four-lane grouping is the <c>cv2</c> wheels' x86 baseline (SSE, <c>CV_SIMD_WIDTH</c> 16), which the parity
        ///     suite pins on its corpus. Reference: <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L2857">modules/imgproc/src/resize.cpp (ResizeAreaFastVec_SIMD_32f)</see>
        ///     and <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L2975">resizeAreaFast_Invoker</see>.
        /// </remarks>
        private static void AreaFast(float[] src, int srcWidth, int srcHeight, int cn, float[] dst, int dstWidth, int dstHeight, int scaleX, int scaleY) {
            int area = scaleX * scaleY;
            float scale = 1f / area;
            int srcStep = srcWidth * cn;
            var ofs = new int[area];
            for (int sy = 0, k = 0; sy < scaleY; sy++) {
                for (int sx = 0; sx < scaleX; sx++)
                    ofs[k++] = sy * srcStep + sx * cn;
            }

            int dwidth = dstWidth * cn, swidth = srcWidth * cn;
            var xofs = new int[dwidth];
            for (int dx = 0; dx < dstWidth; dx++) {
                int j = dx * cn, sx = scaleX * j;
                for (int k = 0; k < cn; k++)
                    xofs[j + k] = sx + k;
            }

            int dwidth1 = srcWidth / scaleX * cn;
            bool fastMode = scaleX == 2 && scaleY == 2 && (cn == 1 || cn == 4);
            for (int dy = 0; dy < dstHeight; dy++) {
                int dRow = dy * dwidth, sy0 = dy * scaleY;
                int w = sy0 + scaleY <= srcHeight ? dwidth1 : 0;
                if (sy0 >= srcHeight) {
                    Array.Clear(dst, dRow, dwidth);
                    continue;
                }

                int sRow = sy0 * srcStep, dx = 0;
                if (fastMode) {
                    // The vector kernel's loop, `dx <= w - 4` by fours: its outputs are elementwise, so they are computed one
                    // by one here, over the same whole groups; the tail goes to the generic loop below.
                    int next = sRow + srcStep, grouped = w / 4 * 4;
                    for (; dx < grouped; dx++) {
                        int s, n, step;
                        if (cn == 1) {
                            // v_load_deinterleave: output dx takes the even and odd pixels 2dx and 2dx + 1 of both rows.
                            s = sRow + 2 * dx;
                            n = next + 2 * dx;
                            step = 1;
                        } else {
                            // Four channels: output float dx is channel dx % 4 of pixel dx / 4, from pixels 2(dx / 4) and 2(dx / 4) + 1.
                            int pixelFloat = dx / 4 * 8 + dx % 4;
                            s = sRow + pixelFloat;
                            n = next + pixelFloat;
                            step = 4;
                        }

                        dst[dRow + dx] = ((src[s] + src[s + step]) + (src[n] + src[n + step])) * 0.25f;
                    }
                }

                for (; dx < w; dx++) {
                    int s = sRow + xofs[dx], k = 0;
                    float sum = 0f;
                    // CV_ENABLE_UNROLLED: four samples summed left to right, then added to the sum, then the rest one by one.
                    for (; k <= area - 4; k += 4)
                        sum += src[s + ofs[k]] + src[s + ofs[k + 1]] + src[s + ofs[k + 2]] + src[s + ofs[k + 3]];
                    for (; k < area; k++)
                        sum += src[s + ofs[k]];
                    dst[dRow + dx] = sum * scale;
                }

                for (; dx < dwidth; dx++) {
                    float sum = 0f;
                    int count = 0, sx0 = xofs[dx];
                    for (int sy = 0; sy < scaleY; sy++) {
                        if (sy0 + sy >= srcHeight)
                            break;
                        int s = (sy0 + sy) * srcStep + sx0;
                        for (int sx = 0; sx < scaleX * cn; sx += cn) {
                            if (sx0 + sx >= swidth)
                                break;
                            sum += src[s + sx];
                            count++;
                        }
                    }

                    // (float)sum / count: a float division, NaN for a block with no pixels.
                    dst[dRow + dx] = sum / count;
                }
            }
        }

        /// <summary>
        ///     OpenCV's <c>resizeArea_&lt;float, float&gt;</c>: the uint8 overload's area-weighted means, accumulated in float32
        ///     row by row the same way, and copied out as they are (<c>inter_area::saturate_store</c> for <c>T == WT</c>).
        /// </summary>
        /// <param name="src">The source floats.</param>
        /// <param name="srcWidth">The source width.</param>
        /// <param name="srcHeight">The source height.</param>
        /// <param name="cn">The channels, at most 4.</param>
        /// <param name="dst">The destination floats.</param>
        /// <param name="dstWidth">The destination width.</param>
        /// <param name="dstHeight">The destination height.</param>
        /// <param name="scaleX">The horizontal factor, at least 1.</param>
        /// <param name="scaleY">The vertical factor, at least 1.</param>
        /// <exception cref="InvalidOperationException">Thrown when the vertical table skips a destination row (OpenCV's assertion; cannot happen for factors of at least 1).</exception>
        /// <remarks>Reference: <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L3183">modules/imgproc/src/resize.cpp (ResizeArea_Invoker)</see>.</remarks>
        private static void Area(float[] src, int srcWidth, int srcHeight, int cn, float[] dst, int dstWidth, int dstHeight, double scaleX, double scaleY) {
            var xtab = new DecimateAlpha[srcWidth * 2];
            var ytab = new DecimateAlpha[srcHeight * 2];
            int xtabSize = ComputeResizeAreaTab(srcWidth, dstWidth, cn, scaleX, xtab);
            int ytabSize = ComputeResizeAreaTab(srcHeight, dstHeight, 1, scaleY, ytab);

            var tabofs = new int[dstHeight + 1];
            int dy = 0;
            for (int k = 0; k < ytabSize; k++) {
                if (k == 0 || ytab[k].Di != ytab[k - 1].Di) {
                    if (ytab[k].Di != dy)
                        throw new InvalidOperationException("Assertion failed: ytab[k].di == dy");
                    tabofs[dy++] = k;
                }
            }

            tabofs[dy] = ytabSize;

            int dwidth = dstWidth * cn, srcStep = srcWidth * cn;
            var buf = new float[dwidth];
            var sum = new float[dwidth];
            // One stripe, as in the uint8 overload: a stripe's first rows add to zero, which changes no float (the buffers
            // start at +0 and never hold -0).
            int jStart = tabofs[0], jEnd = tabofs[dstHeight], prevDy = ytab[jStart].Di;
            for (int j = jStart; j < jEnd; j++) {
                float beta = ytab[j].Alpha;
                int rowDy = ytab[j].Di, sRow = ytab[j].Si * srcStep;
                Array.Clear(buf, 0, dwidth);
                for (int k = 0; k < xtabSize; k++) {
                    int sxn = xtab[k].Si, dxn = xtab[k].Di;
                    float alpha = xtab[k].Alpha;
                    for (int c = 0; c < cn; c++)
                        buf[dxn + c] = buf[dxn + c] + src[sRow + sxn + c] * alpha;
                }

                if (rowDy != prevDy) {
                    Array.Copy(sum, 0, dst, prevDy * dwidth, dwidth);
                    for (int dx = 0; dx < dwidth; dx++)
                        sum[dx] = beta * buf[dx];
                    prevDy = rowDy;
                } else {
                    for (int dx = 0; dx < dwidth; dx++)
                        sum[dx] = sum[dx] + beta * buf[dx];
                }
            }

            Array.Copy(sum, 0, dst, prevDy * dwidth, dwidth);
        }

        /// <summary>
        ///     OpenCV's float bilinear <c>resizeGeneric_</c> (<c>HResizeLinear&lt;float, float, float, 1&gt;</c> and
        ///     <c>VResizeLinear&lt;float, float, float&gt;</c>), with <c>INTER_AREA</c>'s coefficients when
        ///     <paramref name="areaMode"/>: float weights <c>1 - f</c> and <c>f</c>, each row blended horizontally
        ///     <c>S[x]·a0 + S[x + cn]·a1</c>, then the two rows <c>r0·b0 + r1·b1</c>, all in float32 with no fused
        ///     multiply-add (the wheels' SSE baseline has none), so the vector and scalar kernels agree.
        /// </summary>
        /// <param name="src">The source floats.</param>
        /// <param name="srcWidth">The source width.</param>
        /// <param name="srcHeight">The source height.</param>
        /// <param name="cn">The channels.</param>
        /// <param name="dst">The destination floats.</param>
        /// <param name="dstWidth">The destination width.</param>
        /// <param name="dstHeight">The destination height.</param>
        /// <param name="scaleX">The horizontal factor, <c>srcWidth / dstWidth</c>.</param>
        /// <param name="scaleY">The vertical factor.</param>
        /// <param name="invScaleX">Its inverse.</param>
        /// <param name="invScaleY">Its inverse.</param>
        /// <param name="areaMode">Whether the coefficients are <c>INTER_AREA</c>'s emulation rather than bilinear's.</param>
        /// <remarks>Reference: <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L1877">modules/imgproc/src/resize.cpp (HResizeLinear, VResizeLinear)</see>.</remarks>
        private static void Linear(float[] src, int srcWidth, int srcHeight, int cn, float[] dst, int dstWidth, int dstHeight,
            double scaleX, double scaleY, double invScaleX, double invScaleY, bool areaMode) {
            const int ksize = 2, ksize2 = ksize / 2;
            int xmax = dstWidth, width = dstWidth * cn;
            var xofs = new int[width];
            var alpha = new float[width * ksize];
            var yofs = new int[dstHeight];
            var beta = new float[dstHeight * ksize];

            for (int dx = 0; dx < dstWidth; dx++) {
                float fx;
                int sx;
                if (!areaMode) {
                    fx = (float)((dx + 0.5) * scaleX - 0.5);
                    sx = CvFloor(fx);
                    fx -= sx;
                } else {
                    sx = CvFloor(dx * scaleX);
                    fx = (float)(dx + 1 - (sx + 1) * invScaleX);
                    fx = fx <= 0 ? 0f : fx - CvFloor(fx);
                }

                if (sx < ksize2 - 1 && sx < 0) {
                    fx = 0;
                    sx = 0;
                }

                if (sx + ksize2 >= srcWidth) {
                    xmax = Math.Min(xmax, dx);
                    if (sx >= srcWidth - 1) {
                        fx = 0;
                        sx = srcWidth - 1;
                    }
                }

                for (int k = 0; k < cn; k++)
                    xofs[dx * cn + k] = sx * cn + k;

                int b = dx * cn * ksize;
                alpha[b] = 1f - fx;
                alpha[b + 1] = fx;
                for (int k = ksize; k < cn * ksize; k++)
                    alpha[b + k] = alpha[b + k - ksize];
            }

            for (int dy = 0; dy < dstHeight; dy++) {
                float fy;
                int sy;
                if (!areaMode) {
                    fy = (float)((dy + 0.5) * scaleY - 0.5);
                    sy = CvFloor(fy);
                    fy -= sy;
                } else {
                    sy = CvFloor(dy * scaleY);
                    fy = (float)(dy + 1 - (sy + 1) * invScaleY);
                    fy = fy <= 0 ? 0f : fy - CvFloor(fy);
                }

                // Not clamped: the rows are clipped when read, the weights kept (an edge row blends with itself).
                yofs[dy] = sy;
                beta[dy * ksize] = 1f - fy;
                beta[dy * ksize + 1] = fy;
            }

            int xmaxCn = xmax * cn, srcStep = srcWidth * cn;
            var row0 = new float[width];
            var row1 = new float[width];
            for (int dy = 0; dy < dstHeight; dy++) {
                int sy0 = yofs[dy];
                HResize(src, Clip(sy0, 0, srcHeight) * srcStep, row0, xofs, alpha, width, xmaxCn, cn);
                HResize(src, Clip(sy0 + 1, 0, srcHeight) * srcStep, row1, xofs, alpha, width, xmaxCn, cn);

                float b0 = beta[dy * ksize], b1 = beta[dy * ksize + 1];
                int dRow = dy * width;
                for (int x = 0; x < width; x++)
                    dst[dRow + x] = row0[x] * b0 + row1[x] * b1;
            }
        }

        /// <summary>
        ///     OpenCV's <c>HResizeLinear&lt;float, float, float, 1&gt;</c>: one source row blended horizontally, two taps up
        ///     to <paramref name="xmax"/>, the edge pixel alone (times one) past it.
        /// </summary>
        /// <param name="src">The source floats.</param>
        /// <param name="sRow">The row's first float.</param>
        /// <param name="row">The destination row, written.</param>
        /// <param name="xofs">Each destination element's first source element.</param>
        /// <param name="alpha">The two weights of each destination element.</param>
        /// <param name="width">The destination row length, in elements.</param>
        /// <param name="xmax">The first element whose second tap would be past the edge.</param>
        /// <param name="cn">The channels: the second tap is <paramref name="cn"/> elements on.</param>
        /// <remarks>Reference: <see href="https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/resize.cpp#L1877">modules/imgproc/src/resize.cpp (HResizeLinear)</see>.</remarks>
        private static void HResize(float[] src, int sRow, float[] row, int[] xofs, float[] alpha, int width, int xmax, int cn) {
            int dx = 0;
            for (; dx < xmax; dx++) {
                int sx = sRow + xofs[dx];
                row[dx] = src[sx] * alpha[dx * 2] + src[sx + cn] * alpha[dx * 2 + 1];
            }

            for (; dx < width; dx++)
                row[dx] = src[sRow + xofs[dx]];
        }

        /// <summary>Writes a row of float sums as bytes (<c>inter_area::saturate_store</c>): rounded half to even, clamped.</summary>
        /// <param name="sum">The sums.</param>
        /// <param name="dst">The destination bytes.</param>
        /// <param name="offset">The row's first byte.</param>
        /// <param name="width">The row length.</param>
        private static void StoreRow(float[] sum, byte[] dst, int offset, int width) {
            for (int dx = 0; dx < width; dx++)
                dst[offset + dx] = SaturateByte(sum[dx]);
        }

        /// <summary>
        ///     <c>saturate_cast&lt;uchar&gt;(float)</c>: <c>cvRound</c> (round half to even, NaN to <c>INT_MIN</c>) then
        ///     clamped to [0, 255].
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The byte.</returns>
        private static byte SaturateByte(float value) {
            if (float.IsNaN(value))
                return 0;
            return (byte)Math.Clamp(MathF.Round(value, MidpointRounding.ToEven), 0f, 255f);
        }

        /// <summary><c>saturate_cast&lt;short&gt;(float)</c>: <c>cvRound</c>, then clamped to the short range.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The short.</returns>
        private static short SaturateShort(float value) => (short)Math.Clamp(MathF.Round(value, MidpointRounding.ToEven), short.MinValue, short.MaxValue);

        /// <summary><c>cvRound(double)</c> (<c>saturate_cast&lt;int&gt;</c>): the nearest integer, ties to even.</summary>
        /// <param name="value">The value, within the int range.</param>
        /// <returns>The integer.</returns>
        private static int CvRound(double value) => (int)Math.Round(value, MidpointRounding.ToEven);

        /// <summary><c>cvFloor(double)</c>.</summary>
        /// <param name="value">The value, within the int range.</param>
        /// <returns>The largest integer not above it.</returns>
        private static int CvFloor(double value) => (int)Math.Floor(value);

        /// <summary><c>cvFloor(float)</c>.</summary>
        /// <param name="value">The value, within the int range.</param>
        /// <returns>The largest integer not above it.</returns>
        private static int CvFloor(float value) => (int)MathF.Floor(value);

        /// <summary><c>cvCeil(double)</c>.</summary>
        /// <param name="value">The value, within the int range.</param>
        /// <returns>The smallest integer not below it.</returns>
        private static int CvCeil(double value) => (int)Math.Ceiling(value);

        /// <summary>OpenCV's <c>clip(x, a, b)</c>: <paramref name="x"/> limited to [<paramref name="a"/>, <paramref name="b"/> - 1].</summary>
        /// <param name="x">The value.</param>
        /// <param name="a">The lower limit.</param>
        /// <param name="b">One past the upper limit.</param>
        /// <returns>The clipped value.</returns>
        private static int Clip(int x, int a, int b) => x >= a ? (x < b ? x : b - 1) : a;

        /// <summary>One entry of an area table (OpenCV's <c>DecimateAlpha</c>): a destination index, a source index and the weight.</summary>
        /// <param name="Di">The destination element.</param>
        /// <param name="Si">The source element.</param>
        /// <param name="Alpha">The source element's share of the destination cell.</param>
        private readonly record struct DecimateAlpha(int Di, int Si, float Alpha);
    }
}
