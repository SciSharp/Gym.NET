using System;
using System.IO;
using System.IO.Compression;
using System.Threading;

namespace Gym.Environments.Envs.MuJoCo.Rendering {
    /// <summary>
    ///     The reference GPU's viewport-stage reciprocal <c>1/w</c>, bit for bit: the correctly rounded reciprocal nudged by
    ///     the measured difference (−1, 0 or +1 ulp) of the GPU's own result for the same mantissa.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The GPU divides clip coordinates by <c>w</c> through a reciprocal unit that is faithful (always within one ulp)
    ///     but not correctly rounded: 1,108,522 of the 2^23 float32 mantissas give the neighbouring float instead
    ///     (measured 2026-11-21 on the RTX 4060 Ti, driver 616.56, DR-98, by drawing one point per mantissa at clip depth 1,
    ///     whose stored depth is the reciprocal itself; <c>tools/parity/mujoco/gpu_probe.py</c>'s <c>rcp</c> probe). The
    ///     result depends only on the mantissa: a power of two scales it exactly, so one table of the 2^23 differences
    ///     serves every exponent. The window depth is then <c>z · rcp(w)</c> rounded to nearest, which reproduced every one
    ///     of 8,192 random clip depths where the correctly rounded quotient got 71%.
    ///     </para>
    ///     <para>
    ///     The table is the embedded <c>assets/mujoco/gpu/rcp.bin</c> (151 KB): an 8-byte header <c>GYMRCP1\0</c>, then a
    ///     raw deflate stream of the differences, two bits each (difference + 1), four to a byte, the first in the low
    ///     bits. It is decoded once per process on first use, into 2 MB.
    ///     </para>
    ///     <para>
    ///     Footgun: this is one GPU's arithmetic, not OpenGL's; another GPU's reciprocal differs, which is why the frame
    ///     cells are pinned to the reference GPU (DR-71).
    ///     </para>
    /// </remarks>
    internal static class GlReciprocal {
        /// <summary>The embedded table's resource name.</summary>
        private const string ResourceName = "assets/mujoco/gpu/rcp.bin";

        /// <summary>The decoded table, 2 bits per mantissa; created by the first <see cref="Of"/>.</summary>
        private static readonly Lazy<byte[]> Table = new(Load, LazyThreadSafetyMode.ExecutionAndPublication);

        /// <summary>
        ///     The GPU's reciprocal of <paramref name="w"/>: <c>1/w</c> rounded to nearest, then moved by the measured
        ///     difference for <paramref name="w"/>'s mantissa. Zero, subnormal, infinite and NaN inputs, and results that
        ///     would leave the normal range, keep the plain reciprocal: the table was measured on normal numbers only.
        /// </summary>
        /// <param name="w">The clip coordinate's w (positive for a vertex inside the clip volume).</param>
        /// <returns>The reciprocal as the reference GPU computes it.</returns>
        /// <exception cref="InvalidDataException">The embedded table is missing or damaged (thrown on first use).</exception>
        internal static float Of(float w) {
            float r = 1f / w;
            int wbits = BitConverter.SingleToInt32Bits(w);
            int exponent = (wbits >> 23) & 0xFF;
            // the table covers normal inputs with a normal, finite result
            if (exponent == 0 || exponent == 0xFF || !float.IsNormal(r))
                return r;
            int mantissa = wbits & 0x7FFFFF;
            int delta = ((Table.Value[mantissa >> 2] >> (2 * (mantissa & 3))) & 3) - 1;
            if (delta == 0)
                return r;
            // a difference in ulps is a difference of the bit patterns' magnitudes, the sign bit apart
            return BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(r) + delta);
        }

        /// <summary>Decodes the embedded table.</summary>
        /// <returns>The 2-bit codes of the 2^23 mantissas, four to a byte.</returns>
        /// <exception cref="InvalidDataException">The resource is missing, has another header, or decodes to another length.</exception>
        private static byte[] Load() {
            using Stream stream = typeof(GlReciprocal).Assembly.GetManifestResourceStream(ResourceName)
                                  ?? throw new InvalidDataException($"Gym.Environments embeds no '{ResourceName}'.");
            Span<byte> header = stackalloc byte[8];
            stream.ReadExactly(header);
            if (!header.SequenceEqual("GYMRCP1\0"u8))
                throw new InvalidDataException($"'{ResourceName}' does not start with GYMRCP1.");
            var table = new byte[(1 << 23) / 4];
            using var deflate = new DeflateStream(stream, CompressionMode.Decompress);
            deflate.ReadExactly(table);
            // nothing may follow: a longer stream is another table than the one measured
            if (deflate.ReadByte() != -1)
                throw new InvalidDataException($"'{ResourceName}' decodes to more than 2^23 entries.");
            return table;
        }
    }
}
