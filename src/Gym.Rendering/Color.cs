namespace Gym.Rendering {
    /// <summary>
    ///     An RGBA colour as pygame takes it: <c>(r, g, b)</c> is opaque (<paramref name="A"/> 255) and <c>(r, g, b, a)</c>
    ///     blends with weight <c>a</c>. Gymnasium passes 3-tuples, so its primitives draw opaque unless anti-aliasing weights
    ///     the edge pixels.
    /// </summary>
    /// <param name="R">Red, 0–255.</param>
    /// <param name="G">Green, 0–255.</param>
    /// <param name="B">Blue, 0–255.</param>
    /// <param name="A">Alpha, 0–255; 255 overwrites, anything lower blends with the surface (<c>d + (s - d) * a / 256</c>, floored).</param>
    public readonly record struct Color(byte R, byte G, byte B, byte A = 255) {
        /// <summary>
        ///     The colour packed as SDL_gfx's primitives take it, <c>0xRRGGBBAA</c> (<c>hlineRGBA</c> builds exactly this
        ///     from its four bytes).
        /// </summary>
        internal uint Packed => ((uint)R << 24) | ((uint)G << 16) | ((uint)B << 8) | A;

        /// <summary>
        ///     The pixel value on a 32-bit XRGB surface: <c>SDL_MapRGBA</c> for a format without an alpha mask, which drops
        ///     the alpha (<c>0x00RRGGBB</c>).
        /// </summary>
        internal uint Mapped => ((uint)R << 16) | ((uint)G << 8) | B;
    }
}
