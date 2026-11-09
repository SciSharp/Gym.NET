#nullable enable
using System;
using System.Collections.Concurrent;
using System.IO;
using Gym.Rendering;

namespace Gym.Environments.Envs {
    /// <summary>
    ///     The images Gym.NET's envs render with, embedded in this assembly (DR-19; <c>Assets/README.md</c> lists their
    ///     sources), decoded once per process and shared.
    /// </summary>
    /// <remarks>
    ///     A loaded surface is shared by every caller, so it must only be read: draw on, scale or flip a copy (every
    ///     <see cref="Transform"/> function returns a new surface, which is how Gymnasium's envs use the images).
    /// </remarks>
    internal static class EnvAssets {
        /// <summary>The decoded images by resource name.</summary>
        private static readonly ConcurrentDictionary<string, Surface> Images = new(StringComparer.Ordinal);

        /// <summary><c>pygame.image.load(path.join(path.dirname(__file__), name))</c> for an embedded asset: the shared, read-only surface.</summary>
        /// <param name="name">The asset's path under <c>Assets</c>, with forward slashes (<c>classic_control/clockwise.png</c>).</param>
        /// <returns>The surface; never modify it.</returns>
        /// <exception cref="FileNotFoundException">Thrown when no asset has that name.</exception>
        /// <exception cref="NotSupportedException">Thrown when the asset is not an 8-bit RGB, RGBA or grey PNG.</exception>
        public static Surface Load(string name) => Images.GetOrAdd(name, Decode);

        /// <summary>Decodes one embedded PNG.</summary>
        /// <param name="name">The asset's path under <c>Assets</c>.</param>
        /// <returns>The surface.</returns>
        /// <exception cref="FileNotFoundException">Thrown when no asset has that name.</exception>
        /// <exception cref="NotSupportedException">Thrown when the asset is not an 8-bit RGB, RGBA or grey PNG.</exception>
        private static Surface Decode(string name) {
            using Stream stream = typeof(EnvAssets).Assembly.GetManifestResourceStream("assets/" + name)
                                  ?? throw new FileNotFoundException($"Gym.Environments embeds no asset '{name}'.", name);
            return ImageIO.Load(stream, name);
        }
    }
}
