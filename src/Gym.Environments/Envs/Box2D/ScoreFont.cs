/*
  SDL_ttf:  A companion library to SDL for working with TrueType (tm) fonts
  Copyright (C) 2001-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
// Altered source version (Gym.NET, 2026-11-11): the single-line path of SDL_ttf 2.24.0's TTF_RenderUTF8_Shaded_Wrapped,
// translated to C# for recorded glyph data: TTF_Size_Internal's HarfBuzz layout (26.6 positions floored per glyph, the
// bounding box, the start offsets), TTF_Render_Wrapped_Internal's row height (the larger of the text's height and the
// line skip), and Render_Line's shaded blit (each glyph's coverage OR-ed into an 8-bit buffer, clipped to it).
#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text.Json;

namespace Gym.Environments.Envs.Box2D {
    /// <summary>
    ///     The font CarRacing draws its score with (DR-27): pygame's default font at size 42, as recorded glyph data (each
    ///     glyph's 8-bit coverage, bearings and advance, the pairs' kerning) laid out the way SDL_ttf 2.24 lays out shaded text,
    ///     so the score's pixels are pygame's for every string of <c>0</c>–<c>9</c> and <c>-</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The layout is SDL_ttf's HarfBuzz path: glyph <c>j</c> sits at <c>floor(Σ (advance + kern) / 64)</c> pixels (26.6 fixed
    ///     point, accumulated without rounding), its box offset by its bearings below the ascent; the surface spans the boxes
    ///     and the final pen position, and is as tall as the larger of their extent and the line skip, as pygame-ce renders
    ///     through the wrapped variant. Overlapping glyphs combine by bitwise OR, as SDL_ttf's shaded blit does. The data's
    ///     advances are FreeType's hinted ones and its kerning HarfBuzz's scaling of the font's kern table.
    ///     </para>
    ///     <para>
    ///     The data is recorded from live pygame (<c>tools/parity/record_score_font.py</c>) and is not shipped yet: the font
    ///     is GNU FreeFont's FreeSans Bold of 2002, whose embedded licence is the GPL, and whether its rasterized glyphs may ship
    ///     in Gym.NET is the maintainer's decision (DR-27's licence check). Until then <see cref="Current"/> is
    ///     <see langword="null"/> unless a caller installs data, and CarRacing leaves the score out. Thread-safe once built:
    ///     rendering only reads.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/libsdl-org/SDL_ttf/blob/release-2.24.0/SDL_ttf.c#L3156">SDL_ttf.c</see> (TTF_Size_Internal, Render_Line, TTF_Render_Wrapped_Internal).</para>
    /// </remarks>
    internal sealed class ScoreFont {
        /// <summary>The embedded resource the recorded data ships as, once it may ship.</summary>
        internal const string ResourceName = "assets/fonts/freesansbold-42.json";

        /// <summary>The glyphs by character.</summary>
        private readonly Dictionary<char, Glyph> _glyphs;

        /// <summary>The kerning of each kerned pair, in 26.6 pixels.</summary>
        private readonly Dictionary<(char Left, char Right), int> _kerning;

        /// <summary>The data a caller installed, or the embedded data, or <see langword="null"/>; see <see cref="Current"/>.</summary>
        private static ScoreFont? _current = LoadEmbedded();

        /// <summary>A font from its parts.</summary>
        /// <param name="glyphs">The glyphs by character.</param>
        /// <param name="kerning">The kerned pairs' adjustments in 26.6 pixels.</param>
        /// <param name="ascent">SDL_ttf's ascent in pixels (the baseline's row).</param>
        /// <param name="height">SDL_ttf's font height in pixels.</param>
        /// <param name="lineSkip">SDL_ttf's line skip in pixels.</param>
        private ScoreFont(Dictionary<char, Glyph> glyphs, Dictionary<(char Left, char Right), int> kerning, int ascent, int height, int lineSkip) {
            _glyphs = glyphs;
            _kerning = kerning;
            Ascent = ascent;
            Height = height;
            LineSkip = lineSkip;
        }

        /// <summary>
        ///     The font CarRacing uses: data a caller installed (the parity suite installs data it records from live pygame),
        ///     else the embedded resource, else <see langword="null"/>, and then the score is not drawn.
        /// </summary>
        /// <remarks>Footgun: process-wide; installing data changes every CarRacing env's frames from their next render.</remarks>
        internal static ScoreFont? Current {
            get => _current;
            set => _current = value;
        }

        /// <summary>The baseline's row from the top, in pixels (34).</summary>
        internal int Ascent { get; }

        /// <summary>The font's height in pixels (42), the least height of a laid-out line.</summary>
        internal int Height { get; }

        /// <summary>The line skip in pixels (46), the rendered surface's least height.</summary>
        internal int LineSkip { get; }

        /// <summary>
        ///     Lays a text out and draws it as SDL_ttf's shaded rendering does: the coverage levels 0 (background) to 255
        ///     (foreground) of a <c>width × height</c> surface, row by row.
        /// </summary>
        /// <param name="text">The text, of characters the data has (<c>0</c>–<c>9</c> and <c>-</c>).</param>
        /// <returns>The levels (row-major), the width and the height.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="text"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for an empty text (SDL_ttf's "Text has zero width") or a character without a glyph.</exception>
        internal (byte[] Levels, int Width, int Height) RenderShaded(string text) {
            if (text is null)
                throw new ArgumentNullException(nameof(text));
            if (text.Length == 0)
                throw new ArgumentException("Text has zero width", nameof(text));

            // TTF_Size_Internal, the HarfBuzz path: pen positions in 26.6, each glyph's box from its floored position.
            var placed = new (int PosX, int PosY, Glyph Glyph)[text.Length];
            int x = 0;
            int minx = 0, maxx = 0, miny = 0, maxy = Height;
            int posY = Ascent * 64; // F26Dot6(font->ascent); y_advance and y_offset are 0 for a horizontal line
            for (int i = 0; i < text.Length; i++) {
                if (!_glyphs.TryGetValue(text[i], out Glyph? glyph))
                    throw new ArgumentException($"The score font has no glyph for '{text[i]}'.", nameof(text));
                // HarfBuzz splits a pair's kerning between the two glyphs' advances and the second's offset; the positions are
                // those of the whole kerning added to the first glyph's advance.
                int kern = i + 1 < text.Length && _kerning.TryGetValue((text[i], text[i + 1]), out int k) ? k : 0;
                placed[i] = (x, posY, glyph);
                x += glyph.Advance + kern;

                int boxX = FtFloor(placed[i].PosX) + glyph.Left;
                int boxY = FtFloor(posY) - glyph.Top;
                minx = Math.Min(minx, boxX);
                maxx = Math.Max(maxx, boxX + glyph.Width);
                miny = Math.Min(miny, boxY);
                maxy = Math.Max(maxy, boxY + glyph.Rows);
            }

            // Allows to render a string with only one space (bug 4344): the final pen position counts too.
            maxx = Math.Max(maxx, FtFloor(x));
            int xstart = minx < 0 ? -minx : 0;
            int ystart = miny < 0 ? -miny : 0;
            int width = maxx - minx;
            int lineHeight = maxy - miny;
            if (width == 0)
                throw new ArgumentException("Text has zero width", nameof(text));

            // TTF_Render_Wrapped_Internal with wrapLength 0 and one line: the row is at least the line skip tall.
            int height = Math.Max(lineHeight, LineSkip);
            var levels = new byte[width * height];

            // Render_Line, shaded: each glyph's coverage OR-ed in at its floored position, clipped to the surface.
            foreach ((int px, int py, Glyph glyph) in placed) {
                int gx = xstart + FtFloor(px) + glyph.Left;
                int gy = ystart + FtFloor(py) - glyph.Top;
                for (int row = 0; row < glyph.Rows; row++) {
                    int ty = gy + row;
                    if (ty < 0 || ty >= height)
                        continue;
                    for (int column = 0; column < glyph.Width; column++) {
                        int tx = gx + column;
                        if (tx < 0 || tx >= width)
                            continue;
                        levels[ty * width + tx] |= glyph.Coverage[row * glyph.Width + column];
                    }
                }
            }

            return (levels, width, height);
        }

        /// <summary>
        ///     Reads recorded data (the JSON <c>tools/parity/record_score_font.py</c> writes): the metrics, each glyph's
        ///     <c>left</c>, <c>top</c>, <c>width</c>, <c>rows</c>, 26.6 <c>advance</c> and base64 <c>coverage</c>, and the
        ///     kerning of character pairs in 26.6.
        /// </summary>
        /// <param name="json">The JSON text.</param>
        /// <returns>The font.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="json"/> is <see langword="null"/>.</exception>
        /// <exception cref="JsonException">Thrown for text that is not JSON.</exception>
        /// <exception cref="InvalidDataException">Thrown for JSON missing a field, or a glyph whose coverage does not fill its box.</exception>
        internal static ScoreFont Parse(string json) {
            if (json is null)
                throw new ArgumentNullException(nameof(json));
            using JsonDocument document = JsonDocument.Parse(json);
            JsonElement root = document.RootElement;
            try {
                var glyphs = new Dictionary<char, Glyph>();
                foreach (JsonProperty entry in root.GetProperty("glyphs").EnumerateObject()) {
                    JsonElement g = entry.Value;
                    byte[] coverage = Convert.FromBase64String(g.GetProperty("coverage").GetString()!);
                    var glyph = new Glyph(g.GetProperty("left").GetInt32(), g.GetProperty("top").GetInt32(), g.GetProperty("width").GetInt32(),
                                          g.GetProperty("rows").GetInt32(), g.GetProperty("advance").GetInt32(), coverage);
                    if (coverage.Length != glyph.Width * glyph.Rows)
                        throw new InvalidDataException($"Glyph '{entry.Name}' has {coverage.Length} coverage bytes for a {glyph.Width}x{glyph.Rows} box.");
                    glyphs[entry.Name[0]] = glyph;
                }

                var kerning = new Dictionary<(char Left, char Right), int>();
                foreach (JsonProperty pair in root.GetProperty("kerning").EnumerateObject())
                    kerning[(pair.Name[0], pair.Name[1])] = pair.Value.GetInt32();
                return new ScoreFont(glyphs, kerning, root.GetProperty("ascent").GetInt32(), root.GetProperty("height").GetInt32(),
                                     root.GetProperty("lineskip").GetInt32());
            } catch (KeyNotFoundException e) {
                throw new InvalidDataException("The score font's data lacks a field: " + e.Message, e);
            } catch (FormatException e) {
                throw new InvalidDataException("The score font's coverage is not base64: " + e.Message, e);
            }
        }

        /// <summary>The embedded data, or <see langword="null"/> when the resource is not shipped.</summary>
        /// <returns>The font, or <see langword="null"/>.</returns>
        private static ScoreFont? LoadEmbedded() {
            Assembly assembly = typeof(ScoreFont).Assembly;
            using Stream? stream = assembly.GetManifestResourceStream(ResourceName);
            if (stream is null)
                return null;
            using var reader = new StreamReader(stream);
            return Parse(reader.ReadToEnd());
        }

        /// <summary>FreeType's <c>FT_FLOOR</c> of a 26.6 value: the whole pixels, rounded down (toward negative infinity).</summary>
        /// <param name="value">The 26.6 value.</param>
        /// <returns>The pixels.</returns>
        private static int FtFloor(int value) => (value & -64) / 64;

        /// <summary>One glyph's recorded data.</summary>
        /// <param name="Left">The bitmap's left bearing in pixels (SDL_ttf's <c>sz_left</c>).</param>
        /// <param name="Top">The bitmap's top above the baseline in pixels (<c>sz_top</c>).</param>
        /// <param name="Width">The bitmap's width in pixels.</param>
        /// <param name="Rows">The bitmap's height in pixels.</param>
        /// <param name="Advance">The hinted advance in 26.6 pixels.</param>
        /// <param name="Coverage">The 8-bit coverage, row-major, <c>Width × Rows</c> bytes.</param>
        internal sealed record Glyph(int Left, int Top, int Width, int Rows, int Advance, byte[] Coverage);
    }
}
