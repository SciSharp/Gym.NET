using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;
using Gym.Envs;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Showreel;

/// <summary>
///     The docs site's cartridge gallery (the 8-bit "select your environment" screen under the video): for every recipe a
///     looping WebP at a fifth of the frame size, a poster PNG, and <c>envs.json</c> describing them, plus the site's logo
///     and icon in the video's chrome lettering.
/// </summary>
/// <remarks>
///     The loops are deliberately low resolution: the page scales them up with <c>image-rendering: pixelated</c>, which is
///     the 8-bit look the maintainer chose for this section, and it keeps all thirteen loops to a few megabytes.
/// </remarks>
internal static class Gallery {
    /// <summary>Frames per second of the loops.</summary>
    private const int Fps = 15;

    /// <summary>How much smaller than the env's frame a loop is (the "8-bit" pixel size).</summary>
    private const int Pixelate = 5;

    /// <summary>The longest loop, in seconds of play before the final hold.</summary>
    private const double MaxSeconds = 6.5;

    /// <summary>Seconds the final frame is held before the loop restarts.</summary>
    private const double HoldSeconds = 1.2;

    /// <summary>Writes every loop, poster and the manifest into a folder.</summary>
    /// <param name="folder">The output folder, e.g. <c>docs/website-src/images/envs</c>; created if missing.</param>
    /// <param name="ffmpeg">The ffmpeg executable.</param>
    /// <param name="log">Progress lines.</param>
    /// <exception cref="InvalidOperationException">ffmpeg failed.</exception>
    /// <exception cref="IOException">A file cannot be written.</exception>
    public static void Write(string folder, string ffmpeg, TextWriter log) {
        Directory.CreateDirectory(folder);
        var entries = new List<Dictionary<string, object>>();
        foreach (EpisodeRecipe recipe in Episodes.All) {
            var (ticks, steps, ret, stamp, wins, losses, action, observation) = Measure(recipe);
            int loopTicks = Math.Min(ticks, (int)(recipe.StepsPerSecond * MaxSeconds));
            int frames = (int)Math.Ceiling(loopTicks / recipe.StepsPerSecond * Fps) + (int)(HoldSeconds * Fps);
            using var player = new EpisodePlayer(recipe);
            RgbImage first = player.Frame;
            int w = Math.Max(48, (int)Math.Round(first.Width / (double)Pixelate)), h = Math.Max(32, (int)Math.Round(first.Height / (double)Pixelate));
            string loop = Path.Combine(folder, recipe.Id + ".webp"), poster = Path.Combine(folder, recipe.Id + ".png");
            using (var pipe = new FfmpegPipe(ffmpeg, first.Width, first.Height, Fps, FfmpegPipe.PixelWebP(w, h), loop)) {
                for (int f = 0; f < frames; f++) {
                    player.AdvanceTo(Math.Min(loopTicks, (int)Math.Floor(f * recipe.StepsPerSecond / Fps)));
                    pipe.Write(player.Frame);
                    if (f == frames * 2 / 5)
                        Downscale(player.Frame, w, h).SavePng(poster);
                }
                pipe.Finish();
            }
            log.WriteLine($"gallery: {recipe.Id}: {frames} frames at {w}x{h}, {new FileInfo(loop).Length / 1024} KB");
            entries.Add(new Dictionary<string, object> {
                ["id"] = recipe.Id,
                ["family"] = recipe.Family,
                ["blurb"] = recipe.Blurb,
                ["seed"] = recipe.Seed,
                ["steps"] = steps,
                ["return"] = Math.Round(ret, 1),
                ["result"] = recipe.Hands > 1 ? $"{wins} WON, {losses} LOST" : stamp ?? "",
                ["actionSpace"] = action,
                ["observationSpace"] = observation,
                ["make"] = $"Gymnasium.Make(\"{recipe.Id}\")",
                ["loop"] = $"images/envs/{recipe.Id}.webp",
                ["poster"] = $"images/envs/{recipe.Id}.png",
                ["width"] = w,
                ["height"] = h,
            });
        }
        // Relaxed escaping keeps the quotes in the make snippets readable; the file is fetched as JSON, never inlined in HTML.
        var json = new JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
        File.WriteAllText(Path.Combine(folder, "envs.json"), JsonSerializer.Serialize(entries, json) + "\n");
    }

    /// <summary>
    ///     Plays a recipe to its end without rendering, for the manifest: ticks to the end, the last hand's steps and
    ///     return, its stamp, the hands won and lost, and the spaces as Gymnasium prints them.
    /// </summary>
    /// <param name="recipe">The episode.</param>
    /// <returns>The measurements.</returns>
    private static (int Ticks, int Steps, double Return, string? Stamp, int Wins, int Losses, string Action, string Observation) Measure(EpisodeRecipe recipe) {
        using var player = new EpisodePlayer(recipe);
        // TimeLimit ends every id within 2,000 steps; the cap only guards against a recipe that never finishes.
        while (!player.Finished && player.Tick < 5_000)
            player.AdvanceTo(player.Tick + 1);
        using IEnv env = Gymnasium.Make(recipe.Id);
        return (player.Tick, player.Step, player.Return, player.Stamp, player.Wins, player.Losses, env.ActionSpace.ToString()!, env.ObservationSpace.ToString()!);
    }

    /// <summary>Shrinks a frame by averaging each output pixel's footprint (an area filter, like ffmpeg's).</summary>
    /// <param name="src">The frame.</param>
    /// <param name="w">Output width.</param>
    /// <param name="h">Output height.</param>
    /// <returns>The small image.</returns>
    public static RgbImage Downscale(RgbImage src, int w, int h) {
        var dst = new RgbImage(w, h);
        double sx = (double)src.Width / w, sy = (double)src.Height / h;
        for (int y = 0; y < h; y++) {
            int y0 = (int)(y * sy), y1 = Math.Max(y0 + 1, (int)((y + 1) * sy));
            for (int x = 0; x < w; x++) {
                int x0 = (int)(x * sx), x1 = Math.Max(x0 + 1, (int)((x + 1) * sx));
                double r = 0, g = 0, b = 0;
                int n = 0;
                for (int yy = y0; yy < y1 && yy < src.Height; yy++)
                    for (int xx = x0; xx < x1 && xx < src.Width; xx++) {
                        int o = (yy * src.Width + xx) * 3;
                        r += src.Pixels[o];
                        g += src.Pixels[o + 1];
                        b += src.Pixels[o + 2];
                        n++;
                    }
                int d = (y * w + x) * 3;
                dst.Pixels[d] = (byte)Math.Round(r / n);
                dst.Pixels[d + 1] = (byte)Math.Round(g / n);
                dst.Pixels[d + 2] = (byte)Math.Round(b / n);
            }
        }
        return dst;
    }

    /// <summary>
    ///     Writes the site's navbar logo and favicon: the chrome <c>GYM.NET</c> lettering and a chrome <c>G</c>, on
    ///     transparent backgrounds (the lettering has no anti-aliasing, so keying out the background colour is exact).
    /// </summary>
    /// <param name="logoPath">The logo PNG.</param>
    /// <param name="iconPath">The icon PNG.</param>
    /// <exception cref="IOException">A file cannot be written.</exception>
    public static void WriteLogo(string logoPath, string iconPath) {
        var logo = new RgbImage(232, 52);
        PixelFont.DrawChrome(logo, "GYM.NET", 116, 8, 5);
        SaveKeyed(logo, logoPath);
        var icon = new RgbImage(64, 64);
        PixelFont.DrawChrome(icon, "G", 32, 4, 7.4f);
        SaveKeyed(icon, iconPath);
    }

    /// <summary>Saves an image as RGBA PNG with pure black made transparent.</summary>
    /// <param name="img">The image (drawn on black).</param>
    /// <param name="path">The file.</param>
    private static void SaveKeyed(RgbImage img, string path) {
        using var png = new Image<Rgba32>(img.Width, img.Height);
        for (int y = 0; y < img.Height; y++)
            for (int x = 0; x < img.Width; x++) {
                int o = (y * img.Width + x) * 3;
                byte r = img.Pixels[o], g = img.Pixels[o + 1], b = img.Pixels[o + 2];
                png[x, y] = new Rgba32(r, g, b, (byte)(r == 0 && g == 0 && b == 0 ? 0 : 255));
            }
        png.SaveAsPng(path);
    }
}
