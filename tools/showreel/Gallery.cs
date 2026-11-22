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
///     looping WebP at a fifth of the frame size for its cartridge, the same moments at the env's full frame size and
///     twice the frame rate as an MP4 for the "now playing" screen, a poster PNG, and <c>envs.json</c> describing them,
///     plus the site's logo and icon in the video's chrome lettering.
/// </summary>
/// <remarks>
///     <para>
///     The cartridge loops are deliberately low resolution: the page scales them up with
///     <c>image-rendering: pixelated</c>, which is the 8-bit look the maintainer chose for the cartridges (DR-60), and it
///     keeps all 24 of them to about 5 MB, all loaded at once with the page.
///     </para>
///     <para>
///     The screen loops are the opposite (DR-101: "The gif is pixelated, it needs to be high quality (preview is okay to
///     be pixelated)"): every frame at the size the env renders it, H.264 at 30 fps. Measured on the 24 recipes, H.264 is
///     the smallest full-size form by far: 4.2 MB for all of them, against 13.4 MB as lossy animated WebP at the same
///     rate (7.7 MB at 15 fps) and 28.9 MB as lossless WebP at 15 fps (up to 4.9 MB for one MuJoCo loop). The page
///     downloads only the loop on screen.
///     </para>
/// </remarks>
internal static class Gallery {
    /// <summary>Frames per second of the cartridge loops.</summary>
    private const int Fps = 15;

    /// <summary>
    ///     Frames per second of the screen loops: exactly twice <see cref="Fps"/>, which is what lets one pass over the
    ///     episode feed both encoders (screen frame <c>2k</c> falls on the same tick as cartridge frame <c>k</c>). Changing
    ///     either rate breaks that pairing: <see cref="Write"/> would then hand the cartridge loop other ticks than before.
    /// </summary>
    private const int ScreenFps = Fps * 2;

    /// <summary>How much smaller than the env's frame a loop is (the "8-bit" pixel size).</summary>
    private const int Pixelate = 5;

    /// <summary>The longest loop, in seconds of play before the final hold.</summary>
    private const double MaxSeconds = 6.5;

    /// <summary>Seconds the final frame is held before the loop restarts.</summary>
    private const double HoldSeconds = 1.2;

    /// <summary>Writes every loop (cartridge and screen), poster and the manifest into a folder.</summary>
    /// <param name="folder">The output folder, e.g. <c>docs/website-src/images/envs</c>; created if missing.</param>
    /// <param name="ffmpeg">The ffmpeg executable.</param>
    /// <param name="log">Progress lines.</param>
    /// <exception cref="InvalidOperationException">ffmpeg failed (for instance an env whose frame has an odd side, which H.264's 4:2:0 refuses).</exception>
    /// <exception cref="IOException">A file cannot be written.</exception>
    /// <remarks>
    ///     Each recipe is played once, its frames sent to both encoders as they are rendered: the MuJoCo frames are the
    ///     costly part of the run, and a second pass for the screen loop would render each of them again.
    /// </remarks>
    public static void Write(string folder, string ffmpeg, TextWriter log) {
        Directory.CreateDirectory(folder);
        var entries = new List<Dictionary<string, object>>();
        foreach (EpisodeRecipe recipe in Episodes.All) {
            var (ticks, seed, steps, ret, stamp, wins, losses, action, observation) = Measure(recipe);
            int loopTicks = Math.Min(ticks, (int)(recipe.StepsPerSecond * MaxSeconds));
            int frames = (int)Math.Ceiling(loopTicks / recipe.StepsPerSecond * Fps) + (int)(HoldSeconds * Fps);
            int screenFrames = (int)Math.Ceiling(loopTicks / recipe.StepsPerSecond * ScreenFps) + (int)(HoldSeconds * ScreenFps);
            using var player = new EpisodePlayer(recipe);
            RgbImage first = player.Frame;
            int w = Math.Max(48, (int)Math.Round(first.Width / (double)Pixelate)), h = Math.Max(32, (int)Math.Round(first.Height / (double)Pixelate));
            string loop = Path.Combine(folder, recipe.Id + ".webp"), poster = Path.Combine(folder, recipe.Id + ".png");
            string screen = Path.Combine(folder, recipe.Id + ".mp4");
            // Both pipes are disposed on any failure, which kills an unfinished encoder instead of leaving it waiting on
            // its input; only Finish, reached once every frame is written, closes a file properly.
            using (var pipe = new FfmpegPipe(ffmpeg, first.Width, first.Height, Fps, FfmpegPipe.PixelWebP(w, h), loop))
            using (var screenPipe = new FfmpegPipe(ffmpeg, first.Width, first.Height, ScreenFps, FfmpegPipe.ScreenMp4(), screen)) {
                // Screen frame s shows tick floor(s * sps / 30). For s = 2k that is floor(2k * sps / 30), the very double
                // floor(k * sps / 15) gives (doubling is exact and both quotients round the same real number), so the
                // cartridge loop keeps exactly the frames it had before the screen loops existed. The last cartridge
                // frame falls on s = 2 * frames - 2, which ceil(2x) >= 2 * ceil(x) - 1 keeps below screenFrames; the Max
                // only keeps the cartridge loop whole should the rates or the hold ever change.
                for (int s = 0; s < Math.Max(screenFrames, 2 * frames - 1); s++) {
                    player.AdvanceTo(Math.Min(loopTicks, (int)Math.Floor(s * recipe.StepsPerSecond / ScreenFps)));
                    if (s < screenFrames)
                        screenPipe.Write(player.Frame);
                    if (s % 2 == 0 && s / 2 < frames) {
                        pipe.Write(player.Frame);
                        if (s / 2 == frames * 2 / 5)
                            Downscale(player.Frame, w, h).SavePng(poster);
                    }
                }
                pipe.Finish();
                screenPipe.Finish();
            }
            log.WriteLine($"gallery: {recipe.Id}: {frames} frames at {w}x{h}, {new FileInfo(loop).Length / 1024} KB; "
                + $"screen {screenFrames} frames at {first.Width}x{first.Height}, {new FileInfo(screen).Length / 1024} KB");
            entries.Add(new Dictionary<string, object> {
                ["id"] = recipe.Id,
                ["family"] = recipe.Family,
                ["blurb"] = recipe.Blurb,
                // The seed of the episode whose steps and return the card shows: the last hand's, for multi-hand recipes.
                ["seed"] = seed,
                ["steps"] = steps,
                ["return"] = Math.Round(ret, 1),
                ["result"] = recipe.ScoresHands ? $"{wins} WON, {losses} LOST" : stamp ?? "",
                ["actionSpace"] = action,
                ["observationSpace"] = observation,
                ["make"] = MakeSnippet(recipe),
                ["loop"] = $"images/envs/{recipe.Id}.webp",
                ["poster"] = $"images/envs/{recipe.Id}.png",
                // width and height are the cartridge loop's (the page sets them on its img); the screen loop is the env's
                // own frame size, which the page fits into the screen, so only its path is listed.
                ["width"] = w,
                ["height"] = h,
                ["screen"] = $"images/envs/{recipe.Id}.mp4",
            });
        }
        // Relaxed escaping keeps the quotes in the make snippets readable; the file is fetched as JSON, never inlined in HTML.
        var json = new JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
        File.WriteAllText(Path.Combine(folder, "envs.json"), JsonSerializer.Serialize(entries, json) + "\n");
    }

    /// <summary>
    ///     The C# line that makes the env as the gallery's loop was rendered: <c>render_mode</c> and the recipe's string
    ///     keyword arguments (the locomotion envs' <c>camera_name="track"</c>) in an anonymous object; a dictionary argument
    ///     (InvertedDoublePendulum's camera) is named in a trailing comment instead, since spelling it out would bury the call.
    /// </summary>
    /// <param name="recipe">The recipe.</param>
    /// <returns>The statement, e.g. <c>using IEnv env = Gymnasium.Make("Hopper-v5", new { render_mode = "rgb_array", camera_name = "track" });</c>.</returns>
    private static string MakeSnippet(EpisodeRecipe recipe) {
        var arguments = new List<string> { "render_mode = \"rgb_array\"" };
        var omitted = new List<string>();
        foreach (var (name, value) in recipe.Kwargs ?? new Dictionary<string, object?>()) {
            if (value is string text)
                arguments.Add($"{name} = \"{text}\"");
            else
                omitted.Add(name);
        }
        string comment = omitted.Count == 0 ? "" : $" // plus the showreel's {string.Join(", ", omitted)}";
        return $"using IEnv env = Gymnasium.Make(\"{recipe.Id}\", new {{ {string.Join(", ", arguments)} }});{comment}";
    }

    /// <summary>
    ///     Plays a recipe to its end without rendering, for the manifest: ticks to the end, the last hand's seed, steps and
    ///     return, its stamp, the hands won and lost, and the spaces as Gymnasium prints them.
    /// </summary>
    /// <param name="recipe">The episode.</param>
    /// <returns>The measurements.</returns>
    /// <remarks>
    ///     A multi-hand recipe (Reacher's four targets, Blackjack's five hands) resets each hand with the next seed, so the
    ///     last hand's seed is the recipe's plus the hand's index: the card names the episode its numbers come from.
    /// </remarks>
    private static (int Ticks, long Seed, int Steps, double Return, string? Stamp, int Wins, int Losses, string Action, string Observation) Measure(EpisodeRecipe recipe) {
        using var player = new EpisodePlayer(recipe);
        // TimeLimit ends every id within 2,000 steps; the cap only guards against a recipe that never finishes.
        while (!player.Finished && player.Tick < 5_000)
            player.AdvanceTo(player.Tick + 1);
        using IEnv env = Gymnasium.Make(recipe.Id);
        return (player.Tick, recipe.Seed + player.Hand, player.Step, player.Return, player.Stamp, player.Wins, player.Losses,
            env.ActionSpace.ToString()!, env.ObservationSpace.ToString()!);
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
