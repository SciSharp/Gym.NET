#nullable enable
using System;
using System.Collections.Generic;
using Gym.Utils;
using NumSharp;

namespace Gym.Envs {
    /// <summary>
    ///     The render modes as constants, so <c>render_mode</c> values are discoverable and checked by the compiler instead of
    ///     typed as strings: <c>Gymnasium.Make("CartPole-v1", new { render_mode = RenderModes.RgbArray })</c>. Each constant is
    ///     Gymnasium's own string, so a constant and the literal are interchangeable everywhere (a spec, <c>metadata</c>,
    ///     <see cref="IEnv.RenderMode"/>). <see cref="Name"/>, <see cref="Parse"/> and <see cref="TryParse(string)"/> convert
    ///     between these strings and the <see cref="RenderMode"/> enum (DR-84, DR-93).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     An env renders only the modes its <c>metadata["render_modes"]</c> lists: CartPole lists <see cref="Human"/> and
    ///     <see cref="RgbArray"/>, the toy_text grids add <see cref="Ansi"/>, CarRacing adds <see cref="StatePixels"/>. The
    ///     <c>_list</c> modes are not an env's own: <c>make</c> wraps an env in <c>RenderCollection</c> for them, which
    ///     collects the base mode's frames between resets.
    ///     </para>
    ///     <para>
    ///     The strings stay the canonical form (DR-93): an env's mode is any string Gymnasium accepts, including modes no
    ///     constant here names (a user's env's own, or <c>make</c>'s <c>"no-render-mode"</c> test), so
    ///     <see cref="TryParse(string)"/> answers <see langword="null"/> for those rather than failing.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L173">gymnasium/core.py (Env.render)</see>.</para>
    /// </remarks>
    public static class RenderModes {
        /// <summary><c>"human"</c>: the env draws into a window during <c>Step</c> and <c>Reset</c>, and <c>Render()</c> returns <see cref="RenderResult.None"/>.</summary>
        public const string Human = "human";

        /// <summary><c>"rgb_array"</c>: <c>Render()</c> returns the frame, an <c>(H, W, 3)</c> uint8 array (<see cref="RenderResult.Rgb"/>).</summary>
        public const string RgbArray = "rgb_array";

        /// <summary><c>"ansi"</c>: <c>Render()</c> returns the frame as text (<see cref="RenderResult.Ansi"/>), FrozenLake's, CliffWalking's and Taxi's grids.</summary>
        public const string Ansi = "ansi";

        /// <summary><c>"depth_array"</c>: <c>Render()</c> returns a depth image (<see cref="RenderResult.Depth"/>), the MuJoCo envs' mode.</summary>
        public const string DepthArray = "depth_array";

        /// <summary><c>"rgbd_tuple"</c>: <c>Render()</c> returns the frame and its depth image together (<see cref="RenderResult.RgbDepth"/>), the MuJoCo envs' mode.</summary>
        public const string RgbdTuple = "rgbd_tuple";

        /// <summary><c>"state_pixels"</c>: CarRacing's 96×96 frame, the image its observations are, rather than the 600×400 <see cref="RgbArray"/> view.</summary>
        public const string StatePixels = "state_pixels";

        /// <summary><c>"rgb_array_list"</c>: every <see cref="RgbArray"/> frame since the last <c>Render()</c> or reset, collected by <c>RenderCollection</c> (<see cref="RenderResult.Frames"/>).</summary>
        public const string RgbArrayList = "rgb_array_list";

        /// <summary><c>"ansi_list"</c>: every <see cref="Ansi"/> frame since the last <c>Render()</c> or reset, collected by <c>RenderCollection</c>.</summary>
        public const string AnsiList = "ansi_list";

        /// <summary><c>"depth_array_list"</c>: every <see cref="DepthArray"/> frame since the last <c>Render()</c> or reset, collected by <c>RenderCollection</c>.</summary>
        public const string DepthArrayList = "depth_array_list";

        /// <summary><c>"rgbd_tuple_list"</c>: every <see cref="RgbdTuple"/> frame since the last <c>Render()</c> or reset, collected by <c>RenderCollection</c>.</summary>
        public const string RgbdTupleList = "rgbd_tuple_list";

        /// <summary><c>"state_pixels_list"</c>: every <see cref="StatePixels"/> frame since the last <c>Render()</c> or reset, collected by <c>RenderCollection</c>.</summary>
        public const string StatePixelsList = "state_pixels_list";

        /// <summary>Every <see cref="RenderMode"/> member, in the order the enum declares them (the order of the constants above).</summary>
        /// <remarks>A shared, read-only list.</remarks>
        public static IReadOnlyList<RenderMode> All => PyEnums.Members<RenderMode>();

        /// <summary>
        ///     Gymnasium's string for a member: <c>RenderMode.RgbArray</c> is <c>"rgb_array"</c>. The env constructors that take
        ///     the enum pass it on through this, so both spellings build the same env.
        /// </summary>
        /// <param name="mode">The mode.</param>
        /// <returns>The string, one of this class's constants.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="RenderMode"/> does not declare (<c>(RenderMode)99</c>).</exception>
        public static string Name(RenderMode mode) => PyEnums.Value(mode);

        /// <summary>
        ///     The member for Gymnasium's string: <c>"rgb_array"</c> is <c>RenderMode.RgbArray</c>. Only Gymnasium's strings are
        ///     accepted, not the members' C# names (<c>"RgbArray"</c>), as Python's <c>Enum(value)</c> accepts values only.
        /// </summary>
        /// <param name="name">The string.</param>
        /// <returns>The member.</returns>
        /// <exception cref="ValueError">
        ///     Thrown for a string no member stands for (<c>'no-render-mode' is not a valid RenderMode</c>), which Gymnasium
        ///     may still accept as a render mode: use <see cref="TryParse(string)"/> on an env's own mode.
        /// </exception>
        public static RenderMode Parse(string? name) => PyEnums.Of<RenderMode>(name);

        /// <summary>
        ///     The member for Gymnasium's string, or <see langword="null"/> when no member stands for it: a custom mode, or no
        ///     mode at all. <c>RenderModes.TryParse(env.RenderMode) == RenderMode.RgbArray</c> compares an env's mode with a
        ///     member without failing on modes the enum does not list.
        /// </summary>
        /// <param name="name">The string, typically <see cref="IEnv.RenderMode"/>; <see langword="null"/> gives <see langword="null"/>.</param>
        /// <returns>The member, or <see langword="null"/>.</returns>
        public static RenderMode? TryParse(string? name) => PyEnums.TryOf(name, out RenderMode mode) ? mode : null;

        /// <summary>The .NET <c>TryParse</c> pattern of <see cref="TryParse(string)"/>: the member for Gymnasium's string, if one stands for it.</summary>
        /// <param name="name">The string; <see langword="null"/> matches no member.</param>
        /// <param name="mode">The member, when <see langword="true"/> is returned; <see cref="RenderMode.Human"/> (the enum's default) otherwise.</param>
        /// <returns><see langword="true"/> when a member stands for <paramref name="name"/>.</returns>
        public static bool TryParse(string? name, out RenderMode mode) => PyEnums.TryOf(name, out mode);
    }
}
