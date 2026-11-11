#nullable enable
namespace Gym.Envs {
    /// <summary>
    ///     The render modes as constants, so <c>render_mode</c> values are discoverable and checked by the compiler instead of
    ///     typed as strings: <c>Gymnasium.Make("CartPole-v1", new { render_mode = RenderModes.RgbArray })</c>. Each constant is
    ///     Gymnasium's own string, so a constant and the literal are interchangeable everywhere (a spec, <c>metadata</c>,
    ///     <see cref="IEnv.RenderMode"/>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     An env renders only the modes its <c>metadata["render_modes"]</c> lists: CartPole lists <see cref="Human"/> and
    ///     <see cref="RgbArray"/>, the toy_text grids add <see cref="Ansi"/>, CarRacing adds <see cref="StatePixels"/>. The
    ///     <c>_list</c> modes are not an env's own: <c>make</c> wraps an env in <c>RenderCollection</c> for them, which
    ///     collects the base mode's frames between resets.
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
    }
}
