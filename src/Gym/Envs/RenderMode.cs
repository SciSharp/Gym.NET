#nullable enable
using Gym.Utils;

namespace Gym.Envs {
    /// <summary>
    ///     The render modes Gymnasium's envs know, as an enum (DR-84, DR-93): what the env constructors, <c>Make</c> and
    ///     <c>MakeVec</c> take beside Gymnasium's string, <c>new CartPoleEnv(RenderMode.RgbArray)</c> being
    ///     <c>new CartPoleEnv(renderMode: "rgb_array")</c>. Each member stands for its Gymnasium string
    ///     (<see cref="RenderModes.Name"/>), and <see cref="RenderModes.Parse"/> and <see cref="RenderModes.TryParse(string)"/>
    ///     read a string back.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The string stays the canonical form (DR-93): <see cref="IEnv.RenderMode"/>, an env's <c>metadata["render_modes"]</c>,
    ///     a spec's kwargs and the attribute protocol hold Gymnasium's strings, because Gymnasium accepts render modes no enum
    ///     can list: <c>make("CartPole-v1", render_mode="no-render-mode")</c> warns and builds the env, and a user's env may
    ///     declare modes of its own. To compare an env's mode with a member, parse it:
    ///     <c>RenderModes.TryParse(env.RenderMode) == RenderMode.RgbArray</c>, or compare the strings
    ///     (<c>env.RenderMode == RenderModes.RgbArray</c>).
    ///     </para>
    ///     <para>
    ///     Footgun: inside a class derived from <c>Env</c>, the simple name <c>RenderMode</c> in an expression is the
    ///     inherited <c>RenderMode</c> property (a string), so <c>RenderMode.Human</c> does not compile there; write
    ///     <c>Gym.Envs.RenderMode.Human</c>. As a parameter or variable type the name is this enum everywhere.
    ///     </para>
    ///     <para>
    ///     Passed in a kwargs dictionary or anonymous object (<c>new { render_mode = RenderMode.Human }</c>), a member becomes
    ///     its string (<see cref="PyEnums.ToPython"/>), so specs record what Gymnasium's do.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L173">gymnasium/core.py (Env.render)</see>.</para>
    /// </remarks>
    [PyEnum]
    public enum RenderMode {
        /// <summary><c>"human"</c>: the env draws into a window during <c>Step</c> and <c>Reset</c>, and <c>Render()</c> returns <see cref="RenderResult.None"/>.</summary>
        [PyValue(RenderModes.Human)] Human,

        /// <summary><c>"rgb_array"</c>: <c>Render()</c> returns the frame, an <c>(H, W, 3)</c> uint8 array (<see cref="RenderResult.Rgb"/>).</summary>
        [PyValue(RenderModes.RgbArray)] RgbArray,

        /// <summary><c>"ansi"</c>: <c>Render()</c> returns the frame as text (<see cref="RenderResult.Ansi"/>), the toy_text grids' mode.</summary>
        [PyValue(RenderModes.Ansi)] Ansi,

        /// <summary><c>"depth_array"</c>: <c>Render()</c> returns a depth image (<see cref="RenderResult.Depth"/>), the MuJoCo envs' mode.</summary>
        [PyValue(RenderModes.DepthArray)] DepthArray,

        /// <summary><c>"rgbd_tuple"</c>: <c>Render()</c> returns the frame and its depth image together (<see cref="RenderResult.RgbDepth"/>), the MuJoCo envs' mode.</summary>
        [PyValue(RenderModes.RgbdTuple)] RgbdTuple,

        /// <summary><c>"state_pixels"</c>: CarRacing's 96×96 frame, the image its observations are.</summary>
        [PyValue(RenderModes.StatePixels)] StatePixels,

        /// <summary><c>"rgb_array_list"</c>: every <see cref="RgbArray"/> frame since the last <c>Render()</c> or reset, which <c>make</c> collects with <c>RenderCollection</c>.</summary>
        [PyValue(RenderModes.RgbArrayList)] RgbArrayList,

        /// <summary><c>"ansi_list"</c>: every <see cref="Ansi"/> frame since the last <c>Render()</c> or reset, collected by <c>RenderCollection</c>.</summary>
        [PyValue(RenderModes.AnsiList)] AnsiList,

        /// <summary><c>"depth_array_list"</c>: every <see cref="DepthArray"/> frame since the last <c>Render()</c> or reset, collected by <c>RenderCollection</c>.</summary>
        [PyValue(RenderModes.DepthArrayList)] DepthArrayList,

        /// <summary><c>"rgbd_tuple_list"</c>: every <see cref="RgbdTuple"/> frame since the last <c>Render()</c> or reset, collected by <c>RenderCollection</c>.</summary>
        [PyValue(RenderModes.RgbdTupleList)] RgbdTupleList,

        /// <summary><c>"state_pixels_list"</c>: every <see cref="StatePixels"/> frame since the last <c>Render()</c> or reset, collected by <c>RenderCollection</c>.</summary>
        [PyValue(RenderModes.StatePixelsList)] StatePixelsList,
    }
}
