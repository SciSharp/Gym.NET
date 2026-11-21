using System.Collections.Generic;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Pusher-v4: refuses to be built, with Gymnasium's <c>ImportError</c>. Gymnasium supports this version only on MuJoCo
    ///     below 3 (Gymnasium issue #950), so on MuJoCo 3 and later its constructor raises before anything else, and Gym.NET's
    ///     engine follows MuJoCo 4.0.0 (DR-64). Pusher-v5 is the pusher to use.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The class is an env so that <c>Pusher-v4</c> registers and <c>make("Pusher-v4")</c> fails exactly as
    ///         Gymnasium's does, at construction (DR-61).
    ///     </para>
    ///     <para>
    ///         Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/pusher_v4.py#L14">gymnasium/envs/mujoco/pusher_v4.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("PusherEnv")]
    public partial class PusherEnvV4 : MujocoEnv {
        /// <summary>Gymnasium's message for Pusher-v4 on MuJoCo 3 and later.</summary>
        internal const string RefusalMessage =
            "`Pusher-v4` is only supported on `mujoco<3`, for more information https://github.com/Farama-Foundation/Gymnasium/issues/950";

        /// <summary>Always refuses (Gymnasium's <c>PusherEnv.__init__</c> on MuJoCo 3 and later), before any model is loaded.</summary>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>; never used.</param>
        /// <param name="width">The rendered frame's width; never used.</param>
        /// <param name="height">The rendered frame's height; never used.</param>
        /// <param name="cameraId">The camera id; never used.</param>
        /// <param name="cameraName">The camera name; never used.</param>
        /// <param name="maxGeom">The most geoms a frame draws; never used.</param>
        /// <param name="visualOptions">MuJoCo visual flags; never used.</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c>; never used.</param>
        /// <exception cref="ImportError">Always, with Gymnasium's message.</exception>
        public PusherEnvV4(string? renderMode = null, int width = DefaultSize, int height = DefaultSize, int? cameraId = null,
                           string? cameraName = null, int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                           EnvViewerFactory? viewerFactory = null)
            : base(Refuse(), 5, null, ClassMetadata, renderMode, width, height, cameraId, cameraName, null, maxGeom, visualOptions,
                   viewerFactory) { }

        /// <summary>
        ///     Builds the env with the render mode given as the <see cref="Gym.Envs.RenderMode"/> enum (DR-84, DR-93), as
        ///     <see cref="PusherEnvV4(string, int, int, int?, string, int, IReadOnlyDictionary{int,bool}, EnvViewerFactory)"/>
        ///     builds it from the mode's string: <c>new PusherEnvV4(RenderMode.RgbArray)</c> builds what
        ///     <c>new PusherEnvV4(renderMode: "rgb_array")</c> builds, the member reaching the env as Gymnasium's string
        ///     (<see cref="RenderModes.Name"/>), which its <c>RenderMode</c> property then holds.
        /// </summary>
        /// <remarks>
        ///     The mode is required and comes first, so <c>new PusherEnvV4()</c> keeps meaning the other constructor, without a
        ///     mode; leave the mode out there for no rendering. Every other parameter means what it means there.
        /// </remarks>
        /// <param name="renderMode">
        ///     The render mode, which the env must list in its <c>metadata["render_modes"]</c> as for the string; it behaves
        ///     exactly as its Gymnasium string does in the other constructor.
        /// </param>
        /// <param name="width">The rendered frame's width; never used.</param>
        /// <param name="height">The rendered frame's height; never used.</param>
        /// <param name="cameraId">The camera id; never used.</param>
        /// <param name="cameraName">The camera name; never used.</param>
        /// <param name="maxGeom">The most geoms a frame draws; never used.</param>
        /// <param name="visualOptions">MuJoCo visual flags; never used.</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c>; never used.</param>
        /// <exception cref="System.ArgumentOutOfRangeException">Thrown for a value <see cref="Gym.Envs.RenderMode"/> does not declare (a cast integer such as <c>(RenderMode)99</c>).</exception>
        /// <exception cref="ImportError">Always, with Gymnasium's message.</exception>
        public PusherEnvV4(RenderMode renderMode, int width = DefaultSize, int height = DefaultSize, int? cameraId = null, string? cameraName = null,
                           int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                           EnvViewerFactory? viewerFactory = null)
            : this(RenderModes.Name(renderMode), width, height, cameraId, cameraName, maxGeom, visualOptions, viewerFactory) { }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes at 20 frames per second. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(20);

        /// <summary>Never runs: no instance can exist.</summary>
        /// <param name="action">The action.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="ImportError">Always, with Gymnasium's message.</exception>
        public override StepResult<NDArray> Step(NDArray action) => throw new ImportError(RefusalMessage);

        /// <summary>Never runs: no instance can exist.</summary>
        /// <returns>Never returns.</returns>
        /// <exception cref="ImportError">Always, with Gymnasium's message.</exception>
        protected override NDArray ResetModel() => throw new ImportError(RefusalMessage);

        /// <summary>
        ///     Raises the refusal from the base constructor's first argument, so it fires before the model would load, as
        ///     Gymnasium's check precedes its <c>MujocoEnv.__init__</c>.
        /// </summary>
        /// <returns>Never returns.</returns>
        /// <exception cref="ImportError">Always, with Gymnasium's message.</exception>
        private static string Refuse() => throw new ImportError(RefusalMessage);
    }
}
