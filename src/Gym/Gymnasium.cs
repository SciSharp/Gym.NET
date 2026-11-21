#nullable enable
using System;
using System.Collections.Generic;
using Gym.Environments;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Utils;
using Gym.Vector;
using NumSharp;

namespace Gym {
    /// <summary>
    ///     Gymnasium's top-level module, <c>import gymnasium as gym</c>: <see cref="Make(string, object, long?, bool?)"/>,
    ///     <see cref="Spec"/>, <see cref="Register"/>, <see cref="Registry"/>, <see cref="PprintRegistry"/> and
    ///     <see cref="RegisterEnvs"/>, re-exported from <see cref="EnvRegistry"/> as Python re-exports them from
    ///     <c>gymnasium.envs.registration</c>, plus the app-wide human-mode viewer (DR-20).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The built-in envs register their ids before <c>Main</c> runs: the <c>Gym.NET.Environments</c> package compiles a
    ///     module initializer into a C# app that calls its <c>Register()</c> (DR-17). F# and Visual Basic apps call
    ///     <c>GymEnvironments.Register()</c> themselves; setting the environment variable
    ///     <c>GYMNET_DISABLE_ENV_REGISTRATION</c> turns the automatic registration off (DR-14).
    ///     </para>
    ///     <para>
    ///     In Gym.NET's own unit tests the namespace <c>Gym.Tests.Gymnasium</c> hides this class, so they import it with
    ///     <c>global using static Gym.Gymnasium;</c> and call <c>Make("CartPole-v1")</c>, as <c>from gymnasium import make</c>
    ///     reads.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/__init__.py">gymnasium/__init__.py</see>.</para>
    /// </remarks>
    public static partial class Gymnasium {
        /// <summary>
        ///     The viewer every env renders <c>render_mode="human"</c> into unless it was given its own <c>viewer_factory</c>
        ///     (DR-20): set it once at start-up (<c>Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory;</c>). Left
        ///     <see langword="null"/>, a human render raises <see cref="DependencyNotInstalled"/>, where Gymnasium raises it for
        ///     a missing pygame; nothing falls back to a headless viewer silently.
        /// </summary>
        /// <remarks>Read when an env first renders in human mode, so setting it after an env was built still applies to that env.</remarks>
        public static EnvViewerFactory? HumanViewerFactory { get; set; }

        /// <summary>The global registry (<c>gymnasium.registry</c>): every id <see cref="Make(string, object, long?, bool?)"/> knows.</summary>
        public static EnvSpecRegistry Registry => EnvRegistry.Registry;

        /// <summary>
        ///     The viewer factory an env uses for <c>human</c> mode (DR-20): its own <paramref name="viewerFactory"/>, else
        ///     <see cref="HumanViewerFactory"/>, else Gymnasium's missing-dependency error. Envs call it on their first human
        ///     render.
        /// </summary>
        /// <param name="viewerFactory">The env's <c>viewer_factory</c> argument, or <see langword="null"/>.</param>
        /// <returns>The factory to create the window with.</returns>
        /// <exception cref="DependencyNotInstalled">
        ///     Thrown when neither is set: <c>No human-mode viewer is set, set `Gymnasium.HumanViewerFactory` (for example to
        ///     `AvaloniaEnvViewer.Factory` of the Gym.NET.Rendering.Avalonia package) or pass `viewer_factory` to the
        ///     environment</c>.
        /// </exception>
        public static EnvViewerFactory ResolveHumanViewerFactory(EnvViewerFactory? viewerFactory) =>
            viewerFactory ?? HumanViewerFactory
            ?? throw new DependencyNotInstalled("No human-mode viewer is set, set `Gymnasium.HumanViewerFactory` (for example to `AvaloniaEnvViewer.Factory` of the Gym.NET.Rendering.Avalonia package) or pass `viewer_factory` to the environment");

        /// <summary>Builds a registered env with <c>make</c>'s wrappers (<c>gymnasium.make</c>); see <see cref="EnvRegistry.Make(string, object, long?, bool?)"/>.</summary>
        /// <param name="id">The id: <c>"CartPole-v1"</c>, an unversioned <c>"CartPole"</c>, or <c>"Assembly:Id"</c>.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name: a dictionary, an options record, or an anonymous object.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the env cannot be built from the kwargs.</exception>
        /// <exception cref="Error">Thrown when the id is not registered or has no entry point.</exception>
        /// <exception cref="NamespaceNotFound">Thrown when the id's namespace is unknown.</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown.</exception>
        /// <exception cref="VersionNotFound">Thrown when the version is newer than any registered.</exception>
        /// <exception cref="DeprecatedEnv">Thrown when the version is older than the latest.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a module named by the id or the entry point does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static IEnv Make(string id, object? kwargs = null, long? maxEpisodeSteps = null, bool? disableEnvChecker = null) =>
            EnvRegistry.Make(id, kwargs, maxEpisodeSteps, disableEnvChecker);

        /// <summary>Builds the env a spec describes (<c>gymnasium.make(env_spec)</c>); see <see cref="EnvRegistry.Make(EnvSpec, object, long?, bool?)"/>.</summary>
        /// <param name="spec">The spec.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name, over the spec's.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the env cannot be built from the kwargs.</exception>
        /// <exception cref="Error">Thrown when the spec has no entry point.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a string entry point names a module that does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static IEnv Make(EnvSpec spec, object? kwargs = null, long? maxEpisodeSteps = null, bool? disableEnvChecker = null) =>
            EnvRegistry.Make(spec, kwargs, maxEpisodeSteps, disableEnvChecker);

        /// <summary>Builds a registered env typed (DR-10); see <see cref="EnvRegistry.Make{TObs, TAct}(string, object, long?, bool?)"/>.</summary>
        /// <typeparam name="TObs">The observation type the env must have.</typeparam>
        /// <typeparam name="TAct">The action type the env must have.</typeparam>
        /// <param name="id">The id.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name, over the spec's.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the env is not an <c>Env&lt;TObs, TAct&gt;</c> or cannot be built from the kwargs.</exception>
        /// <exception cref="Error">Thrown when the id is not registered or has no entry point.</exception>
        /// <exception cref="NamespaceNotFound">Thrown when the id's namespace is unknown.</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown.</exception>
        /// <exception cref="VersionNotFound">Thrown when the version is newer than any registered.</exception>
        /// <exception cref="DeprecatedEnv">Thrown when the version is older than the latest.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a module named by the id or the entry point does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static Env<TObs, TAct> Make<TObs, TAct>(string id, object? kwargs = null, long? maxEpisodeSteps = null, bool? disableEnvChecker = null) =>
            EnvRegistry.Make<TObs, TAct>(id, kwargs, maxEpisodeSteps, disableEnvChecker);

        /// <summary>Builds the env a spec describes, typed (DR-10); see <see cref="EnvRegistry.Make{TObs, TAct}(EnvSpec, object, long?, bool?)"/>.</summary>
        /// <typeparam name="TObs">The observation type the env must have.</typeparam>
        /// <typeparam name="TAct">The action type the env must have.</typeparam>
        /// <param name="spec">The spec.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name, over the spec's.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the env is not an <c>Env&lt;TObs, TAct&gt;</c> or cannot be built from the kwargs.</exception>
        /// <exception cref="Error">Thrown when the spec has no entry point.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a string entry point names a module that does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static Env<TObs, TAct> Make<TObs, TAct>(EnvSpec spec, object? kwargs = null, long? maxEpisodeSteps = null, bool? disableEnvChecker = null) =>
            EnvRegistry.Make<TObs, TAct>(spec, kwargs, maxEpisodeSteps, disableEnvChecker);

        /// <summary>
        ///     Builds a vector env of copies of a registered env (<c>gymnasium.make_vec</c>): its own vectorized class when the
        ///     spec has one, else a <c>SyncVectorEnv</c>; see <see cref="EnvRegistry.MakeVec(string, int, object, object, IEnumerable{Func{IEnv, IEnv}}, object)"/>.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorizationMode">A <see cref="VectorizeMode"/> or its value (<c>"sync"</c>, <c>"async"</c>, <c>"vector_entry_point"</c>), or <see langword="null"/> for the spec's default.</param>
        /// <param name="vectorKwargs">The vector env's constructor arguments by Python name (<c>autoreset_mode</c>, <c>copy</c>, <c>observation_mode</c>).</param>
        /// <param name="wrappers">Wrappers applied to every sub-env, innermost first.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <returns>The vector env, untyped; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the env cannot be built from the kwargs.</exception>
        /// <exception cref="ValueError">Thrown for an unknown vectorization mode.</exception>
        /// <exception cref="Error">Thrown when the id is not registered, has no (vector) entry point, or the vector entry point is asked for with vector kwargs or wrappers.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static IVectorEnv MakeVec(string id, int numEnvs = 1, object? vectorizationMode = null, object? vectorKwargs = null, IEnumerable<Func<IEnv, IEnv>>? wrappers = null,
            object? kwargs = null) =>
            EnvRegistry.MakeVec(id, numEnvs, vectorizationMode, vectorKwargs, wrappers, kwargs);

        /// <summary>Builds a vector env from a spec (<c>gymnasium.make_vec(env_spec)</c>); see <see cref="EnvRegistry.MakeVec(EnvSpec, int, object, object, IEnumerable{Func{IEnv, IEnv}}, object)"/>.</summary>
        /// <param name="spec">The spec: a registered one, or one a previous <c>MakeVec</c> recorded.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorizationMode">The vectorization mode, or <see langword="null"/>.</param>
        /// <param name="vectorKwargs">The vector env's constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <returns>The vector env, untyped; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the env cannot be built.</exception>
        /// <exception cref="ValueError">Thrown for an unknown vectorization mode.</exception>
        /// <exception cref="Error">Thrown when the spec has no (vector) entry point.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static IVectorEnv MakeVec(EnvSpec spec, int numEnvs = 1, object? vectorizationMode = null, object? vectorKwargs = null, IEnumerable<Func<IEnv, IEnv>>? wrappers = null,
            object? kwargs = null) =>
            EnvRegistry.MakeVec(spec, numEnvs, vectorizationMode, vectorKwargs, wrappers, kwargs);

        /// <summary>Builds a vector env typed by its batched types (DR-21); see <see cref="EnvRegistry.MakeVec{TObs, TAct}(string, int, object, object, IEnumerable{Func{IEnv, IEnv}}, object)"/>.</summary>
        /// <typeparam name="TObs">The batched observation type (<c>NDArray</c> for array spaces).</typeparam>
        /// <typeparam name="TAct">The batched action type.</typeparam>
        /// <param name="id">The id.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorizationMode">The vectorization mode, or <see langword="null"/>.</param>
        /// <param name="vectorKwargs">The vector env's constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <returns>The vector env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the batched types are not the ones asked for, or the env cannot be built.</exception>
        /// <exception cref="ValueError">Thrown for an unknown vectorization mode.</exception>
        /// <exception cref="Error">Thrown when the id is not registered or has no (vector) entry point.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static VectorEnv<TObs, TAct> MakeVec<TObs, TAct>(string id, int numEnvs = 1, object? vectorizationMode = null, object? vectorKwargs = null,
            IEnumerable<Func<IEnv, IEnv>>? wrappers = null, object? kwargs = null) =>
            EnvRegistry.MakeVec<TObs, TAct>(id, numEnvs, vectorizationMode, vectorKwargs, wrappers, kwargs);

        /// <summary>Builds a vector env from a spec, typed (DR-21); see <see cref="EnvRegistry.MakeVec{TObs, TAct}(EnvSpec, int, object, object, IEnumerable{Func{IEnv, IEnv}}, object)"/>.</summary>
        /// <typeparam name="TObs">The batched observation type.</typeparam>
        /// <typeparam name="TAct">The batched action type.</typeparam>
        /// <param name="spec">The spec.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorizationMode">The vectorization mode, or <see langword="null"/>.</param>
        /// <param name="vectorKwargs">The vector env's constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <returns>The vector env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the batched types differ or the env cannot be built.</exception>
        /// <exception cref="ValueError">Thrown for an unknown vectorization mode.</exception>
        /// <exception cref="Error">Thrown when the spec has no (vector) entry point.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static VectorEnv<TObs, TAct> MakeVec<TObs, TAct>(EnvSpec spec, int numEnvs = 1, object? vectorizationMode = null, object? vectorKwargs = null,
            IEnumerable<Func<IEnv, IEnv>>? wrappers = null, object? kwargs = null) =>
            EnvRegistry.MakeVec<TObs, TAct>(spec, numEnvs, vectorizationMode, vectorKwargs, wrappers, kwargs);

        /// <summary>
        ///     Builds a registered env in a render mode, with the viewer its human window opens from (DR-84, DR-89):
        ///     <c>Make("CartPole-v1", RenderMode.Human, AvaloniaEnvViewer.Factory)</c> is
        ///     <c>make("CartPole-v1", render_mode="human")</c> with that window. The mode joins the kwargs as Gymnasium's string,
        ///     so the spec records <c>render_mode="human"</c>; the factory is passed to the env, or to the <c>HumanRendering</c>
        ///     wrapper <c>make</c> applies to an env that only renders images, and is never recorded in the spec.
        /// </summary>
        /// <param name="id">The id: <c>"CartPole-v1"</c>, an unversioned <c>"CartPole"</c>, or <c>"Assembly:Id"</c>.</param>
        /// <param name="renderMode">The render mode.</param>
        /// <param name="viewerFactory">
        ///     The factory the human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>,
        ///     then to <see cref="DependencyNotInstalled"/> at the first human render, as DR-20 resolves it.
        /// </param>
        /// <param name="kwargs">The env's other constructor arguments by Python name: a dictionary, an options record, or an anonymous object.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">
        ///     Thrown when <paramref name="kwargs"/> also holds <c>render_mode</c> (<c>make() got multiple values for keyword
        ///     argument 'render_mode'</c>), or the env cannot be built from the kwargs (an env without a <c>viewer_factory</c>
        ///     parameter refuses a factory).
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="RenderMode"/> does not declare.</exception>
        /// <exception cref="Error">Thrown when the id is not registered or has no entry point.</exception>
        /// <exception cref="NamespaceNotFound">Thrown when the id's namespace is unknown.</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown.</exception>
        /// <exception cref="VersionNotFound">Thrown when the version is newer than any registered.</exception>
        /// <exception cref="DeprecatedEnv">Thrown when the version is older than the latest.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a module named by the id or the entry point does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static IEnv Make(string id, RenderMode renderMode, EnvViewerFactory? viewerFactory = null, object? kwargs = null, long? maxEpisodeSteps = null,
            bool? disableEnvChecker = null) =>
            EnvRegistry.MakeWithViewer(id, WithRenderMode("make", kwargs, RenderModes.Name(renderMode)), maxEpisodeSteps, disableEnvChecker, viewerFactory, null);

        /// <summary>
        ///     <see cref="Make(string, RenderMode, EnvViewerFactory, object, long?, bool?)"/> with Gymnasium's string for the
        ///     mode (DR-84): any string Gymnasium accepts, a custom env's own modes included (DR-93); <see langword="null"/>
        ///     leaves the spec's mode as it is.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="renderMode">The render mode's string (<c>"rgb_array"</c>, <see cref="RenderModes"/>), or <see langword="null"/> for none given.</param>
        /// <param name="viewerFactory">The factory the human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <param name="kwargs">The env's other constructor arguments by Python name.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="kwargs"/> also holds <c>render_mode</c>, or the env cannot be built from the kwargs.</exception>
        /// <exception cref="Error">Thrown when the id is not registered or has no entry point.</exception>
        /// <exception cref="NamespaceNotFound">Thrown when the id's namespace is unknown.</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown.</exception>
        /// <exception cref="VersionNotFound">Thrown when the version is newer than any registered.</exception>
        /// <exception cref="DeprecatedEnv">Thrown when the version is older than the latest.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a module named by the id or the entry point does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static IEnv Make(string id, string? renderMode, EnvViewerFactory? viewerFactory = null, object? kwargs = null, long? maxEpisodeSteps = null,
            bool? disableEnvChecker = null) =>
            EnvRegistry.MakeWithViewer(id, WithRenderMode("make", kwargs, renderMode), maxEpisodeSteps, disableEnvChecker, viewerFactory, null);

        /// <summary>Builds the env a spec describes in a render mode, with its viewer (DR-84, DR-89); see <see cref="Make(string, RenderMode, EnvViewerFactory, object, long?, bool?)"/>.</summary>
        /// <param name="spec">The spec.</param>
        /// <param name="renderMode">The render mode.</param>
        /// <param name="viewerFactory">The factory the human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <param name="kwargs">The env's other constructor arguments by Python name, over the spec's.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/>, <paramref name="kwargs"/> also holds <c>render_mode</c>, or the env cannot be built.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="RenderMode"/> does not declare.</exception>
        /// <exception cref="Error">Thrown when the spec has no entry point.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a string entry point names a module that does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static IEnv Make(EnvSpec spec, RenderMode renderMode, EnvViewerFactory? viewerFactory = null, object? kwargs = null, long? maxEpisodeSteps = null,
            bool? disableEnvChecker = null) =>
            EnvRegistry.MakeWithViewer(SpecOrRefuse(spec), WithRenderMode("make", kwargs, RenderModes.Name(renderMode)), maxEpisodeSteps, disableEnvChecker, viewerFactory, null);

        /// <summary>Builds the env a spec describes in a render mode given as Gymnasium's string, with its viewer (DR-84, DR-89).</summary>
        /// <param name="spec">The spec.</param>
        /// <param name="renderMode">The render mode's string, or <see langword="null"/> for none given.</param>
        /// <param name="viewerFactory">The factory the human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <param name="kwargs">The env's other constructor arguments by Python name, over the spec's.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/>, <paramref name="kwargs"/> also holds <c>render_mode</c>, or the env cannot be built.</exception>
        /// <exception cref="Error">Thrown when the spec has no entry point.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a string entry point names a module that does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static IEnv Make(EnvSpec spec, string? renderMode, EnvViewerFactory? viewerFactory = null, object? kwargs = null, long? maxEpisodeSteps = null,
            bool? disableEnvChecker = null) =>
            EnvRegistry.MakeWithViewer(SpecOrRefuse(spec), WithRenderMode("make", kwargs, renderMode), maxEpisodeSteps, disableEnvChecker, viewerFactory, null);

        /// <summary>Builds a registered env typed (DR-10), in a render mode, with its viewer (DR-84, DR-89); see <see cref="Make(string, RenderMode, EnvViewerFactory, object, long?, bool?)"/>.</summary>
        /// <typeparam name="TObs">The observation type the env must have.</typeparam>
        /// <typeparam name="TAct">The action type the env must have.</typeparam>
        /// <param name="id">The id.</param>
        /// <param name="renderMode">The render mode.</param>
        /// <param name="viewerFactory">The factory the human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <param name="kwargs">The env's other constructor arguments by Python name, over the spec's.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the env is not an <c>Env&lt;TObs, TAct&gt;</c>, <paramref name="kwargs"/> also holds <c>render_mode</c>, or the env cannot be built.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="RenderMode"/> does not declare.</exception>
        /// <exception cref="Error">Thrown when the id is not registered or has no entry point.</exception>
        /// <exception cref="NamespaceNotFound">Thrown when the id's namespace is unknown.</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown.</exception>
        /// <exception cref="VersionNotFound">Thrown when the version is newer than any registered.</exception>
        /// <exception cref="DeprecatedEnv">Thrown when the version is older than the latest.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a module named by the id or the entry point does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static Env<TObs, TAct> Make<TObs, TAct>(string id, RenderMode renderMode, EnvViewerFactory? viewerFactory = null, object? kwargs = null,
            long? maxEpisodeSteps = null, bool? disableEnvChecker = null) =>
            (Env<TObs, TAct>)EnvRegistry.MakeWithViewer(id, WithRenderMode("make", kwargs, RenderModes.Name(renderMode)), maxEpisodeSteps, disableEnvChecker, viewerFactory,
                (typeof(TObs), typeof(TAct)));

        /// <summary>Builds a registered env typed (DR-10), in a render mode given as Gymnasium's string, with its viewer (DR-84, DR-89).</summary>
        /// <typeparam name="TObs">The observation type the env must have.</typeparam>
        /// <typeparam name="TAct">The action type the env must have.</typeparam>
        /// <param name="id">The id.</param>
        /// <param name="renderMode">The render mode's string, or <see langword="null"/> for none given.</param>
        /// <param name="viewerFactory">The factory the human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <param name="kwargs">The env's other constructor arguments by Python name, over the spec's.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the env is not an <c>Env&lt;TObs, TAct&gt;</c>, <paramref name="kwargs"/> also holds <c>render_mode</c>, or the env cannot be built.</exception>
        /// <exception cref="Error">Thrown when the id is not registered or has no entry point.</exception>
        /// <exception cref="NamespaceNotFound">Thrown when the id's namespace is unknown.</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown.</exception>
        /// <exception cref="VersionNotFound">Thrown when the version is newer than any registered.</exception>
        /// <exception cref="DeprecatedEnv">Thrown when the version is older than the latest.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a module named by the id or the entry point does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static Env<TObs, TAct> Make<TObs, TAct>(string id, string? renderMode, EnvViewerFactory? viewerFactory = null, object? kwargs = null,
            long? maxEpisodeSteps = null, bool? disableEnvChecker = null) =>
            (Env<TObs, TAct>)EnvRegistry.MakeWithViewer(id, WithRenderMode("make", kwargs, renderMode), maxEpisodeSteps, disableEnvChecker, viewerFactory,
                (typeof(TObs), typeof(TAct)));

        /// <summary>Builds the env a spec describes, typed (DR-10), in a render mode, with its viewer (DR-84, DR-89).</summary>
        /// <typeparam name="TObs">The observation type the env must have.</typeparam>
        /// <typeparam name="TAct">The action type the env must have.</typeparam>
        /// <param name="spec">The spec.</param>
        /// <param name="renderMode">The render mode.</param>
        /// <param name="viewerFactory">The factory the human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <param name="kwargs">The env's other constructor arguments by Python name, over the spec's.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/>, the env is not an <c>Env&lt;TObs, TAct&gt;</c>, <paramref name="kwargs"/> also holds <c>render_mode</c>, or the env cannot be built.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="RenderMode"/> does not declare.</exception>
        /// <exception cref="Error">Thrown when the spec has no entry point.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a string entry point names a module that does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static Env<TObs, TAct> Make<TObs, TAct>(EnvSpec spec, RenderMode renderMode, EnvViewerFactory? viewerFactory = null, object? kwargs = null,
            long? maxEpisodeSteps = null, bool? disableEnvChecker = null) =>
            (Env<TObs, TAct>)EnvRegistry.MakeWithViewer(SpecOrRefuse(spec), WithRenderMode("make", kwargs, RenderModes.Name(renderMode)), maxEpisodeSteps, disableEnvChecker,
                viewerFactory, (typeof(TObs), typeof(TAct)));

        /// <summary>Builds the env a spec describes, typed (DR-10), in a render mode given as Gymnasium's string, with its viewer (DR-84, DR-89).</summary>
        /// <typeparam name="TObs">The observation type the env must have.</typeparam>
        /// <typeparam name="TAct">The action type the env must have.</typeparam>
        /// <param name="spec">The spec.</param>
        /// <param name="renderMode">The render mode's string, or <see langword="null"/> for none given.</param>
        /// <param name="viewerFactory">The factory the human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <param name="kwargs">The env's other constructor arguments by Python name, over the spec's.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/>, the env is not an <c>Env&lt;TObs, TAct&gt;</c>, <paramref name="kwargs"/> also holds <c>render_mode</c>, or the env cannot be built.</exception>
        /// <exception cref="Error">Thrown when the spec has no entry point.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a string entry point names a module that does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static Env<TObs, TAct> Make<TObs, TAct>(EnvSpec spec, string? renderMode, EnvViewerFactory? viewerFactory = null, object? kwargs = null,
            long? maxEpisodeSteps = null, bool? disableEnvChecker = null) =>
            (Env<TObs, TAct>)EnvRegistry.MakeWithViewer(SpecOrRefuse(spec), WithRenderMode("make", kwargs, renderMode), maxEpisodeSteps, disableEnvChecker, viewerFactory,
                (typeof(TObs), typeof(TAct)));

        /// <summary>
        ///     Builds a vector env of copies of a registered env in a render mode, with the viewer every sub-env's human window
        ///     opens from (DR-84, DR-89): the mode joins the env kwargs as Gymnasium's string, the vectorization and autoreset
        ///     modes are the enums, and every sub-env's <c>make</c> gets the factory (the vector entry point's constructor does,
        ///     as <c>viewer_factory</c>, in that mode). Strings for the other modes go through
        ///     <see cref="MakeVec(string, int, object, object, IEnumerable{Func{IEnv, IEnv}}, object)"/>.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="renderMode">The sub-envs' render mode.</param>
        /// <param name="viewerFactory">The factory each sub-env's human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <param name="vectorizationMode">The vectorization mode, or <see langword="null"/> for the spec's default (its vector entry point, else sync).</param>
        /// <param name="autoresetMode">The vector env's <c>autoreset_mode</c>, or <see langword="null"/> for its default (<c>NEXT_STEP</c>).</param>
        /// <param name="vectorKwargs">The vector env's other constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env, innermost first.</param>
        /// <param name="kwargs">The env's other constructor arguments by Python name.</param>
        /// <returns>The vector env, untyped; the caller owns it.</returns>
        /// <exception cref="TypeError">
        ///     Thrown when <paramref name="kwargs"/> also holds <c>render_mode</c> or <paramref name="vectorKwargs"/> also holds
        ///     <c>autoreset_mode</c> (<c>make_vec() got multiple values for keyword argument '...'</c>), or the env cannot be built.
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="RenderMode"/> does not declare.</exception>
        /// <exception cref="ValueError">Thrown for a vectorization mode the enum does not declare.</exception>
        /// <exception cref="Error">Thrown when the id is not registered, has no (vector) entry point, or the vector entry point is asked for with vector kwargs (an autoreset mode included) or wrappers.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static IVectorEnv MakeVec(string id, int numEnvs, RenderMode renderMode, EnvViewerFactory? viewerFactory = null, VectorizeMode? vectorizationMode = null,
            AutoresetMode? autoresetMode = null, object? vectorKwargs = null, IEnumerable<Func<IEnv, IEnv>>? wrappers = null, object? kwargs = null) =>
            EnvRegistry.MakeVecWithViewer(id, numEnvs, vectorizationMode, WithAutoresetMode(vectorKwargs, autoresetMode), wrappers,
                WithRenderMode("make_vec", kwargs, RenderModes.Name(renderMode)), viewerFactory, null);

        /// <summary>
        ///     Builds a vector env of copies of a registered env with the vectorization and autoreset modes as enums (DR-84):
        ///     <c>MakeVec("CartPole-v1", 4, VectorizeMode.Async, AutoresetMode.SameStep)</c> is
        ///     <c>make_vec("CartPole-v1", 4, "async", vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP})</c>.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorizationMode">The vectorization mode.</param>
        /// <param name="autoresetMode">The vector env's <c>autoreset_mode</c>, or <see langword="null"/> for its default (<c>NEXT_STEP</c>).</param>
        /// <param name="vectorKwargs">The vector env's other constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env, innermost first.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <param name="viewerFactory">The factory each sub-env's human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <returns>The vector env, untyped; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="vectorKwargs"/> also holds <c>autoreset_mode</c>, or the env cannot be built.</exception>
        /// <exception cref="ValueError">Thrown for a vectorization mode the enum does not declare.</exception>
        /// <exception cref="Error">Thrown when the id is not registered, has no (vector) entry point, or the vector entry point is asked for with vector kwargs (an autoreset mode included) or wrappers.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static IVectorEnv MakeVec(string id, int numEnvs, VectorizeMode vectorizationMode, AutoresetMode? autoresetMode = null, object? vectorKwargs = null,
            IEnumerable<Func<IEnv, IEnv>>? wrappers = null, object? kwargs = null, EnvViewerFactory? viewerFactory = null) =>
            EnvRegistry.MakeVecWithViewer(id, numEnvs, vectorizationMode, WithAutoresetMode(vectorKwargs, autoresetMode), wrappers, kwargs, viewerFactory, null);

        /// <summary>Builds a vector env from a spec in a render mode, with the sub-envs' viewer (DR-84, DR-89); see <see cref="MakeVec(string, int, RenderMode, EnvViewerFactory, VectorizeMode?, AutoresetMode?, object, IEnumerable{Func{IEnv, IEnv}}, object)"/>.</summary>
        /// <param name="spec">The spec: a registered one, or one a previous <c>MakeVec</c> recorded.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="renderMode">The sub-envs' render mode.</param>
        /// <param name="viewerFactory">The factory each sub-env's human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <param name="vectorizationMode">The vectorization mode, or <see langword="null"/> for the spec's default.</param>
        /// <param name="autoresetMode">The vector env's <c>autoreset_mode</c>, or <see langword="null"/> for its default.</param>
        /// <param name="vectorKwargs">The vector env's other constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env.</param>
        /// <param name="kwargs">The env's other constructor arguments by Python name.</param>
        /// <returns>The vector env, untyped; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/>, a mode is given twice, or the env cannot be built.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="RenderMode"/> does not declare.</exception>
        /// <exception cref="ValueError">Thrown for a vectorization mode the enum does not declare.</exception>
        /// <exception cref="Error">Thrown when the spec has no (vector) entry point.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static IVectorEnv MakeVec(EnvSpec spec, int numEnvs, RenderMode renderMode, EnvViewerFactory? viewerFactory = null, VectorizeMode? vectorizationMode = null,
            AutoresetMode? autoresetMode = null, object? vectorKwargs = null, IEnumerable<Func<IEnv, IEnv>>? wrappers = null, object? kwargs = null) =>
            EnvRegistry.MakeVecWithViewer(SpecOrRefuse(spec), numEnvs, vectorizationMode, WithAutoresetMode(vectorKwargs, autoresetMode), wrappers,
                WithRenderMode("make_vec", kwargs, RenderModes.Name(renderMode)), viewerFactory, null);

        /// <summary>Builds a vector env from a spec with the vectorization and autoreset modes as enums (DR-84).</summary>
        /// <param name="spec">The spec.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorizationMode">The vectorization mode.</param>
        /// <param name="autoresetMode">The vector env's <c>autoreset_mode</c>, or <see langword="null"/> for its default.</param>
        /// <param name="vectorKwargs">The vector env's other constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <param name="viewerFactory">The factory each sub-env's human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <returns>The vector env, untyped; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/>, the autoreset mode is given twice, or the env cannot be built.</exception>
        /// <exception cref="ValueError">Thrown for a vectorization mode the enum does not declare.</exception>
        /// <exception cref="Error">Thrown when the spec has no (vector) entry point.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static IVectorEnv MakeVec(EnvSpec spec, int numEnvs, VectorizeMode vectorizationMode, AutoresetMode? autoresetMode = null, object? vectorKwargs = null,
            IEnumerable<Func<IEnv, IEnv>>? wrappers = null, object? kwargs = null, EnvViewerFactory? viewerFactory = null) =>
            EnvRegistry.MakeVecWithViewer(SpecOrRefuse(spec), numEnvs, vectorizationMode, WithAutoresetMode(vectorKwargs, autoresetMode), wrappers, kwargs, viewerFactory, null);

        /// <summary>Builds a vector env typed by its batched types (DR-21), in a render mode, with the sub-envs' viewer (DR-84, DR-89).</summary>
        /// <typeparam name="TObs">The batched observation type (<c>NDArray</c> for array spaces).</typeparam>
        /// <typeparam name="TAct">The batched action type.</typeparam>
        /// <param name="id">The id.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="renderMode">The sub-envs' render mode.</param>
        /// <param name="viewerFactory">The factory each sub-env's human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <param name="vectorizationMode">The vectorization mode, or <see langword="null"/> for the spec's default.</param>
        /// <param name="autoresetMode">The vector env's <c>autoreset_mode</c>, or <see langword="null"/> for its default.</param>
        /// <param name="vectorKwargs">The vector env's other constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env.</param>
        /// <param name="kwargs">The env's other constructor arguments by Python name.</param>
        /// <returns>The vector env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the batched types are not the ones asked for, a mode is given twice, or the env cannot be built.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="RenderMode"/> does not declare.</exception>
        /// <exception cref="ValueError">Thrown for a vectorization mode the enum does not declare.</exception>
        /// <exception cref="Error">Thrown when the id is not registered or has no (vector) entry point.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static VectorEnv<TObs, TAct> MakeVec<TObs, TAct>(string id, int numEnvs, RenderMode renderMode, EnvViewerFactory? viewerFactory = null,
            VectorizeMode? vectorizationMode = null, AutoresetMode? autoresetMode = null, object? vectorKwargs = null, IEnumerable<Func<IEnv, IEnv>>? wrappers = null,
            object? kwargs = null) =>
            (VectorEnv<TObs, TAct>)EnvRegistry.MakeVecWithViewer(id, numEnvs, vectorizationMode, WithAutoresetMode(vectorKwargs, autoresetMode), wrappers,
                WithRenderMode("make_vec", kwargs, RenderModes.Name(renderMode)), viewerFactory, (typeof(TObs), typeof(TAct)));

        /// <summary>Builds a vector env typed by its batched types (DR-21), with the vectorization and autoreset modes as enums (DR-84).</summary>
        /// <typeparam name="TObs">The batched observation type.</typeparam>
        /// <typeparam name="TAct">The batched action type.</typeparam>
        /// <param name="id">The id.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorizationMode">The vectorization mode.</param>
        /// <param name="autoresetMode">The vector env's <c>autoreset_mode</c>, or <see langword="null"/> for its default.</param>
        /// <param name="vectorKwargs">The vector env's other constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <param name="viewerFactory">The factory each sub-env's human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <returns>The vector env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the batched types are not the ones asked for, the autoreset mode is given twice, or the env cannot be built.</exception>
        /// <exception cref="ValueError">Thrown for a vectorization mode the enum does not declare.</exception>
        /// <exception cref="Error">Thrown when the id is not registered or has no (vector) entry point.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static VectorEnv<TObs, TAct> MakeVec<TObs, TAct>(string id, int numEnvs, VectorizeMode vectorizationMode, AutoresetMode? autoresetMode = null,
            object? vectorKwargs = null, IEnumerable<Func<IEnv, IEnv>>? wrappers = null, object? kwargs = null, EnvViewerFactory? viewerFactory = null) =>
            (VectorEnv<TObs, TAct>)EnvRegistry.MakeVecWithViewer(id, numEnvs, vectorizationMode, WithAutoresetMode(vectorKwargs, autoresetMode), wrappers, kwargs,
                viewerFactory, (typeof(TObs), typeof(TAct)));

        /// <summary>Builds a vector env from a spec, typed (DR-21), in a render mode, with the sub-envs' viewer (DR-84, DR-89).</summary>
        /// <typeparam name="TObs">The batched observation type.</typeparam>
        /// <typeparam name="TAct">The batched action type.</typeparam>
        /// <param name="spec">The spec.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="renderMode">The sub-envs' render mode.</param>
        /// <param name="viewerFactory">The factory each sub-env's human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <param name="vectorizationMode">The vectorization mode, or <see langword="null"/> for the spec's default.</param>
        /// <param name="autoresetMode">The vector env's <c>autoreset_mode</c>, or <see langword="null"/> for its default.</param>
        /// <param name="vectorKwargs">The vector env's other constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env.</param>
        /// <param name="kwargs">The env's other constructor arguments by Python name.</param>
        /// <returns>The vector env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/>, the batched types differ, a mode is given twice, or the env cannot be built.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="RenderMode"/> does not declare.</exception>
        /// <exception cref="ValueError">Thrown for a vectorization mode the enum does not declare.</exception>
        /// <exception cref="Error">Thrown when the spec has no (vector) entry point.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static VectorEnv<TObs, TAct> MakeVec<TObs, TAct>(EnvSpec spec, int numEnvs, RenderMode renderMode, EnvViewerFactory? viewerFactory = null,
            VectorizeMode? vectorizationMode = null, AutoresetMode? autoresetMode = null, object? vectorKwargs = null, IEnumerable<Func<IEnv, IEnv>>? wrappers = null,
            object? kwargs = null) =>
            (VectorEnv<TObs, TAct>)EnvRegistry.MakeVecWithViewer(SpecOrRefuse(spec), numEnvs, vectorizationMode, WithAutoresetMode(vectorKwargs, autoresetMode), wrappers,
                WithRenderMode("make_vec", kwargs, RenderModes.Name(renderMode)), viewerFactory, (typeof(TObs), typeof(TAct)));

        /// <summary>Builds a vector env from a spec, typed (DR-21), with the vectorization and autoreset modes as enums (DR-84).</summary>
        /// <typeparam name="TObs">The batched observation type.</typeparam>
        /// <typeparam name="TAct">The batched action type.</typeparam>
        /// <param name="spec">The spec.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorizationMode">The vectorization mode.</param>
        /// <param name="autoresetMode">The vector env's <c>autoreset_mode</c>, or <see langword="null"/> for its default.</param>
        /// <param name="vectorKwargs">The vector env's other constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <param name="viewerFactory">The factory each sub-env's human window opens from; <see langword="null"/> falls back to <see cref="HumanViewerFactory"/>.</param>
        /// <returns>The vector env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/>, the batched types differ, the autoreset mode is given twice, or the env cannot be built.</exception>
        /// <exception cref="ValueError">Thrown for a vectorization mode the enum does not declare.</exception>
        /// <exception cref="Error">Thrown when the spec has no (vector) entry point.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static VectorEnv<TObs, TAct> MakeVec<TObs, TAct>(EnvSpec spec, int numEnvs, VectorizeMode vectorizationMode, AutoresetMode? autoresetMode = null,
            object? vectorKwargs = null, IEnumerable<Func<IEnv, IEnv>>? wrappers = null, object? kwargs = null, EnvViewerFactory? viewerFactory = null) =>
            (VectorEnv<TObs, TAct>)EnvRegistry.MakeVecWithViewer(SpecOrRefuse(spec), numEnvs, vectorizationMode, WithAutoresetMode(vectorKwargs, autoresetMode), wrappers,
                kwargs, viewerFactory, (typeof(TObs), typeof(TAct)));

        /// <summary>
        ///     The call's kwargs with the render mode joined as Gymnasium's string: what <c>make(id, render_mode=..., **kwargs)</c>
        ///     passes on. A mode in the kwargs as well is Python's duplicate keyword.
        /// </summary>
        /// <param name="function">The Python function the call stands for (<c>make</c>, <c>make_vec</c>), named in the error.</param>
        /// <param name="kwargs">The call's kwargs, any form <see cref="PyKwargs.From"/> takes.</param>
        /// <param name="renderMode">The render mode's string, or <see langword="null"/> to add none.</param>
        /// <returns>A new dict the caller owns.</returns>
        /// <exception cref="TypeError">Thrown when the kwargs hold <c>render_mode</c> too: <c>make() got multiple values for keyword argument 'render_mode'</c>.</exception>
        private static Info WithRenderMode(string function, object? kwargs, string? renderMode) {
            Info merged = PyKwargs.From(kwargs);
            if (renderMode is null)
                return merged;
            if (merged.ContainsKey("render_mode"))
                throw new TypeError($"{function}() got multiple values for keyword argument 'render_mode'");
            merged["render_mode"] = renderMode;
            return merged;
        }

        /// <summary>
        ///     The call's vector kwargs with the autoreset mode joined as the <see cref="AutoresetMode"/> member, as
        ///     <c>vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP}</c> holds it. A mode in the vector kwargs as well is
        ///     refused, as a duplicate keyword is.
        /// </summary>
        /// <param name="vectorKwargs">The call's vector kwargs, any form <see cref="PyKwargs.From"/> takes.</param>
        /// <param name="autoresetMode">The autoreset mode, or <see langword="null"/> to add none.</param>
        /// <returns>The vector kwargs as given when no mode is added (<see langword="null"/> stays <see langword="null"/>), else a new dict.</returns>
        /// <exception cref="TypeError">Thrown when the vector kwargs hold <c>autoreset_mode</c> too.</exception>
        private static object? WithAutoresetMode(object? vectorKwargs, AutoresetMode? autoresetMode) {
            if (autoresetMode is null)
                return vectorKwargs;
            Info merged = PyKwargs.From(vectorKwargs);
            if (merged.ContainsKey("autoreset_mode"))
                throw new TypeError("make_vec() got multiple values for keyword argument 'autoreset_mode'");
            merged["autoreset_mode"] = autoresetMode.Value;
            return merged;
        }

        /// <summary>A spec, refused when it is <see langword="null"/> as Gymnasium refuses a <c>None</c> id.</summary>
        /// <param name="spec">The spec.</param>
        /// <returns><paramref name="spec"/>.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/>: <c>Expected id to be a string or EnvSpec, got &lt;class 'NoneType'&gt;</c>.</exception>
        private static EnvSpec SpecOrRefuse(EnvSpec spec) =>
            spec ?? throw new TypeError($"Expected id to be a string or EnvSpec, got {PyBuiltins.TypeRepr(null)}");

        /// <summary>The spec registered under an id exactly (<c>gymnasium.spec</c>); see <see cref="EnvRegistry.Spec"/>.</summary>
        /// <param name="envId">The id.</param>
        /// <returns>The registered spec itself.</returns>
        /// <exception cref="System.ArgumentNullException">Thrown when <paramref name="envId"/> is <see langword="null"/>.</exception>
        /// <exception cref="Error">Thrown for a malformed or unregistered id.</exception>
        /// <exception cref="NamespaceNotFound">Thrown when the namespace is unknown.</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown.</exception>
        /// <exception cref="VersionNotFound">Thrown when the version is newer than any registered.</exception>
        /// <exception cref="DeprecatedEnv">Thrown when the version is older than the latest.</exception>
        public static EnvSpec Spec(string envId) => EnvRegistry.Spec(envId);

        /// <summary>Registers an env for <see cref="Make(string, object, long?, bool?)"/> (<c>gymnasium.register</c>); see <see cref="EnvRegistry.Register"/>.</summary>
        /// <param name="id">The id, <c>[namespace/](env-name)[-v(version)]</c>.</param>
        /// <param name="entryPoint">The env's class, an <see cref="EnvCreator"/>, or a <c>"Namespace:Class"</c> string.</param>
        /// <param name="rewardThreshold">The return at which the task counts as solved.</param>
        /// <param name="nondeterministic">Whether the same seed and actions can still give different observations.</param>
        /// <param name="maxEpisodeSteps">The step limit <c>make</c>'s <c>TimeLimit</c> enforces.</param>
        /// <param name="orderEnforce">Whether <c>make</c> adds <c>OrderEnforcing</c>.</param>
        /// <param name="disableEnvChecker">Whether <c>make</c> leaves out <c>PassiveEnvChecker</c>.</param>
        /// <param name="additionalWrappers">Wrappers <c>make</c> applies on top, in order.</param>
        /// <param name="vectorEntryPoint">What builds a vectorized version.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <exception cref="ValueError">Thrown when neither entry point is given.</exception>
        /// <exception cref="Error">Thrown for a malformed id.</exception>
        /// <exception cref="RegistrationError">Thrown when a versioned and an unversioned id of the same name would both be registered.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="kwargs"/> is not a mapping.</exception>
        public static void Register(string id, object? entryPoint = null, double? rewardThreshold = null, bool nondeterministic = false, long? maxEpisodeSteps = null,
            bool orderEnforce = true, bool disableEnvChecker = false, IEnumerable<WrapperSpec>? additionalWrappers = null, object? vectorEntryPoint = null,
            object? kwargs = null) =>
            EnvRegistry.Register(id, entryPoint, rewardThreshold, nondeterministic, maxEpisodeSteps, orderEnforce, disableEnvChecker, additionalWrappers, vectorEntryPoint, kwargs);

        /// <summary>Prints (or returns) every registered id grouped by namespace (<c>gymnasium.pprint_registry</c>); see <see cref="EnvRegistry.PprintRegistry"/>.</summary>
        /// <param name="printRegistry">The registry to print; <see langword="null"/> is <see cref="Registry"/>.</param>
        /// <param name="numCols">How many ids per line.</param>
        /// <param name="excludeNamespaces">Namespaces to leave out, or <see langword="null"/>.</param>
        /// <param name="disablePrint">Return the text instead of writing it to the console.</param>
        /// <returns>The text when <paramref name="disablePrint"/> is set, otherwise <see langword="null"/>.</returns>
        /// <exception cref="ZeroDivisionError">Thrown when <paramref name="numCols"/> is 0.</exception>
        public static string? PprintRegistry(IReadOnlyDictionary<string, EnvSpec>? printRegistry = null, int numCols = 3, IEnumerable<string?>? excludeNamespaces = null,
            bool disablePrint = false) =>
            EnvRegistry.PprintRegistry(printRegistry, numCols, excludeNamespaces, disablePrint);

        /// <summary>A no-op that marks a module as used (<c>gymnasium.register_envs</c>); registration is the env assembly's own (DR-12).</summary>
        /// <param name="envModule">Anything; ignored.</param>
        public static void RegisterEnvs(object? envModule) => EnvRegistry.RegisterEnvs(envModule);
    }
}
