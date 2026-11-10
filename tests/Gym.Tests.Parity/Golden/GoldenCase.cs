using System;
using System.Collections.Generic;
using System.Globalization;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;

namespace Gym.Tests.Parity.Golden {
    /// <summary>
    ///     One case of the golden catalog (plan §9): everything needed to run one trajectory live against Gymnasium, to record
    ///     it, and to replay it without Python. The same instance drives all three, so the Python-free replay can never check
    ///     something the live cell did not.
    /// </summary>
    /// <remarks>
    ///     A case is identified by <see cref="Key"/> (<c>suite/name</c>), which is also its golden's path under
    ///     <c>tests/Golden</c>. Changing anything that shapes the trajectory (seed, actions, steps, frames, kwargs) makes the
    ///     committed golden stale: the replay refuses it and asks for a re-recording, rather than comparing against a
    ///     trajectory of a different case.
    /// </remarks>
    public sealed record GoldenCase {
        /// <summary>The family directory under <c>tests/Golden</c>, Gymnasium's env package name (<c>classic_control</c>).</summary>
        public required string Suite { get; init; }

        /// <summary>The case's file name without extension, for example <c>CartPole-v1.policy.seed0</c>; unique within the suite.</summary>
        public required string Name { get; init; }

        /// <summary>The Gymnasium id the Python side is made from (<c>gymnasium.make(EnvId, **Kwargs)</c>).</summary>
        public required string EnvId { get; init; }

        /// <summary>Constructor keyword arguments in Python spelling, passed to <c>gymnasium.make</c>; the .NET factory must apply the same ones.</summary>
        public IReadOnlyDictionary<string, object?> Kwargs { get; init; } = new Dictionary<string, object?>();

        /// <summary>The seed of the trajectory's one <c>reset(seed=…)</c>.</summary>
        public required long Seed { get; init; }

        /// <summary>Where the actions come from, live and in replay.</summary>
        public required GoldenActions Actions { get; init; }

        /// <summary>The most steps to take; the trajectory stops earlier when the episode ends or both sides raise.</summary>
        public required int MaxSteps { get; init; }

        /// <summary>Compare (and record) a frame after the reset and every this many steps; 0 records no frames.</summary>
        public int FrameEvery { get; init; }

        /// <summary>An attribute both sides expose and the pair compares after every call (<c>state</c>), or <see langword="null"/>.</summary>
        public string? StateAttribute { get; init; }

        /// <summary>
        ///     Whether the Python side keeps <c>make</c>'s wrapper stack (<c>TimeLimit</c>, <c>OrderEnforcing</c>,
        ///     <c>PassiveEnvChecker</c>) instead of stepping the unwrapped env; <see cref="CreateDotNet"/> must then build
        ///     Gym.NET's stack too (<c>Gymnasium.Make</c>).
        /// </summary>
        /// <remarks>
        ///     For a case whose id differs from another only in its registration, such as CartPole-v0's 200-step limit: the
        ///     unwrapped env would record CartPole-v1's dynamics again, while the stack records the truncation. The golden's
        ///     header stores the flag only when it is set, so the goldens recorded before it existed stay current.
        /// </remarks>
        public bool ViaMake { get; init; }

        /// <summary>
        ///     What makes the trajectory depend on the platform (<c>CartPole's cos/sin</c>), or <see langword="null"/>. Such a
        ///     case is strict only on the manifest's reference platform; elsewhere a replay divergence is reported Inconclusive
        ///     (port plan §14.9), because another libm may round one ulp differently.
        /// </summary>
        public string? PlatformDependence { get; init; }

        /// <summary>
        ///     Builds the .NET side, given the render mode (<c>rgb_array</c> when the case records frames, otherwise
        ///     <see langword="null"/>). Called once per live run and once per replay; the caller owns the result.
        /// </summary>
        public required Func<string?, IDotNetEnv> CreateDotNet { get; init; }

        /// <summary>The case's identity and golden path, <c>suite/name</c>.</summary>
        public string Key => $"{Suite}/{Name}";

        /// <summary>The render mode both sides are built with: <c>rgb_array</c> when frames are recorded, otherwise none.</summary>
        public string? RenderMode => FrameEvery > 0 ? "rgb_array" : null;

        /// <summary>
        ///     Makes the Python side: the unwrapped env, as the cases recorded before P3's <c>make</c> step it, or
        ///     <c>make</c>'s whole stack when <see cref="ViaMake"/> is set.
        /// </summary>
        /// <returns>The live Gymnasium env; the caller disposes it.</returns>
        /// <exception cref="global::Python.Runtime.PythonException">Thrown when <c>gymnasium.make</c> raises.</exception>
        public PyGymEnv CreatePython() => PyGymEnv.Make(EnvId, Kwargs, viaMake: ViaMake, renderMode: RenderMode);

        /// <summary>The kwargs as the golden's header stores them: each value described, in the order given.</summary>
        /// <returns>The described kwargs.</returns>
        public IReadOnlyList<KeyValuePair<string, ValueFacts>> DescribedKwargs() {
            var described = new List<KeyValuePair<string, ValueFacts>>();
            foreach (KeyValuePair<string, object?> kwarg in Kwargs)
                described.Add(new KeyValuePair<string, ValueFacts>(kwarg.Key, ValueFacts.Of(kwarg.Value)));
            return described;
        }

        /// <inheritdoc />
        public override string ToString() => Key;
    }

    /// <summary>
    ///     Where a golden case's actions come from. Live, each kind is an <see cref="IActionSource"/>; in replay, the
    ///     recorded actions are fed back (open-loop), except for <see cref="Sampled"/>, whose replay samples the .NET action
    ///     space itself and requires every sample to equal the recorded one, so space sampling stays checked without Python.
    /// </summary>
    public abstract record GoldenActions {
        /// <summary>Restricts the hierarchy to the kinds below, so every replay knows every kind.</summary>
        private GoldenActions() { }

        /// <summary>How reports and the golden's header name the source, for example <c>policy cartpole_balance</c>.</summary>
        public abstract string Description { get; }

        /// <summary>Whether replay samples the .NET action space (seeded after the reset) instead of feeding the recorded actions.</summary>
        public virtual bool ReplaySamples => false;

        /// <summary>The source the live run draws its actions from.</summary>
        /// <returns>A new source.</returns>
        public abstract IActionSource CreateLive();

        /// <summary>A named policy evaluated on Gymnasium's observations (<see cref="ActionSources.Policy"/>), replayed open-loop.</summary>
        /// <param name="Name">The policy, for example <c>cartpole_balance</c>.</param>
        public sealed record Policy(string Name) : GoldenActions {
            /// <inheritdoc />
            public override string Description => $"policy {Name}";

            /// <inheritdoc />
            public override IActionSource CreateLive() => ActionSources.Policy(Name);
        }

        /// <summary>
        ///     Seeded sampling of both action spaces (<see cref="ActionSources.Sampled"/>); replay samples the .NET space with
        ///     the same seed and compares each sample with the recorded one.
        /// </summary>
        /// <param name="Seed">The action spaces' seed.</param>
        public sealed record Sampled(long Seed) : GoldenActions {
            /// <inheritdoc />
            public override string Description => $"action_space.sample() seeded {Seed.ToString(CultureInfo.InvariantCulture)}";

            /// <inheritdoc />
            public override bool ReplaySamples => true;

            /// <inheritdoc />
            public override IActionSource CreateLive() => ActionSources.Sampled(Seed);
        }

        /// <summary>The same discrete action every step (<see cref="ActionSources.Constant"/>), replayed open-loop.</summary>
        /// <param name="Action">The action.</param>
        public sealed record Constant(long Action) : GoldenActions {
            /// <inheritdoc />
            public override string Description => $"always {Action.ToString(CultureInfo.InvariantCulture)}";

            /// <inheritdoc />
            public override IActionSource CreateLive() => ActionSources.Constant(Action);
        }
    }
}
