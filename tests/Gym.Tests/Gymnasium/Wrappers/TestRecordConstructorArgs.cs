#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>RecordConstructorArgs</c> tests of the wrappers (<c>test_record_constructor_args.py</c>), run against every wrapper in <c>Gym.Wrappers</c>.</summary>
    /// <remarks>
    ///     Python compares the keywords each <c>__init__</c> passes to <c>RecordConstructorArgs.__init__</c>, read from its
    ///     source, with the constructor's signature. Here each wrapper is built with the arguments it needs and its recorded
    ///     <c>_saved_kwargs</c> compared with its public constructor's parameters by Python name; a wrapper the table below
    ///     does not know fails, so a new wrapper gets checked too.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_constructor_args.py">tests/wrappers/test_record_constructor_args.py</see>.
    /// </remarks>
    [TestClass]
    public class TestRecordConstructorArgs {
        /// <summary>
        ///     How each wrapper is built for the check: the class (closed where its type arguments cannot be inferred from the
        ///     env), the env it wraps, and the required arguments by Python name (the optional ones keep their defaults).
        /// </summary>
        private static readonly Dictionary<string, (Type Type, Func<IEnv> Env, Info Kwargs)> Wrappers = new() {
            ["AddRenderObservation"] = (typeof(AddRenderObservation<,>), () => Make("CartPole-v1", new { render_mode = "rgb_array" }), new Info()),
            ["Autoreset"] = (typeof(Autoreset<,>), () => Make("CartPole-v1"), new Info()),
            ["ClipAction"] = (typeof(ClipAction<,>), () => Make("Pendulum-v1"), new Info()),
            ["ClipReward"] = (typeof(ClipReward<,>), () => Make("CartPole-v1"), new Info { ["min_reward"] = 0.0 }),
            ["DelayObservation"] = (typeof(DelayObservation<,>), () => Make("CartPole-v1"), new Info { ["delay"] = 1L }),
            ["DiscretizeAction"] = (typeof(DiscretizeAction<,>), () => Make("Pendulum-v1"), new Info { ["bins"] = 3L }),
            ["DiscretizeObservation"] = (typeof(DiscretizeObservation<,>), () => Make("MountainCar-v0"), new Info { ["bins"] = 3L }),
            ["DtypeObservation"] = (typeof(DtypeObservation<,>), () => Make("CartPole-v1"), new Info { ["dtype"] = "float64" }),
            ["FilterObservation"] = (typeof(FilterObservation<,>), () => new TimeAwareObservation<NDArray, NDArray>((Env<NDArray, NDArray>)Make("CartPole-v1"), flatten: false),
                new Info { ["filter_keys"] = new List<object?> { "obs" } }),
            ["FlattenObservation"] = (typeof(FlattenObservation<,>), () => Make("CartPole-v1"), new Info()),
            ["FrameStackObservation"] = (typeof(FrameStackObservation<,>), () => Make("CartPole-v1"), new Info { ["stack_size"] = 2L }),
            ["GrayscaleObservation"] = (typeof(GrayscaleObservation<,>), () => new AddRenderObservation<NDArray, NDArray>((Env<NDArray, NDArray>)Make("CartPole-v1", new { render_mode = "rgb_array" })), new Info()),
            ["MaxAndSkipObservation"] = (typeof(MaxAndSkipObservation<,>), () => Make("CartPole-v1"), new Info()),
            ["NormalizeObservation"] = (typeof(NormalizeObservation<,>), () => Make("CartPole-v1"), new Info()),
            ["NormalizeReward"] = (typeof(NormalizeReward<,>), () => Make("CartPole-v1"), new Info()),
            ["OrderEnforcing"] = (typeof(OrderEnforcing<,>), () => new CartPoleEnv(), new Info()),
            ["PassiveEnvChecker"] = (typeof(PassiveEnvChecker<,>), () => new CartPoleEnv(), new Info()),
            ["RecordEpisodeStatistics"] = (typeof(RecordEpisodeStatistics<,>), () => Make("CartPole-v1"), new Info()),
            ["RenderCollection"] = (typeof(RenderCollection<,>), () => Make("CartPole-v1", new { render_mode = "rgb_array" }), new Info()),
            ["RepeatAction"] = (typeof(RepeatAction<,>), () => Make("CartPole-v1"), new Info { ["num_repeats"] = 2L }),
            ["RescaleAction"] = (typeof(RescaleAction<,>), () => Make("Pendulum-v1"), new Info { ["min_action"] = -1.0, ["max_action"] = 1.0 }),
            ["RescaleObservation"] = (typeof(RescaleObservation<,>), () => Make("Pendulum-v1"), new Info { ["min_obs"] = -1.0, ["max_obs"] = 1.0 }),
            ["ReshapeObservation"] = (typeof(ReshapeObservation<,>), () => Make("CartPole-v1"), new Info { ["shape"] = new long[] { 2, 2 } }),
            ["ResizeObservation"] = (typeof(ResizeObservation<,>), () => new AddRenderObservation<NDArray, NDArray>((Env<NDArray, NDArray>)Make("CartPole-v1", new { render_mode = "rgb_array" })),
                new Info { ["shape"] = new long[] { 10, 10 } }),
            ["StickyAction"] = (typeof(StickyAction<,>), () => Make("CartPole-v1"), new Info { ["repeat_action_probability"] = 0.5 }),
            ["TimeAwareObservation"] = (typeof(TimeAwareObservation<,>), () => Make("CartPole-v1"), new Info()),
            ["TimeLimit"] = (typeof(TimeLimit<,>), () => new CartPoleEnv(), new Info { ["max_episode_steps"] = 10L }),
            ["TransformAction"] = (typeof(TransformAction<NDArray, NDArray, NDArray>), () => Make("Pendulum-v1"),
                new Info { ["func"] = (Func<NDArray, NDArray>)(action => action), ["action_space"] = null }),
            ["TransformObservation"] = (typeof(TransformObservation<NDArray, NDArray, NDArray>), () => Make("CartPole-v1"),
                new Info { ["func"] = (Func<NDArray, NDArray>)(obs => obs), ["observation_space"] = null }),
            ["TransformReward"] = (typeof(TransformReward<,>), () => Make("CartPole-v1"), new Info { ["func"] = (Func<double, double>)(reward => reward) }),
        };

        /// <summary>Every public wrapper class of <c>Gym.Wrappers</c> by name (<c>WRAPPER_NAMES</c>: <c>gymnasium.wrappers.__all__</c> without the vector module).</summary>
        public static IEnumerable<object[]> WrapperNames => typeof(TimeLimit<,>).Assembly.GetTypes()
            .Where(type => type.IsPublic && !type.IsAbstract && type.Namespace == "Gym.Wrappers" && typeof(IWrapper).IsAssignableFrom(type))
            .Select(type => type.Name.Split('`')[0])
            .Distinct()
            .OrderBy(name => name, StringComparer.Ordinal)
            .Select(name => new object[] { name });

        /// <summary>
        ///     Every constructor argument of a wrapper is recorded under its Python name: <c>make</c> rebuilds a wrapper from its
        ///     spec's saved arguments, so an argument not recorded is lost in the round trip, and one recorded under another
        ///     name raises <c>TypeError</c>.
        /// </summary>
        /// <param name="wrapperName">The wrapper class's name.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_constructor_args.py#L53">tests/wrappers/test_record_constructor_args.py (test_wrappers_record_every_constructor_argument)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(WrapperNames))]
        public void TestWrappersRecordEveryConstructorArgument(string wrapperName) {
            Assert.IsTrue(Wrappers.TryGetValue(wrapperName, out var entry), $"{wrapperName} is not in this test's table: add how to build it.");
            Type definition = entry.Type.IsGenericType ? entry.Type.GetGenericTypeDefinition() : entry.Type;
            if (!typeof(IRecordConstructorArgs).IsAssignableFrom(definition))
                Assert.Inconclusive($"{wrapperName} does not record its constructor arguments at all");

            var kwargs = new Info(entry.Kwargs) { ["env"] = entry.Env() };
            using var wrapper = (IEnv)PyKwargs.Construct(entry.Type, kwargs);
            var recorded = new HashSet<string>(((IRecordConstructorArgs)wrapper).SavedKwargs.Keys);
            HashSet<string> parameters = ConstructorArgumentNames(wrapper.GetType());
            Assert.IsTrue(recorded.SetEquals(parameters),
                $"{wrapperName} records [{string.Join(", ", recorded.OrderBy(n => n))}] but takes [{string.Join(", ", parameters.OrderBy(n => n))}]");
        }

        /// <summary>An env rebuilt from its spec is wrapped with the arguments it was given.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_constructor_args.py#L75">tests/wrappers/test_record_constructor_args.py (test_env_spec_roundtrip_keeps_wrapper_arguments)</see>.</remarks>
        [TestMethod]
        public void TestEnvSpecRoundtripKeepsWrapperArguments() {
            using (var env = new DiscretizeAction<NDArray, NDArray>((Env<NDArray, NDArray>)Make("Pendulum-v1"), bins: 2L, multidiscrete: true))
            using (IEnv rebuilt = Make(env.Spec!))
                Assert.AreEqual(env.ActionSpace, rebuilt.ActionSpace);

            using (var env = new RecordEpisodeStatistics<NDArray, NDArray>((Env<NDArray, NDArray>)Make("CartPole-v1"), bufferLength: 17, statsKey: "my_stats"))
            using (IEnv rebuilt = Make(env.Spec!)) {
                Assert.AreEqual(PyAttributes.GetAttr(env, "_stats_key"), PyAttributes.GetAttr(rebuilt, "_stats_key"));
                Assert.AreEqual(env.TimeQueue.MaxLen, ((Deque<double>)PyAttributes.GetAttr(rebuilt, "time_queue")!).MaxLen);
            }
        }

        /// <summary>The named parameters of a wrapper's public constructor by Python name, other than the wrapped env (<c>constructor_argument_names</c>).</summary>
        /// <param name="type">The wrapper's class.</param>
        /// <returns>The names.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_constructor_args.py#L41">tests/wrappers/test_record_constructor_args.py (constructor_argument_names)</see>.</remarks>
        private static HashSet<string> ConstructorArgumentNames(Type type) {
            // The constructor make binds: the public one that takes the env, with the most parameters, not counting the one the
            // generator adds from the options record (DR-34), (env, <Class>Options options), which takes the same arguments.
            string optionsRecord = type.Name.Split('`')[0] + "Options";
            ConstructorInfo constructor = type.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .Where(c => c.GetParameters().Any(p => p.Name == "env"))
                .Where(c => !c.GetParameters().Any(p => p.ParameterType.Name.Split('`')[0] == optionsRecord))
                .OrderByDescending(c => c.GetParameters().Length)
                .First();
            return new HashSet<string>(constructor.GetParameters().Where(p => p.Name != "env").Select(PyKwargs.ParameterName));
        }
    }
}
