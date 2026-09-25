using System;
using System.Reflection;
using Gym.Envs;
using Gym.Observations;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Oracle;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SixLabors.ImageSharp;

namespace Gym.Tests.Parity.Pair {
    /// <summary>
    ///     Puts a Gym.NET env into a <see cref="GymPair"/>, bridging its legacy API to Gymnasium's contract without changing
    ///     any value it produces.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The bridge is stated in <see cref="Adaptations"/>, which every report prints: <c>Seed(seed)</c> then
    ///     <c>Reset()</c> stands for <c>reset(seed=seed)</c>; the missing reset info and a step's null information are
    ///     reported as <c>{}</c>, what Gymnasium's classic envs return; <c>done</c> is reported as <c>terminated</c> and
    ///     <c>truncated</c> as <c>False</c>, what an env without <c>TimeLimit</c> returns. None of this touches observations,
    ///     rewards or the generator, so every byte compared is Gym.NET's own. When P1 ports the 5-tuple API, the adaptations
    ///     go away.
    ///     </para>
    ///     <para>
    ///     Observations returned by <c>Step</c> are not disposed: Gym.NET's legacy envs return their own state array as the
    ///     observation (B-13), which the next step still reads. <c>Reset</c> returns a copy, which is disposed after it is
    ///     described.
    ///     </para>
    /// </remarks>
    public sealed class GymNetEnv : IDotNetEnv {
        /// <summary>The env, owned.</summary>
        private readonly Env _env;

        /// <summary>Wraps an env; the adapter owns and disposes it.</summary>
        /// <param name="env">The Gym.NET env.</param>
        public GymNetEnv(Env env) {
            _env = env;
            Name = $"Gym.NET {env.GetType().Name}";
        }

        /// <summary>The wrapped env, for tests that also inspect its spaces.</summary>
        public Env Env => _env;

        /// <inheritdoc />
        public string Name { get; }

        /// <inheritdoc />
        public string[] Adaptations { get; } = {
            "legacy API: Seed(seed) + Reset() stands for reset(seed=seed); the reset info it lacks is reported as {}",
            "legacy API: done is reported as terminated, truncated as False (no TimeLimit, like Gymnasium's unwrapped env), null information as {}",
        };

        /// <inheritdoc />
        public CallResult Reset(long seed) {
            try {
                _env.Seed(seed);
                using NDArray observation = _env.Reset();
                return CallResult.Of(new Transition(ValueFacts.Of(observation), null, null, null, Transition.EmptyInfo));
            } catch (Exception exception) when (exception is not UnitTestAssertException) {
                return CallResult.Threw(RaisedError.FromDotNet(exception));
            }
        }

        /// <inheritdoc />
        public CallResult Step(object action) {
            try {
                Step step = _env.Step(action);
                // Not disposed: the observation is the env's own state array (B-13), read again by the next step.
                return CallResult.Of(new Transition(
                    ValueFacts.Of(step.Observation),
                    ValueFacts.Of(step.Reward),
                    ValueFacts.FromBool(step.Done),
                    ValueFacts.FromBool(false),
                    step.Information is null ? Transition.EmptyInfo : ValueFacts.Of(step.Information)));
            } catch (Exception exception) when (exception is not UnitTestAssertException) {
                return CallResult.Threw(RaisedError.FromDotNet(exception));
            }
        }

        /// <inheritdoc />
        public bool TryGetRngState(out ValueFacts? state, out string reason) {
            if (!GeneratorProbe.TryFind(_env, out Generator? generator, out reason)) {
                state = null;
                return false;
            }

            state = RngFacts.Of(generator!);
            return true;
        }

        /// <inheritdoc />
        /// <remarks>
        ///     Writes the env's private <see cref="NDArray"/> field of that name (CartPole's <c>state</c>), the Gym.NET
        ///     counterpart of <c>env.unwrapped.state</c>. The replaced array is left to the collector, since an earlier
        ///     observation may still alias it.
        /// </remarks>
        public void SetAttribute(string name, double[] values) {
            FieldInfo field = StateField(name) ?? throw new NotSupportedException($"{Name} has no NDArray field '{name}'.");
            field.SetValue(_env, np.array(values));
        }

        /// <inheritdoc />
        public ValueFacts? GetAttribute(string name) {
            FieldInfo? field = StateField(name);
            return field is null ? null : ValueFacts.Of(field.GetValue(_env));
        }

        /// <inheritdoc />
        public Frame Render() {
            using Image image = _env.Render("rgb_array");
            return Frame.FromImage(image);
        }

        /// <inheritdoc />
        public void SeedActionSpace(long seed) => _env.ActionSpace.Seed(seed);

        /// <inheritdoc />
        public object SampleAction() => _env.ActionSpace.Sample();

        /// <summary>Closes the env.</summary>
        public void Dispose() => _env.Dispose();

        /// <summary>Finds an instance field of type <see cref="NDArray"/> named <paramref name="name"/> in the env's class hierarchy.</summary>
        /// <param name="name">The field's name.</param>
        /// <returns>The field, or <see langword="null"/>.</returns>
        private FieldInfo? StateField(string name) {
            for (Type? type = _env.GetType(); type != null && type != typeof(object); type = type.BaseType) {
                FieldInfo? field = type.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
                if (field != null && typeof(NDArray).IsAssignableFrom(field.FieldType))
                    return field;
            }

            return null;
        }
    }
}
