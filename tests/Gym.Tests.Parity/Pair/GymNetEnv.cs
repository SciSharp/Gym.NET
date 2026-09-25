using System;
using System.Linq;
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
    ///     Observations returned by <c>Step</c> are not disposed: envs not ported yet may return their own state array as the
    ///     observation (B-13's aliasing, which CartPole no longer has), which the next step still reads. <c>Reset</c> returns
    ///     a copy, which is disposed after it is described.
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
                // Not disposed: an env not ported yet may hand out its own state array (B-13), read again by the next step.
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
        ///     Ported envs expose Gymnasium's attribute as a public property named in PascalCase (CartPole's <c>state</c> is
        ///     <c>State</c>, a float64 <see cref="T:double[]"/>), which receives a copy of the values. Envs not ported yet keep
        ///     an <see cref="NDArray"/> field of the Python name, which receives a new array; the replaced one is left to the
        ///     collector, since an earlier observation may still alias it.
        /// </remarks>
        /// <exception cref="NotSupportedException">Thrown when the env has neither.</exception>
        public void SetAttribute(string name, double[] values) {
            PropertyInfo? property = AttributeProperty(name);
            if (property != null && property.PropertyType == typeof(double[]) && property.CanWrite) {
                property.SetValue(_env, (double[])values.Clone());
                return;
            }

            FieldInfo field = StateField(name) ?? throw new NotSupportedException($"{Name} has neither a double[] property '{PascalCase(name)}' nor an NDArray field '{name}'.");
            field.SetValue(_env, np.array(values));
        }

        /// <inheritdoc />
        /// <remarks>
        ///     A <see cref="T:double[]"/> property is described as the float64 array Gymnasium holds (through NumSharp, so it
        ///     compares with NumPy's <c>(4,) &lt;f8</c>); other property values and legacy fields are described as they are.
        /// </remarks>
        public ValueFacts? GetAttribute(string name) {
            PropertyInfo? property = AttributeProperty(name);
            if (property != null) {
                object? value = property.GetValue(_env);
                if (value is double[] doubles) {
                    using NDArray array = np.array(doubles);
                    return ValueFacts.Of(array);
                }

                return ValueFacts.Of(value);
            }

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

        /// <summary>The public instance property that ports Gymnasium's attribute <paramref name="name"/>, if the env has one.</summary>
        /// <param name="name">The attribute in Python spelling, for example <c>state</c> or <c>steps_beyond_terminated</c>.</param>
        /// <returns>The property named <see cref="PascalCase"/>(<paramref name="name"/>), or <see langword="null"/>.</returns>
        private PropertyInfo? AttributeProperty(string name) {
            return _env.GetType().GetProperty(PascalCase(name), BindingFlags.Instance | BindingFlags.Public);
        }

        /// <summary>The port plan's 1:1 naming rule (§3.8): <c>steps_beyond_terminated</c> becomes <c>StepsBeyondTerminated</c>.</summary>
        /// <param name="name">A Python attribute name.</param>
        /// <returns>Its PascalCase spelling.</returns>
        private static string PascalCase(string name) {
            return string.Concat(name.Split('_', StringSplitOptions.RemoveEmptyEntries).Select(part => char.ToUpperInvariant(part[0]) + part.Substring(1)));
        }

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
