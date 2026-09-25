#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Threading.Tasks;
using Gym.Collections;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Legacy {
    /// <summary>
    ///     Runs a ported env behind the pre-0.26 API (<see cref="ILegacyEnv"/>), so code written for Gym.NET's former
    ///     contract keeps working during the migration (port plan §16). Built on Gymnasium's
    ///     <c>step_api_compatibility</c>: every step is converted with <see cref="StepApiCompatibility.ConvertToDoneStepApi{TObs}"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The mapping: <see cref="Seed"/> records the seed for the next <see cref="Reset"/>, which passes it to
    ///     <c>Reset(seed)</c> (where Gymnasium seeds); <see cref="Reset"/> returns the observation and drops the info;
    ///     <see cref="Step"/> reports <c>terminated || truncated</c> as <see cref="Legacy.Step.Done"/>, marks a truncation in
    ///     <c>Information["TimeLimit.truncated"]</c>, and narrows the float64 reward to float32; <see cref="Render"/> only
    ///     accepts the env's construction-time render mode.
    ///     </para>
    ///     <para>
    ///     A scalar observation (a <c>Discrete</c> observation space) is handed out as a 0-d array, since the legacy API only
    ///     carries arrays. <see cref="Metadata"/> is a snapshot converted on every read.
    ///     </para>
    /// </remarks>
    [Obsolete("LegacyEnvAdapter bridges the pre-0.26 API; call the env directly (Reset(seed), Step returning StepResult<TObs>). Gym.Legacy is removed in 3.0.")]
    public class LegacyEnvAdapter : ILegacyEnv {
        /// <summary>The seed <see cref="Seed"/> recorded for the next reset, or <see langword="null"/>.</summary>
        private BigInteger? _pendingSeed;

        /// <summary>Adapts <paramref name="env"/>, which the adapter owns from now on (disposing the adapter closes it).</summary>
        /// <param name="env">The ported env, possibly wrapped.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public LegacyEnvAdapter(IEnv env) {
            Env = env ?? throw new ArgumentNullException(nameof(env));
        }

        /// <summary>The adapted env.</summary>
        public IEnv Env { get; }

        /// <summary>
        ///     The env's metadata as a legacy <see cref="Dict"/>: a snapshot, so editing it does not reach the env; assigning a
        ///     dict replaces the env's metadata.
        /// </summary>
        /// <exception cref="ArgumentNullException">Thrown when assigning <see langword="null"/>.</exception>
        public Dict Metadata {
            get {
                var dict = new Dict();
                foreach (KeyValuePair<string, object?> entry in Env.Metadata)
                    dict[entry.Key] = entry.Value!;
                return dict;
            }
            set {
                if (value is null)
                    throw new ArgumentNullException(nameof(value));
                var metadata = new Metadata();
                foreach (KeyValuePair<string, object> entry in value)
                    metadata[entry.Key] = entry.Value;
                Env.Metadata = metadata;
            }
        }

        /// <summary><c>gym</c>'s <c>reward_range</c>, <c>(-inf, inf)</c> unless assigned; nothing reads it.</summary>
        public (float From, float To) RewardRange { get; set; } = (float.NegativeInfinity, float.PositiveInfinity);

        /// <summary>The env's action space.</summary>
        public Space ActionSpace {
            get => Env.ActionSpace;
            set => Env.ActionSpace = value;
        }

        /// <summary>The env's observation space.</summary>
        public Space ObservationSpace {
            get => Env.ObservationSpace;
            set => Env.ObservationSpace = value;
        }

        /// <summary>
        ///     Resets the env, with the seed a preceding <see cref="Seed"/> recorded (then forgotten), and returns the first
        ///     observation.
        /// </summary>
        /// <returns>The observation as an array; the caller owns it.</returns>
        /// <exception cref="NotSupportedException">Thrown when the observation is neither an array nor a number.</exception>
        public NDArray Reset() {
            BigInteger? seed = _pendingSeed;
            _pendingSeed = null;
            ResetResult<object?> result = Env.Reset(seed);
            return ToArray(result.Observation);
        }

        /// <summary>Steps the env and converts the transition to the done API.</summary>
        /// <param name="action">The action, converted by the env.</param>
        /// <returns>The legacy transition.</returns>
        /// <exception cref="NotSupportedException">Thrown when the observation is neither an array nor a number.</exception>
        public Step Step(object action) {
            DoneStepResult<object?> done = StepApiCompatibility.ConvertToDoneStepApi(Env.Step(action));
            var information = new Dict();
            foreach (KeyValuePair<string, object?> entry in done.Info)
                information[entry.Key] = entry.Value!;
            return new Step(ToArray(done.Observation), (float)done.Reward, done.Done, information);
        }

        /// <summary>Runs <see cref="Step"/> on the thread pool; await it before the next call on the same env.</summary>
        /// <param name="action">The action.</param>
        /// <returns>A task completing with the legacy transition, or faulting with what the step threw.</returns>
        public Task<Step> StepAsync(object action) => Task.Run(() => Step(action));

        /// <summary>
        ///     Renders the env when <paramref name="mode"/> is its construction-time render mode: an <c>rgb_array</c> frame as
        ///     an image, <see langword="null"/> for a mode that returns nothing (<c>human</c>).
        /// </summary>
        /// <param name="mode">The mode; it must equal the env's <see cref="IEnv.RenderMode"/>.</param>
        /// <returns>The frame as an RGB image the caller owns, or <see langword="null"/>.</returns>
        /// <exception cref="UnsupportedMode">
        ///     Thrown when <paramref name="mode"/> differs from the env's render mode: since Gym 0.26 the mode is fixed at
        ///     construction.
        /// </exception>
        /// <exception cref="NotSupportedException">Thrown when the rendering is not a frame or nothing (text, frame lists).</exception>
        public Image? Render(string mode = "human") {
            if (!string.Equals(mode, Env.RenderMode, StringComparison.Ordinal))
                throw new UnsupportedMode($"The env renders in mode {PyBuiltins.Repr(Env.RenderMode)}, fixed at construction since Gym 0.26; build it with render_mode={PyBuiltins.Repr(mode)} to render that way.");
            switch (Env.Render()) {
                case RenderResult.NoneResult:
                    return null;
                case RenderResult.Rgb rgb: {
                    using NDArray frame = rgb.Frame;
                    int height = (int)frame.shape[0], width = (int)frame.shape[1];
                    return Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), width, height);
                }
                case var other:
                    throw new NotSupportedException($"The legacy API renders images only; the env returned {other}.");
            }
        }

        /// <summary>Closes the env.</summary>
        public void CloseEnvironment() => Env.Close();

        /// <summary>
        ///     Records <paramref name="seed"/> for the next <see cref="Reset"/>, the legacy spelling of <c>Reset(seed: seed)</c>.
        ///     Draws made before that reset still come from the current generator.
        /// </summary>
        /// <param name="seed">A non-negative seed.</param>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message, at once rather than at the reset.</exception>
        public void Seed(long seed) {
            if (seed < 0)
                throw new Error($"Seed must be greater or equal to zero, actual value: {seed.ToString(CultureInfo.InvariantCulture)}");
            _pendingSeed = seed;
        }

        /// <summary>Closes the env (<see cref="CloseEnvironment"/>).</summary>
        public void Dispose() {
            Env.Close();
            GC.SuppressFinalize(this);
        }

        /// <summary>The legacy form of an observation: an array as is, a number as a 0-d array of its type.</summary>
        /// <param name="observation">The observation.</param>
        /// <returns>The array.</returns>
        /// <exception cref="NotSupportedException">Thrown for anything else (text, dicts, tuples), which the legacy API cannot carry.</exception>
        private static NDArray ToArray(object? observation) {
            switch (observation) {
                case NDArray array:
                    return array;
                case bool or sbyte or byte or short or ushort or int or uint or long or ulong or float or double:
                    return NDArray.Scalar(observation);
                default:
                    throw new NotSupportedException($"The legacy API carries array observations only, not {PyBuiltins.Repr(observation)} ({PyBuiltins.TypeRepr(observation)}).");
            }
        }
    }
}
