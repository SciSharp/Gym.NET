using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using Gym.Collections;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using JetBrains.Annotations;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
//using PointF = SixLabors.Primitives.PointF;

namespace Gym.Environments.Envs.Classic {
    public class CartPoleEnv : Env {
        private IEnvironmentViewerFactoryDelegate _viewerFactory;
        private IEnvViewer _viewer;

        //constants
        private const float gravity = 9.8f;
        private const float masscart = 1.0f;
        private const float masspole = 0.1f;
        private const float total_mass = masspole + masscart;
        private const float length = 0.5f;
        private const float polemass_length = masspole * length;
        private const float force_mag = 10.0f;
        private const float tau = 0.02f;
        private const string kinematics_integrator = "euler";

        private const float theta_threshold_radians = (float) (12 * 2 * Math.PI / 360); // Angle at which to fail the episode   

        private const float x_threshold = 2.4f;

        //properties
        /// <summary>The env's generator, Gymnasium's <c>np_random</c>: <see cref="Reset"/> draws the initial state from it.</summary>
        private Generator random;
        private NDArray state;
        private int steps_beyond_done = -1;

        /// <summary>
        ///     Creates the env with its action space (two pushes) and observation space. Every generator involved is the env's
        ///     own: the env draws its initial states from <paramref name="randomState"/>, and each space has a separate,
        ///     entropy-seeded generator that <c>ActionSpace.Seed</c> reseeds.
        /// </summary>
        /// <param name="viewerFactory">Creates the viewer on the first <see cref="Render"/>; <see langword="null"/> means <see cref="NullEnvViewer"/>.</param>
        /// <param name="randomState">
        ///     The generator <see cref="Reset"/> draws from (<see cref="Gym.Utils.Seeding.NpRandom"/> makes one), or
        ///     <see langword="null"/> for one of the env's own, seeded from OS entropy (B-15: lazy seeding as in Gymnasium waits
        ///     for P4).
        /// </param>
        public CartPoleEnv(IEnvironmentViewerFactoryDelegate viewerFactory, Generator randomState) {
            _viewerFactory = viewerFactory;
            // Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds  
            var high = np.array(x_threshold * 2, float.MaxValue, theta_threshold_radians * 2, float.MaxValue);
            ActionSpace = new Discrete(2);
            ObservationSpace = new Box((-1 * high), high, np.float32);
            // An unseeded generator draws OS entropy (NumPy's SeedSequence()), so envs created back to back never share a
            // stream; NumSharp's legacy np.random.RandomState() fallback seeded from Environment.TickCount (trap 6).
            random = randomState ?? Seeding.NpRandom().Rng;

            Metadata = new Dict("render.modes", new[] {"human", "rgb_array"}, "video.frames_per_second", 50);
        }

        public CartPoleEnv(IEnvironmentViewerFactoryDelegate viewerFactory) : this(viewerFactory, null) {}

        public CartPoleEnv([NotNull] IEnvViewer viewer) : this((IEnvironmentViewerFactoryDelegate) null) {
            _viewer = viewer ?? throw new ArgumentNullException(nameof(viewer));
        }

        public CartPoleEnv() : this(NullEnvViewer.Factory) {
        }

        public override NDArray Reset() {
            steps_beyond_done = -1;
            state = random.uniform(-0.05, 0.05, 4);
            return np.array(state);
        }

        public override Image Render(string mode = "human") {
            float b, t, r, l;
            const int screen_width = 600;
            const int screen_height = 400;
            const float world_width = x_threshold * 2;
            const float scale = screen_width / world_width;
            const int carty = 300;
            const float polewidth = 10.0f;
            const float poleheight = scale * (2 * length);
            const float cartwidth = 50.0f;
            const float cartheight = 30.0f;


            if (_viewer == null)
                lock (this) {
                    //to prevent double initalization.
                    if (_viewer == null) {
                        if (_viewerFactory == null)
                            _viewerFactory = NullEnvViewer.Factory;
                        _viewer = _viewerFactory(screen_width, screen_height, "cartpole-v1").GetAwaiter().GetResult();
                    }
                }

            //pole
            l = -polewidth / 2;
            r = polewidth / 2;
            t = poleheight - polewidth / 2;
            b = -polewidth / 2;
            var pole = new RectangularPolygon(-polewidth / 2, carty - poleheight, polewidth, poleheight);
            var circle = new EllipsePolygon(0, carty - polewidth / 2, polewidth / 2);

            //cart
            l = -cartwidth / 2;
            r = cartwidth / 2;
            t = cartheight / 2;
            b = -cartheight / 2;
            var axleoffset = cartheight / 4.0;
            var cart = new RectangularPolygon(-cartwidth / 2, carty - cartheight / 2, cartwidth, cartheight);
            var draw = new List<(IPath, Rgba32)>();

            if (!(state is null)) {
                var center_x = (float) (state.GetDouble(0) * scale + screen_width / 2.0f);
                //no y cuz it doesnt change.
                var cbounds = circle.Bounds;
                var pivotPoint = new PointF(cbounds.X + cbounds.Width / 2f, cbounds.Y + cbounds.Height / 2f);

                draw.Add((cart.Translate(center_x, 0), new Rgba32(0,0,0)));
                draw.Add((pole.Transform(Matrix3x2.CreateRotation((float) state.GetDouble(2), pivotPoint)).Translate(center_x, 0), new Rgba32(204, 153, 102)));
                draw.Add((circle.Translate(center_x, 0), new Rgba32(204, 153, 102)));
            } else {
                draw.Add((pole, new Rgba32(204, 153, 102)));
                draw.Add((cart, new Rgba32(0, 0, 0)));
                draw.Add((circle, new Rgba32(0, 0, 0)));
            }

            var img = new Image<Rgba32>(screen_width, screen_height);

            //line
            img.Mutate(i => i.BackgroundColor(new Rgba32(255, 255, 255)));
            img.Mutate(i => i.Fill(new Rgba32(0, 0, 0), new RectangleF(new PointF(0, carty), new SizeF(screen_width, 1))));
            foreach (var (path, rgba32) in draw) {
                img.Mutate(i => i.Fill(rgba32, path));
            }

            _viewer.Render(img);
            return img;
        }

        /// <summary>
        ///     Pushes the cart left (action 0) or right (action 1) and integrates one Euler step of <c>tau</c> seconds.
        /// </summary>
        /// <param name="action">
        ///     0 or 1 as any integer width, a <see cref="bool"/>, an enum member, or the 0-d array <c>ActionSpace.Sample()</c>
        ///     returns.
        /// </param>
        /// <returns>
        ///     The new state (a float64 array of shape (4,)), the reward, and the done flag. The observation is the env's own
        ///     state array, so writing to it changes the next step (B-13's aliasing, removed with the ownership step).
        /// </returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="action"/> does not denote an integer.</exception>
        /// <exception cref="OverflowException">Thrown when an unsigned 64-bit action exceeds <see cref="long.MaxValue"/>.</exception>
        /// <remarks>
        ///     An integer other than 0 or 1 is caught only by the <c>Debug.Assert</c> below, so Release builds treat it as
        ///     "push left" (B-14, P4).
        /// </remarks>
        public override Step Step(object action) {
            // (int)action unboxed exactly: it threw InvalidCastException for a long, and for the 0-d int64 NDArray that
            // ActionSpace.Sample() returns (trap 1 of docs/plans/NUMSHARP_MIGRATION_PLAN.md §7).
            long iaction = Discrete.ToInt64(action);
            Debug.Assert(ActionSpace.Contains(iaction), $"{action} ({action.GetType().Name}) invalid action for {GetType().Name} environment");
            //get the last step data
            var x = state.GetDouble(0);
            var x_dot = state.GetDouble(1);
            var theta = state.GetDouble(2);
            var theta_dot = state.GetDouble(3);

            var force = iaction == 1 ? force_mag : -force_mag;
            var costheta = Math.Cos(theta);
            var sintheta = Math.Sin(theta);
            var temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
            var thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
            var xacc = temp - polemass_length * thetaacc * costheta / total_mass;
            // ReSharper disable once ConditionIsAlwaysTrueOrFalse
            if (kinematics_integrator == "euler") {
                x = x + tau * x_dot;
                x_dot = x_dot + tau * xacc;
                theta = theta + tau * theta_dot;
                theta_dot = theta_dot + tau * thetaacc;
            } else {
                // semi-implicit euler
                x_dot = x_dot + tau * xacc;
                x = x + tau * x_dot;
                theta_dot = theta_dot + tau * thetaacc;
                theta = theta + tau * theta_dot;
            }

            state = np.array(x, x_dot, theta, theta_dot);
            var done = x < -x_threshold || x > x_threshold || theta < -theta_threshold_radians || theta > theta_threshold_radians;
            float reward;
            if (!done) {
                reward = 1.0f;
            } else if (steps_beyond_done == -1) {
                // Pole just fell!
                steps_beyond_done = 0;
                reward = 1.0f;
            } else {
                if (steps_beyond_done == 0) {
                    Console.WriteLine("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.");
                    //todo logging: logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.");
                }

                steps_beyond_done += 1;
                reward = 0.0f;
            }

            return new Step(state, reward, done, null);
        }

        /// <remarks>Sets internally stored viewer to null. Might cause problems if factory was not passed.</remarks>
        public override void CloseEnvironment() {
            if (_viewer != null) {
                _viewer.CloseEnvironment();
                _viewer = null;
            }
        }

        /// <summary>
        ///     Gives the env a fresh generator, Gymnasium's <c>seeding.np_random(seed)</c>, which the next <see cref="Reset"/>
        ///     draws its initial state from: <c>Seed(s)</c> then <c>Reset()</c> starts where Gymnasium's
        ///     <c>reset(seed=s)</c> starts. The action space keeps its own generator; seed it separately.
        /// </summary>
        /// <param name="seed">Any non-negative seed.</param>
        /// <exception cref="Gym.Exceptions.Error">Thrown when <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        public override void Seed(long seed) {
            random = Seeding.NpRandom(seed).Rng;
        }
    }
}