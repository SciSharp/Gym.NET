using Gym.Collections;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;
using SixLabors.ImageSharp;
using System;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
using System.Linq;
using Gym.Threading;
using SixLabors.ImageSharp.ColorSpaces;
using Gym.Exceptions;
using System.Diagnostics;
using System.Collections.Generic;

namespace Gym.Environments.Envs.Classic
{
    /// <summary>
    /// The status of the lander during the simulation. Starts in Landing status and 
    /// is updated periodically during the simulation.
    /// </summary>
    public enum LanderStatus
    {
        Landing, // In the process of landing
        Crashed, // Smash
        Landed, // Successful
        FreeFall // Out of fuel
    }
    public class LEMLanderEnv : Env
    {
        // Screen
        public const int VIEWPORT_W = 600;
        public const int VIEWPORT_H = 400;

        private IEnvironmentViewerFactoryDelegate _viewerFactory;
        private IEnvViewer _viewer;

        private const float START_ALTITUDE = 120;
        private const float START_VELOCITY = 1;
        private const float START_MASS = 32500;

        private float _ElapsedTime;
        private float _StepTimeDuration = 10.0f;

        public LanderStatus Status { get; private set; }
        /// <summary>
        /// The LEM physics
        /// </summary>
        private LEM _LEM = null;
        private NumPyRandom RandomState { get; set; }
        /// <summary>
        /// Set to true to get the original output transcript
        /// </summary>
        public bool Verbose { get; set; } = false;

        public LEMLanderEnv(IEnvViewer viewer) : this((IEnvironmentViewerFactoryDelegate)null)
        {
            _viewer = viewer ?? throw new ArgumentNullException(nameof(viewer));
        }

        public LEMLanderEnv(IEnvironmentViewerFactoryDelegate viewerFactory, float g = 0.001f, float netmass = 16500f, Random ran = null, bool quiet = false, NumPyRandom random_state = null)
        {
            RandomState = random_state;
            if (RandomState == null)
            {
                RandomState = np.random;
            }
            _viewerFactory = viewerFactory;
            _LEM = new LEM(a: START_ALTITUDE, m: START_MASS, v: START_VELOCITY, g: g, n: netmass);
            // Altitude, Speed, Fuel
            NDArray low = np.array(new float[] { 0f, -1000, 0 });
            NDArray high = np.array(new float[] { START_ALTITUDE, 1000f, netmass });
            ObservationSpace = new Box(low, high);
            // Burn, Time
            low = np.array(new float[] { 0f, 0f });
            high = np.array(new float[] { 200f, 10f });
            ActionSpace = new Box(low, high, random_state: random_state);
            Metadata = new Dict("render.modes", new[] { "human", "rgb_array" }, "video.frames_per_second", 50);
        }
        /// <summary>
        /// Resets the LEM to the original game parameters and runs a step with 0 burn and 0 time.
        /// </summary>
        /// <returns></returns>
        public override NDArray Reset()
        {
            _LEM = new LEM(a: START_ALTITUDE, m: START_MASS, v: START_VELOCITY, g: _LEM.Gravity, n: _LEM.FuelMass);
            _ElapsedTime = 0;
            Status = LanderStatus.Landing;
            return Step(np.array(new float[] { 0f, 0f })).Observation;
        }

        /// <summary>
        /// Continuous action step that takes an array of 2 floats for the action: (thrust, duration of thrust).
        /// Original game was a 10 second fixed thrust period and variable burn amount. In this environment you will
        /// specify the burn rate and the amount of time for that burn.
        /// </summary>
        /// <param name="action">The action as a 2 element array of float (burn,time)</param>
        /// <returns>Step has (Altitude, Speed, Fuel) as the state. Information dictionary uses the same keys.</returns>
        public override Step Step(object action)
        {
            NDArray c_action = (NDArray)action;
            float burn = c_action[0];
            float time = c_action[1];
            if (burn > 200f)
            {
                burn = 200f;
            }
            if (burn < 0f)
            {
                burn = 0f;
            }
            if (time < 0f)
            {
                time = 0f;
            }
            float score = 0f;
            _ElapsedTime += _LEM.ApplyBurn(burn, time: time);
            if (Verbose)
                System.Diagnostics.Debug.WriteLine("K=: {0:F2}", burn);
            if (_LEM.OutOfFuel)
            {
                _ElapsedTime += _LEM.Drift();
                Status = LanderStatus.FreeFall;
                if (Verbose)
                {
                    System.Diagnostics.Debug.WriteLine("FUEL OUT AT {0} SECONDS", _ElapsedTime);
                }
            }
            if (_LEM.Altitude < 1e-5)
            {
                float U = _LEM.SpeedMPH;
                // Check for landing
                if (U <= 1.2)
                {
                    if (Verbose)
                        System.Diagnostics.Debug.WriteLine("PERFECT LANDING!");
                    Status = LanderStatus.Landed;
                    score = 10.0f - U;
                }
                else if (U <= 10.0)
                {
                    if (Verbose)
                        System.Diagnostics.Debug.WriteLine("GOOD LANDING (COULD BE BETTER) AT {0:F2} MPH",U);
                    score = 2.0f;
                    Status = LanderStatus.Landed;
                }
                else if (U > 60.0)
                {
                    if (Verbose)
                    {
                        System.Diagnostics.Debug.WriteLine("SORRY THERE WERE NO SURVIVORS. YOU BLEW IT!");
                        System.Diagnostics.Debug.WriteLine("IN FACT, YOU BLASTED A NEW LUNAR CRATER {0} FEET DEEP!", 0.277 * U);
                    }
                    Status = LanderStatus.Crashed;
                    score = -U;
                }
                else
                {
                    if (Verbose)
                    {
                        System.Diagnostics.Debug.WriteLine("LANDED AT {0:F2} MPH",U);
                        System.Diagnostics.Debug.WriteLine("CRAFT DAMAGE ... YOU'RE STRANDED HERE UNTIL A RESCUE");
                        System.Diagnostics.Debug.WriteLine("PARTY ARRIVES. HOPE YOU HAVE ENOUGH OXYGEN!");
                    }
                    score = 0.5f;
                    Status = LanderStatus.Landed;
                }
            }
            else if (_LEM.OutOfFuel)
            {
                score = -_LEM.SpeedMPH;
            }
            Step step = new Step();
            step.Information = new Dict();
            step.Information["Distance"] = _LEM.Altitude;
            step.Information["Speed"] = _LEM.Speed;
            step.Information["Fuel"] = _LEM.NetFuel;
            step.Observation = new float[] { _LEM.Altitude, _LEM.Speed, _LEM.NetFuel };
            step.Done = (Status != LanderStatus.Landing);
            step.Reward = score;
            return (step);
        }

        public override Image Render(string mode = "human")
        {
            float ms = 1f / 50f * 1000f; // ms left for this frame
            long lstart = DateTime.Now.Ticks;
            if (Verbose)
            {
                if (_ElapsedTime < 1e-5)
                {
                    System.Diagnostics.Debug.WriteLine("CONTROL CALLING LUNAR MODULE. MANUAL CONTROL IS NECESSARY");
                    System.Diagnostics.Debug.WriteLine("YOU MAY RESET FUEL RATE K EACH 10 SECS TO 0 OR ANY VALUE");
                    System.Diagnostics.Debug.WriteLine("BETWEEN 8 & 200 LBS/SEC. YOU'VE 16000 LBS FUEL. ESTIMATED");
                    System.Diagnostics.Debug.WriteLine("FREE FALL IMPACT TIME-120 SECS. CAPSULE WEIGHT-32500 LBS");
                    System.Diagnostics.Debug.WriteLine("FIRST RADAR CHECK COMING UP");
                    System.Diagnostics.Debug.WriteLine("COMMENCE LANDING PROCEDURE");
                    System.Diagnostics.Debug.WriteLine("TIME,SECS   ALTITUDE,MILES+FEET   VELOCITY,MPH   FUEL,LBS   FUEL RATE");
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine("{0,8:F0}{1,15}{2,7}{3,15:F2}{4,12:F1}", _ElapsedTime, Math.Truncate(_LEM.Altitude), Math.Truncate(_LEM.AltitudeFeet), _LEM.SpeedMPH, _LEM.NetFuel);
                }
            }
            if (_viewer == null && mode == "human")
            {
                lock (this)
                {
                    //to prevent double initalization.
                    if (_viewer == null)
                    {
                        if (_viewerFactory == null)
                            _viewerFactory = NullEnvViewer.Factory;
                        _viewer = _viewerFactory(VIEWPORT_W, VIEWPORT_H, "lemlander-v1").GetAwaiter().GetResult();
                    }
                }
            }            // Define the buffer image for drawing
            var img = new Image<Rgba32>(VIEWPORT_W, VIEWPORT_H);
            var draw = new List<(IPath, Rgba32)>();
            var fill = new List<(IPath, Rgba32)>();
            // Setup the gauge for fuel and speed
            int gauge_w = 100;
            int gauge_h = VIEWPORT_H / 3;

            var fuel_gauge = new RectangularPolygon(VIEWPORT_W/4 - gauge_w/2, VIEWPORT_H/2 - gauge_h, gauge_w, gauge_h);
            draw.Add((fuel_gauge, new Rgba32(204, 153, 102)));
            float net_fuel = _LEM.NetFuel / _LEM.FuelMass*(gauge_h-4); // [0,1]
            fuel_gauge = new RectangularPolygon(fuel_gauge.Left+3, fuel_gauge.Bottom-net_fuel-2, gauge_w-4, net_fuel);
            if (net_fuel < 0.1)
            {
                fill.Add((fuel_gauge, new Rgba32(255, 0, 0)));
            }
            else
            {
                fill.Add((fuel_gauge, new Rgba32(0, 255, 0)));
            }

            var speed_gauge = new RectangularPolygon(3*VIEWPORT_W / 4 - gauge_w / 2, VIEWPORT_H / 2 - gauge_h, gauge_w, gauge_h);
            draw.Add((speed_gauge, new Rgba32(204, 153, 102)));
            float speed = Math.Abs(_LEM.SpeedMPH / 3600 * (gauge_h - 4)); // [0,1]
            speed_gauge = new RectangularPolygon(speed_gauge.Left + 3, speed_gauge.Bottom - speed - 2, gauge_w - 4, speed);
            if (_LEM.Speed < 0f)
            {
                fill.Add((speed_gauge, new Rgba32(255, 0, 0)));
            }
            else
            {
                fill.Add((speed_gauge, new Rgba32(0, 255, 0)));
            }

            var alt_gauge = new RectangularPolygon(VIEWPORT_W / 2 - gauge_w / 2, VIEWPORT_H / 2 - gauge_h, gauge_w, gauge_h);
            draw.Add((alt_gauge, new Rgba32(204, 153, 102)));
            float alt = Math.Abs(_LEM.Altitude / 118 * (gauge_h - 4)); // [0,1]
            alt_gauge = new RectangularPolygon(alt_gauge.Left + 3, alt_gauge.Bottom - alt - 2, gauge_w - 4, alt);
            if (_LEM.Altitude < 15)
            {
                fill.Add((alt_gauge, new Rgba32(255, 0, 0)));
            }
            else
            {
                fill.Add((alt_gauge, new Rgba32(0, 255, 0)));
            }

            // Do the drawing
            img.Mutate(i => i.BackgroundColor(new Rgba32(0, 0, 0))); // Space is black
            foreach (var (path, rgba32) in draw)
            {
                img.Mutate(i => i.Draw(rgba32,1f, path));
            }
            foreach (var (path, rgba32) in fill)
            {
                img.Mutate(i => i.Fill(rgba32, path));
            }
            //img.Mutate(x => x.DrawText("FUEL", ..., Color.Yellow, new PointF(100, 100)));
            //img.Mutate(x => x.DrawText("ALTITUDE", ..., Color.Yellow, new PointF(100, 100)));
            //img.Mutate(x => x.DrawText("SPEED", ..., Color.Yellow, new PointF(100, 100)));

            _viewer.Render(img);
            ms -= (float)(new TimeSpan(DateTime.Now.Ticks - lstart).TotalMilliseconds);
            if (ms > 0)
            {
                System.Threading.Thread.Sleep((int)ms);
            }
            return (img);
        }
        public override void CloseEnvironment()
        {
            if (_viewer != null)
            {
                _viewer.CloseEnvironment();
                _viewer = null;
            }
        }

        public override void Seed(int seed)
        {
            RandomState.seed(seed);
        }
    }
}