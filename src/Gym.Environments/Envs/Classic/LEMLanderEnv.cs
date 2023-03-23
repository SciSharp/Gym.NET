using Gym.Collections;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using tainicom.Aether.Physics2D.Common;
using tainicom.Aether.Physics2D.Collision.Shapes;
using tainicom.Aether.Physics2D.Dynamics;
using tainicom.Aether.Physics2D.Dynamics.Contacts;
using tainicom.Aether.Physics2D.Dynamics.Joints;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
using System.Linq;
using Gym.Threading;
using SixLabors.ImageSharp.ColorSpaces;
using Gym.Exceptions;
using System.Diagnostics;

namespace Gym.Environments.Envs.Classic
{
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
        private LEM _LEM = null;
        private NumPyRandom RandomState { get; set; }
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
        }
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
        /// <param name="action"></param>
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
                Console.WriteLine("K=: {0:F2}", burn);
            if (_LEM.OutOfFuel)
            {
                _ElapsedTime += _LEM.Drift();
                Status = LanderStatus.FreeFall;
                if (Verbose)
                {
                    Console.WriteLine("FUEL OUT AT {0} SECONDS", _ElapsedTime);
                }
            }
            if (_LEM.Altitude < 1e-5)
            {
                float U = _LEM.SpeedMPH;
                // Check for landing
                if (U <= 1.2)
                {
                    if (Verbose)
                        Console.WriteLine("PERFECT LANDING!");
                    Status = LanderStatus.Landed;
                    score = 10.0f - U;
                }
                else if (U <= 10.0)
                {
                    if (Verbose)
                        Console.WriteLine("GOOD LANDING (COULD BE BETTER)");
                    score = 2.0f;
                    Status = LanderStatus.Landed;
                }
                else if (U > 60.0)
                {
                    if (Verbose)
                    {
                        Console.WriteLine("SORRY THERE WERE NO SURVIVORS. YOU BLEW IT!");
                        Console.WriteLine("IN FACT, YOU BLASTED A NEW LUNAR CRATER {0} FEET DEEP!", 0.277 * U);
                    }
                    Status = LanderStatus.Crashed;
                    score = -U;
                }
                else
                {
                    if (Verbose)
                    {
                        Console.WriteLine("CRAFT DAMAGE ... YOU'RE STRANDED HERE UNTIL A RESCUE");
                        Console.WriteLine("PARTY ARRIVES. HOPE YOU HAVE ENOUGH OXYGEN!");
                    }
                    score = 0.5f;
                    Status = LanderStatus.Landed;
                }
            }
            else if (_LEM.OutOfFuel)
            {
                Status = LanderStatus.Crashed;
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
            if (Verbose)
            {
                if (_ElapsedTime < 1e-5)
                {
                    Console.WriteLine("CONTROL CALLING LUNAR MODULE. MANUAL CONTROL IS NECESSARY");
                    Console.WriteLine("YOU MAY RESET FUEL RATE K EACH 10 SECS TO 0 OR ANY VALUE");
                    Console.WriteLine("BETWEEN 8 & 200 LBS/SEC. YOU'VE 16000 LBS FUEL. ESTIMATED");
                    Console.WriteLine("FREE FALL IMPACT TIME-120 SECS. CAPSULE WEIGHT-32500 LBS");
                    Console.WriteLine("FIRST RADAR CHECK COMING UP");
                    Console.WriteLine("COMMENCE LANDING PROCEDURE");
                    Console.WriteLine("TIME,SECS   ALTITUDE,MILES+FEET   VELOCITY,MPH   FUEL,LBS   FUEL RATE");
                }
                else
                {
                    Console.Write("{0,8:F0}", _ElapsedTime);
                    // Altitude
                    Console.Write("{0,15}{1,7}", Math.Truncate(_LEM.Altitude), Math.Truncate(_LEM.AltitudeFeet));
                    // VSI
                    Console.Write("{0,15:F2}", _LEM.SpeedMPH);
                    // Fuel
                    Console.Write("{0,12:F1}", _LEM.NetFuel);
                    Console.WriteLine();
                }
            }
            if (_viewer == null)
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
            // Define the buffer image for drawing
            var img = new Image<Rgba32>(VIEWPORT_W, VIEWPORT_H);
            img.Mutate(i => i.BackgroundColor(new Rgba32(0, 0, 0))); // Space is black
            // Draw 
            _viewer.Render(img);
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