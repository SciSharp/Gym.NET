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
using System.Runtime.InteropServices;

namespace Gym.Environments.Envs.Aether
{
    /// <summary>
    ///## Description
    ///The easiest control task to learn from pixels - a top-down
    ///racing environment. The generated xtrack is random every episode.

    ///Some indicators are shown at the bottom of the window along with the
    ///state RGB buffer. From left to right: true speed, four ABS sensors,
    ///steering wheel position, and gyroscope.
    ///To play yourself (it's rather fast for humans), type:
    ///```
    ///python gymnasium/envs/box2d/car_racing.py
    ///```
    ///Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    ///and turn at the same time.

    ///## Action Space
    ///If continuous there are 3 actions :
    ///- 0: steering, -1 is full left, +1 is full right
    ///- 1: gas
    ///- 2: breaking

    ///If discrete there are 5 actions:
    ///- 0: do nothing
    ///- 1: steer left
    ///- 2: steer right
    ///- 3: gas
    ///- 4: brake

    ///## Observation Space

    ///A top-down 96x96 RGB image of the car and race xtrack.

    ///## Rewards
    ///The reward is -0.1 every frame and +1000/N for every xtrack tile visited,
    ///where N is the total number of tiles visited in the xtrack. For example,
    ///if you have finished in 732 frames, your reward is
    ///1000 - 0.1*732 = 926.8 points.

    ///## Starting State
    ///The car starts at rest in the center of the road.

    ///## Episode Termination
    ///The episode finishes when all the tiles are visited. The car can also go
    ///outside the playfield - that is, far off the xtrack, in which case it will
    ///receive -100 reward and die.

    ///## Arguments
    ///`lap_complete_percent` dictates the percentage of tiles that must be visited by
    ///the agent before a lap is considered complete.

    ///Passing `domain_randomize=True` enables the domain randomized variant of the environment.
    ///In this scenario, the background and xtrack colours are different on every reset.

    ///Passing `continuous=False` converts the environment to use discrete action space.
    ///The discrete action space has 5 actions: [do nothing, left, right, gas, brake].

    ///## Reset Arguments
    ///Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    ///Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    ///`domain_randomize` must be `True` on init for this argument to work.
    ///Example usage:
    ///```python
    ///import gymnasium as gym
    ///env = gym.make("CarRacing-v1", domain_randomize=True)

    ///# normal reset, this changes the colour scheme by default
    ///env.reset()

    ///# reset with colour scheme change
    ///env.reset(options={"randomize": True})

    ///# reset with no colour scheme change
    ///env.reset(options={"randomize": False})
    ///```

    ///## Version History
    ///- v1: Change xtrack completion logic and add domain randomization (0.24.0)
    ///- v0: Original version

    ///## References
    ///- Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ///## Credits
    ///Created by Oleg Klimov
    ///Converted to C# by Jacob Anderson
    /// </summary>
    public class CarRacingEnv : Env
    {
        #region Constants
        public const int STATE_W = 96; // Atari was 160 x 192
        public const int STATE_H = 96;
        public const int VIDEO_W = 600;
        public const int VIDEO_H = 400;
        public const int WINDOW_W = 1000;
        public const int WINDOW_H = 800;

        public const float SCALE = 6; // Track scale
        public const float TRACK_RAD = 900f / SCALE;
        public const float PLAYFIELD = 2000f / SCALE;
        public const float FPS = 50.0f; // Frames per second
        public const float ZOOM = 2.7f; // Camera zoom
        public const bool ZOOM_FOLLOW = true; // Set to false for a fixed view

        public const float TRACK_DETAIL_STEP = 21f / SCALE;
        public const float TRACK_TURN_RATE = 0.31f;
        public const float TRACK_WIDTH = 40f / SCALE;
        public const float BORDER = 8f / SCALE;
        public const int BORDER_MIN_COUNT = 4;
        public const float GRASS_DIM = PLAYFIELD / 20f;
        private float MAX_SHAPE_DIM = Math.Max(Math.Max(GRASS_DIM, TRACK_WIDTH), TRACK_DETAIL_STEP) * 1.414213f * ZOOM * SCALE;
        #endregion

        #region Display Members
        private IEnvironmentViewerFactoryDelegate _viewerFactory;
        private IEnvViewer _viewer;
        public Rgba32 RoadColor { get; set; } = new Rgba32(102, 102, 102);
        public Rgba32 BackgroundColor { get; set; } = new Rgba32(102, 204, 102);
        public Rgba32 GrassColor { get; set; } = new Rgba32(102, 230, 102);
        #endregion

        private bool ContinuousMode { get; set; } = true;
        private bool DomainRandomize { get; set; } = false;
        private float LapCompletePercent { get; set; } = 0.95f;
        private bool Verbose { get; set; } = false;
        private NumPyRandom RandomState { get; set; }
        private FrictionDetector ContactListener { get; set; }

        private CarDynamics Car { get; set; }
        private float Reward { get; set; } = 0f;
        private float PreviousReward { get; set; } = 0f;
        private float T { get; set; } = 0f;
        private bool NewLap { get; set; } = false;
        private float StartAlpha { get; set; } = 0f;

        internal struct RoadPolygon
        {
            internal Vertices Verts;
            internal Rgba32 Color;
        }

        private List<RoadPolygon> RoadPoly { get; set; } = new List<RoadPolygon>();
        private List<Body> Road { get; set; } = new List<Body>();
        private float[] Track { get; set; }

        internal class FrictionDetector
        {
            private CarRacingEnv _env { get; set; }
            private float LapCompletePercent { get; set; }

            public FrictionDetector(CarRacingEnv env, float lap_complete_percent)
            {
                _env = env;
                LapCompletePercent = lap_complete_percent;
            }

            private void DoContact(Contact contact)
            {
                RoadTile tile = contact.FixtureA.Body.Tag as RoadTile;
                CarDynamics car = contact.FixtureB.Body.Tag as CarDynamics;
                if (tile == null)
                {
                    tile = contact.FixtureB.Body.Tag as RoadTile;
                    car = contact.FixtureA.Body.Tag as CarDynamics;
                }
                if (tile == null || car == null)
                {
                    // Contacts are only with Car and Tile instances
                    return;
                }
                if (car.Tiles.Count > 0)
                {
                }
            }
            public bool BeginContact(Contact contact)
            {
                DoContact(contact);
                return (false);
            }

            public void EndContact(Contact contact)
            {
                DoContact(contact);
            }

        }

        public CarRacingEnv(IEnvViewer viewer) : this((IEnvironmentViewerFactoryDelegate)null)
        {
            _viewer = viewer ?? throw new ArgumentNullException(nameof(viewer));
        }

        public CarRacingEnv(IEnvironmentViewerFactoryDelegate viewerFactory, float lap_complete_percent = 0.95f, bool domain_randomize = false, bool continuous = true, bool verbose = false, NumPyRandom random_state = null)
        {
            RandomState = random_state;
            if (RandomState == null)
            {
                RandomState = np.random;
            }
            _viewerFactory = viewerFactory;
            LapCompletePercent = lap_complete_percent;
            DomainRandomize = domain_randomize;
            ContinuousMode = continuous;
            Verbose = verbose;
            if (continuous)
            {
                ActionSpace = new Box(new float[] { -1f, 0f, 0f }, new float[] { 1f, 1f, 1f });
            }
            else
            {
                ActionSpace = new Discrete(5);
            }
            ObservationSpace = new Box(0f, 255f, shape: new NumSharp.Shape(STATE_H, STATE_W, 3));
            InitColors();
            ContactListener = new FrictionDetector(this, lap_complete_percent);
        }

        private void InitColors()
        {
            if (DomainRandomize)
            {
                NDArray rando = RandomState.uniform(0, 210, 3).astype(NPTypeCode.Int32);
                RoadColor = new Rgba32(rando[0], rando[1], rando[2]);
                rando = RandomState.uniform(0, 210, 3).astype(NPTypeCode.Int32);
                BackgroundColor = new Rgba32(rando[0], rando[1], rando[2]);
                int[] offset = new int[] { 0, 0, 0 };
                int idx = RandomState.randint(0, 3);
                offset[idx] = 20;
                GrassColor = new Rgba32(rando[0]+offset[0], rando[1] + offset[1], rando[2] + offset[2]);
            }
            else
            {
                RoadColor = new Rgba32(102, 102, 102);
                BackgroundColor = new Rgba32(102, 204, 102);
                GrassColor = new Rgba32(102, 230, 102);
            }
        }

        private bool CreateTrack()
        {
            int CHECKPOINTS = 12;
            float[] checkpoints = new float[CHECKPOINTS * 3];
            float frac = 2f * (float)Math.PI / (float)CHECKPOINTS;
            // Create checkpoints
            int i = 0;
            for (i = 0; i < CHECKPOINTS; i++)
            {
                int j = i * 3; // index into the checkpoints array
                float noise = RandomState.uniform(0f, frac);
                float alpha = frac * (float)i + noise;
                float rad = RandomState.uniform(TRACK_RAD / 3f, TRACK_RAD);

                if (i == 0)
                {
                    alpha = 0f;
                    rad = 1.5f * TRACK_RAD;
                }
                else if(i == CHECKPOINTS-1)
                {
                    alpha = frac * (float)i;
                    StartAlpha = frac * (-0.5f);
                    rad = 1.5f * TRACK_RAD;
                }
                checkpoints[j++] = alpha;
                checkpoints[j++] = rad * (float)Math.Cos(alpha);
                checkpoints[j++] = rad * (float)Math.Sin(alpha);
                Debug.WriteLine("Checkpoint {0}: {1}", i, alpha);
            }
            Debug.WriteLine("Start Alpha={0}", StartAlpha);
            Road.Clear();
            // Go from one checkpoint to another to create the xtrack.
            float x = 1.5f * TRACK_RAD;
            float y = 0f;
            float beta = 0f;
            int dest_i = 0;
            int laps = 0;
            int no_freeze = 2500;
            bool visited_other_size = false;
            List<float> xtrack = new List<float>();
            while (no_freeze > 0)
            {
                float alpha = (float)Math.Atan2(y, x);
                if (visited_other_size && alpha > 0f)
                {
                    laps++;
                    visited_other_size = false;
                }
                if (alpha < 0f)
                {
                    visited_other_size = true;
                    alpha += 2f * (float)Math.PI;
                }
                float dest_x = 0f;
                float dest_y = 0f;
                bool failed = false;
                while (true)
                {
                    failed = true;
                    while (true)
                    {
                        float dest_alpha = checkpoints[dest_i % checkpoints.Length];
                        dest_x = checkpoints[dest_i % checkpoints.Length + 1];
                        dest_y = checkpoints[dest_i % checkpoints.Length + 2];
                        if (alpha <= dest_alpha)
                        {
                            failed = false;
                            break;
                        }
                        dest_i += 3; // each element is 3
                        if (dest_i % checkpoints.Length == 0)
                            break;
                    }
                    if (!failed)
                    {
                        break;
                    }
                    alpha -= 2f * (float)Math.PI;
                }
                Debug.WriteLine("Freeze {0} {3}: dest_i={2}, alpha={1}", no_freeze, alpha, dest_i/3,failed);

                float r1x = (float)Math.Cos(beta);
                float r1y = (float)Math.Sin(beta);
                float p1x = -r1y;
                float p1y = r1x;
                float dest_dx = dest_x - x; // Vector towards destination
                float dest_dy = dest_y - y;
                // Destination vector projected on rad:
                float proj = r1x * dest_dx + r1y * dest_dy;
                while ((beta - alpha) > (1.5f * (float)Math.PI))
                {
                    beta -= 2f * (float)Math.PI;
                }
                while ((beta - alpha) < (1.5f * (float)Math.PI))
                {
                    beta += 2f * (float)Math.PI;
                }
                float prev_beta = beta;
                proj *= SCALE;
                if (proj > 0.3f)
                    beta -= Math.Min(TRACK_TURN_RATE, (float)Math.Abs(0.001 * proj));
                if (proj < -0.3f)
                    beta += Math.Min(TRACK_TURN_RATE, (float)Math.Abs(0.001 * proj));
                x += p1x * TRACK_DETAIL_STEP;
                y += p1y * TRACK_DETAIL_STEP;
                xtrack.Add(alpha);
                xtrack.Add(prev_beta * 0.5f + beta * 0.5f);
                xtrack.Add(x);
                xtrack.Add(y);
                if (laps > 4)
                {
                    break;
                }
                no_freeze--;
            }

            // Find closed loop range i1 .. i2, first loop should be ignored, second is OK
            int i1 = -1;
            int i2 = -1;
            i = xtrack.Count; // Each element is 4 floats
            while (true)
            {
                i -= 4; // 4 floats per element
                if (i == 0)
                {
                    // Failed
                    return (false);
                }
                bool pass_through_start = (xtrack[i] > StartAlpha && xtrack[i - 4] <= StartAlpha);
                if (pass_through_start && i2 == -1)
                {
                    i2 = i;
                }
                else if (pass_through_start && i1 == -1)
                {
                    i1 = i;
                    break;
                }
            }
            if (Verbose)
            {
                Debug.WriteLine("Track generation {0}..{1} -> {2}-tiles track", i1, i2, i2 - i1);
            }
            Debug.Assert(i1 != -1);
            Debug.Assert(i2 != -1);
            float[] track = new float[i2 - i1];
            Array.Copy(xtrack.ToArray().Skip(i1).ToArray(), track, i2 - i1);
            float first_beta = xtrack[1];
            float first_perp_x = (float)Math.Cos(first_beta);
            float first_perp_y = (float)Math.Sin(first_beta);
            // Length of perpendicular jump to put together head and tail
            float a = first_perp_x * (track[2] - track[track.Length - 4 + 2]); // 4 floats per element
            float b = first_perp_y * (track[3] - track[track.Length - 4 + 3]);
            float well_glued_together = (float)Math.Sqrt(a * a + b * b);
            if (well_glued_together > TRACK_DETAIL_STEP)
            {
                return (false);
            }
            // Red-white border on hard turns
            bool[] border = new bool[track.Length >> 2];
            for (i = 0; i < track.Length; i += 4)
            {
                bool good = true;
                int oneside = 0;
                for (int neg = 0; neg < BORDER_MIN_COUNT; neg++) {
                    int idx = i - neg * 4;
                    if (idx < 0)
                    {
                        idx += track.Length;
                    }
                    float beta1 = track[idx  + 1];
                    idx = i - (neg - 1) * 4;
                    if (idx < 0)
                    {
                        idx += track.Length;
                    }
                    float beta2 = track[idx+1]; // index out of bounds TODO!
                    float dbeta = beta1 - beta2;
                    good = good && (Math.Abs(dbeta) > TRACK_TURN_RATE * 0.2f);
                    oneside += (dbeta < 0f ? -1 : 1);
                }
                good = good && (Math.Abs(oneside) == BORDER_MIN_COUNT);
                border[i >> 2] = good;
            }
            for (i = 0; i < track.Length; i += 4)
            {
                int j = i >> 2;
                for (int neg = 0; neg < BORDER_MIN_COUNT; neg++)
                {
                    border[j - neg] |= border[j];
                }
            }
            // Create tiles
            return (true);
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

        public override Image Render(string mode = "human")
        {
            throw new NotImplementedException();
        }

        public override NDArray Reset()
        {
            World world = new World(new Vector2(0f, 0f));
            world.ContactManager.BeginContact = new BeginContactDelegate(ContactListener.BeginContact);
            world.ContactManager.EndContact = new EndContactDelegate(ContactListener.EndContact);
            Reward = 0f;
            PreviousReward = 0f;
            T = 0f;
            NewLap = false;
            RoadPoly.Clear();
            InitColors();
            bool done = false;
            int iters = 0;
            while (!done && iters < 100)
            {
                done = CreateTrack();
                iters++;
                if (Verbose)
                {
                    Debug.WriteLine("Retry to generate track (normal if there are not many instances of this message).");
                }
            }
            Car = new CarDynamics(world, Track[1], Track[2], Track[3]);
            return (Step(null).Observation);
        }

        public override Step Step(object action)
        {
            throw new NotImplementedException();
        }
    }
}
