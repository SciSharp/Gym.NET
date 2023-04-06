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
using System.IO;

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
        public enum StateFormat {
            /// <summary>
            /// State data is pixel data from rendered images in [w,h,3] format
            /// </summary>
            Pixels,
            /// <summary>
            /// State data is the telemetry of the car
            /// </summary>
            Telemetry
        };

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

        /// <summary>
        /// How the state data is output from the step. 
        /// </summary>
        public StateFormat StateOutputFormat { get; set; } = StateFormat.Pixels;
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
        private int TileVisitedCount { get; set; } = 0;

        private Image<Rgba32> _LastRender = null;
        private NDArray _LastImageArray = null;
        private World _World;

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
            private bool EnvNewLap { get; set; } = false;

            public FrictionDetector(CarRacingEnv env, float lap_complete_percent)
            {
                _env = env;
                LapCompletePercent = lap_complete_percent;
            }

            private void DoContact(Contact contact,bool begin)
            {
                RoadTile tile = contact.FixtureA.Body.Tag as RoadTile;
                CarDynamics.Wheel car = contact.FixtureB.Body.Tag as CarDynamics.Wheel;
                if (tile == null)
                {
                    tile = contact.FixtureB.Body.Tag as RoadTile;
                    car = contact.FixtureA.Body.Tag as CarDynamics.Wheel;
                }
                if (tile == null || car == null)
                {
                    // Contacts are only with Car and Tile instances
                    return;
                }
                if (begin)
                {
                    tile.Color = _env.RoadColor;
                    car.Tiles.Add(tile);
                    if (!tile.RoadVisited)
                    {
                        tile.RoadVisited = true;
                        _env.Reward += 1000f / (float)_env.Track.Length;
                        _env.TileVisitedCount++;
                        float visitpct = (float)_env.TileVisitedCount / (float)_env.Track.Length;
                        if (tile.Index == 0 && visitpct > LapCompletePercent)
                        {
                            EnvNewLap = true;
                        }
                    }
                }
                else
                {
                    car.Tiles.Remove(tile);
                }
            }
            public bool BeginContact(Contact contact)
            {
                DoContact(contact,true);
                return (false);
            }

            public void EndContact(Contact contact)
            {
                DoContact(contact,false);
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
                    while (idx < 0)
                    {
                        idx += track.Length;
                    }
                    float beta1 = track[idx  + 1];
                    idx = i - neg*4 - 4;
                    while (idx < 0)
                    {
                        idx += track.Length;
                    }
                    float beta2 = track[idx+1]; // index out of bounds TODO!
                    float dbeta = beta1 - beta2;
                    good &= (Math.Abs(dbeta) > TRACK_TURN_RATE * 0.2f);
                    oneside += (dbeta < 0f ? -1 : 1);
                }
                good = good && (Math.Abs(oneside) == BORDER_MIN_COUNT);
                border[i >> 2] = good;
            }
            for (i = 0; i < border.Length; i++)
            {
                for (int neg = 0; neg < BORDER_MIN_COUNT; neg++)
                {
                    int j = i - neg;
                    if (j < 0)
                    {
                        j += border.Length;
                    }
                    
                    border[j] |= border[i];
                }
            }
            // Create tiles
            int prev_track_index = track.Length - 4;
            for (i = 0; i < track.Length; i += 4, prev_track_index = (prev_track_index+4)%track.Length) // (alpha, beta, x, y)
            {
                // position 1
                float alpha1 = track[i];
                float beta1 = track[i + 1];
                float x1 = track[i + 2];
                float y1 = track[i + 3];
                // previous position
                float alpha2 = track[prev_track_index];
                float beta2 = track[prev_track_index + 1];
                float x2 = track[prev_track_index + 2];
                float y2 = track[prev_track_index + 3];
                float cos_beta1 = (float)Math.Cos(beta1);
                float sin_beta1 = (float)Math.Sin(beta1);
                float cos_beta2 = (float)Math.Cos(beta2);
                float sin_beta2 = (float)Math.Sin(beta2);
                Vector2 road1_l = new Vector2(x1 - TRACK_WIDTH * cos_beta1, y1 - TRACK_WIDTH * sin_beta1);
                Vector2 road1_r = new Vector2(x1 + TRACK_WIDTH * cos_beta1, y1 + TRACK_WIDTH * sin_beta1);
                Vector2 road2_l = new Vector2(x2 - TRACK_WIDTH * cos_beta2, y2 - TRACK_WIDTH * sin_beta2);
                Vector2 road2_r = new Vector2(x2 + TRACK_WIDTH * cos_beta2, y2 + TRACK_WIDTH * sin_beta2);
                Vertices vx = new Vertices();
                vx.Add(road1_l);
                vx.Add(road1_r);
                vx.Add(road2_r);
                vx.Add(road2_l);
                // fixtureDef(shape = polygonShape(vertices =[(0, 0), (1, 0), (1, -1), (0, -1)]))
                Body bx = _World.CreateBody(bodyType:BodyType.Static);
                Fixture t = bx.CreateFixture(new PolygonShape(vx,1f));
                RoadTile tile = new RoadTile();
                bx.Tag = tile;
                tile.Index = i / 4;
                int c = 25 * (tile.Index % 3);
                tile.Color = new Rgba32(RoadColor.R + c, RoadColor.G + c, RoadColor.B + c);
                tile.RoadVisited = false;
                tile.Friction = 1f;
                t.IsSensor = true;
                tile.PhysicsFixture = t;
                RoadPolygon poly = new RoadPolygon();
                poly.Verts = vx;
                poly.Color = tile.Color;
                RoadPoly.Add(poly);
                Road.Add(bx);
                if (border[tile.Index])
                {
                    int side = ((beta2-beta1) < 0) ? -1 : 1;
                    Vector2 b1_l = new Vector2(x1 + side * TRACK_WIDTH * cos_beta1, y1 + side * TRACK_WIDTH * sin_beta1);
                    Vector2 b1_r = new Vector2(x1 + side * (TRACK_WIDTH+BORDER) * cos_beta1, y1 + side * (TRACK_WIDTH+BORDER) * sin_beta1);
                    Vector2 b2_l = new Vector2(x2 + side * TRACK_WIDTH * cos_beta2, y2 + side * TRACK_WIDTH * sin_beta2);
                    Vector2 b2_r = new Vector2(x2 + side * (TRACK_WIDTH + BORDER) * cos_beta2, y2 + side * (TRACK_WIDTH + BORDER) * sin_beta2);
                    vx = new Vertices();
                    vx.Add(b1_l);
                    vx.Add(b1_r);
                    vx.Add(b2_r);
                    vx.Add(b2_l);
                    poly = new RoadPolygon();
                    poly.Verts = vx;
                    poly.Color = (tile.Index % 2 == 0) ? new Rgba32(255,255,255) : new Rgba32(255, 0, 0);
                    RoadPoly.Add(poly);
                }
            }
            Track = track;
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

        /// <summary>
        /// Useful method to draw a filled polygon that is zoomed, translated, and rotated.
        /// </summary>
        /// <param name="img">Target drawing surface</param>
        /// <param name="poly">The path to fill</param>
        /// <param name="c">The color to fill it with</param>
        /// <param name="zoom">Zoom factor</param>
        /// <param name="trans">Translation factor</param>
        /// <param name="angle">Rotation factor</param>
        private void FillPoly(Image<Rgba32> img, Vector2[] poly, Color c, float zoom, Vector2 trans, float angle)
        {
            PointF[] path = new PointF[poly.Length];
            for (int i = 0; i < poly.Length; i++)
            {
                Vector2 v = CarDynamics.RotateVec(poly[i], angle) * zoom + trans;
                path[i] = new PointF(v.X, v.Y);
            }
            img.Mutate(i => i.FillPolygon(c, path));
        }

        #region Indicator Rendering
        private void RenderIndicatorIfMin(Image<Rgba32> img, float v, PointF[] path, Color c)
        {
            if (Math.Abs(v) < 1e-4)
            {
                return;
            }
            img.Mutate(i => i.DrawPolygon(c, 1f, path));

        }
        private PointF[] _indicator(float place, float s, float h, float W, float H, float v, bool vertical=true)
        {
            PointF[] path = null;
            if (vertical)
            {
                path = new PointF[] {
                    new PointF(place*s     ,H-(h+h*v)),
                    new PointF((place+1f)*s,H-(h+h*v)),
                    new PointF((place+1f)*s,H-h),
                    new PointF(place*s     ,H-h),
                };
            }
            else
            {
                path = new PointF[] {
                    new PointF(place*s    ,H-4*h),
                    new PointF((place+v)*s,H-4*h),
                    new PointF((place+v)*s,H-2*h),
                    new PointF(place*s    ,H-2*h),
                };
            }
            return (path);
        }
        private void RenderIndicators(Image<Rgba32> img, int W, int H)
        {
            float s = W / 40f;
            float h = H / 40f;
            Rgba32 color = new Rgba32(0, 0, 0);
            PointF[] poly = new PointF[] {
                new PointF(W,H),
                new PointF(W,H-5f*h),
                new PointF(0f,H-5f*h),
                new PointF(0f,H)
            };
            float v = np.sqrt(np.square(Car.Hull.LinearVelocity.X) + np.square(Car.Hull.LinearVelocity.Y));
            RenderIndicatorIfMin(img, v, _indicator(5f, s, h, W, H, 0.02f * v), new Rgba32(255, 255, 255));
            // ABS indicators
            for (int i = 0; i < 4; i++)
            {
                v = Car.Wheels[i].Omega;
                RenderIndicatorIfMin(img, v, _indicator(7f+i, s, h, W, H, 0.01f * v), i < 2 ? new Rgba32(0, 0, 255) : new Rgba32(51, 0, 255));
            }
            v = Car.Wheels[0].Joint.JointAngle;
            RenderIndicatorIfMin(img, v, _indicator(20f, s, h, W, H, -10f * v, false), new Rgba32(0, 255, 0));
            v = Car.Hull.AngularVelocity;
            RenderIndicatorIfMin(img, v, _indicator(30f, s, h, W, H, -0.8f * v, false), new Rgba32(255, 0, 0));
        }
        #endregion
        private void RenderRoad(Image<Rgba32> img, float zoom, Vector2 trans, float angle)
        {
            float bounds = PLAYFIELD;
            Vector2[] field = new Vector2[] {
                new Vector2(2f*bounds,2f*bounds),
                new Vector2(2f*bounds,0f),
                new Vector2(0f,0f),
                new Vector2(0f, 2f*bounds)
            };
            FillPoly(img, field, new Rgba32(102, 204, 102), zoom, trans, angle);
            float k = bounds / 20f;
            for (int x = 0; x < 40; x += 2)
            {
                for (int y = 0; y < 40; y += 2)
                {
                    Vector2[] poly = new Vector2[] {
                        new Vector2(k*x+k, k*y),
                        new Vector2(k*x  , k*y),
                        new Vector2(k*x  , k*y+k),
                        new Vector2(k*x+k, k*y+k)
                    };
                    FillPoly(img, poly, new Rgba32(102, 230, 102), zoom, trans, angle);
                }
            }
            Vector2 add = new Vector2(PLAYFIELD, PLAYFIELD);
            for (int i = 0; i < RoadPoly.Count; i++)
            {
                Vector2[] road = new Vector2[RoadPoly[i].Verts.Count];
                for (int j = 0; j < road.Length; j++)
                {
                    road[j] = RoadPoly[i].Verts[j] + add;
                }
                FillPoly(img, road, RoadPoly[i].Color, zoom, trans, angle);
            }
        }

        /// <summary>
        /// Renders the current state of the race
        /// </summary>
        /// <param name="mode">Rendering mode, either "human" or "state". State rendering will return the RGB array as the state in the observation.</param>
        /// <returns></returns>
        public override Image Render(string mode = "human")
        {
            if (_viewer == null && mode == "human")
            {
                lock (this)
                {
                    //to prevent double initalization.
                    if (_viewer == null)
                    {
                        if (_viewerFactory == null)
                            _viewerFactory = NullEnvViewer.Factory;
                        _viewer = _viewerFactory(WINDOW_W, WINDOW_H, "carracing").GetAwaiter().GetResult();
                    }
                }
            }
            Image<Rgba32> img = null;
            if (_LastRender == null)
            {
                // Define the buffer image for drawing
                img = new Image<Rgba32>(WINDOW_W, WINDOW_H);
                img.Mutate(i => i.BackgroundColor(new Rgba32(0, 0, 0))); // Space is black
                // Computing transforms
                float angle = -Car.Hull.Rotation;
                // Animating first second zoom.
                float zoom = 0.1f * SCALE * (float)Math.Max(1f - T, 0f) + ZOOM * SCALE * (float)Math.Min(T, 1f);
                float scroll_x = -(Car.Hull.Position.X + PLAYFIELD) * zoom;
                float scroll_y = -(Car.Hull.Position.Y + PLAYFIELD) * zoom;
                Vector2 trans = CarDynamics.RotateVec(new Vector2(scroll_x, scroll_y), angle) + new Vector2(WINDOW_W/2,WINDOW_H/4);
                RenderRoad(img, zoom, trans, angle);
                Car.Draw(img, zoom, trans, angle, mode != "state");
                // flip the surface?
                // Show stats
                RenderIndicators(img, WINDOW_W, WINDOW_H);
                // TODO: Display the reward as 42 pt text at (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)

                _LastRender = img;
            }
            if (mode == "state" && img != null)
            {
                // Convert the image to an NDArray
                _LastImageArray = new float[WINDOW_W, WINDOW_H, 3];
                for (int j = 0; j < WINDOW_H; j++)
                {
                    for (int i = 0; i < WINDOW_W; i++)
                    {
                        _LastImageArray[i, j, 0] = (float)img[i, j].R / 255f;
                        _LastImageArray[i, j, 1] = (float)img[i, j].G / 255f;
                        _LastImageArray[i, j, 2] = (float)img[i, j].B / 255f;
                    }
                }
            }
            // Clear out the last render so a new one is created
            if (mode == "human")
            {
                _viewer.Render(_LastRender);
                _LastRender = null;
            }
            return (img);
        }

        public override NDArray Reset()
        {
            _World = new World(new Vector2(0f, 0f));
            _World.ContactManager.BeginContact = new BeginContactDelegate(ContactListener.BeginContact);
            _World.ContactManager.EndContact = new EndContactDelegate(ContactListener.EndContact);
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
            Car = new CarDynamics(_World, Track[1], Track[2], Track[3]);
            return (Step(null).Observation);
        }

        public override Step Step(object action)
        {
            bool null_action = true;
            NDArray a = null;
            if (action != null)
            {
                null_action = false;
                a = (NDArray)action;
                Car.Steer(-1f * a[0]);
                Car.Gas(a[1]);
                Car.Brake(a[2]);
            }
            Car.Step(1f / FPS);
            SolverIterations si = new SolverIterations();
            si.PositionIterations = 2 * 30;
            si.VelocityIterations = 6 * 30;
            _World.Step(1f / FPS, ref si);
            T += 1f / FPS;

            // TODO: gather the 'state' pixels 

            float step_reward = 0f;
            bool done = false;

            if (!null_action) // First step without action, called from reset()
            {
                Reward -= 0.1f;
                //# We actually don't want to count fuel spent, we want car to be faster.
                //# self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
                Car.FuelSpent = 0f;
                step_reward -= PreviousReward;
                PreviousReward = Reward;
                if (TileVisitedCount == Track.Length || NewLap)
                {
                    done = true;
                }
                Vector2 pos = Car.Hull.Position;
                if (Math.Abs(pos.X) > PLAYFIELD || Math.Abs(pos.Y) > PLAYFIELD)
                {
                    done = true;
                    step_reward = -100f;
                }
            }
            Step step = new Step();
            step.Done = done;
            step.Reward = step_reward;
            if (StateOutputFormat == StateFormat.Pixels) {
                Image img = Render("state");
                step.Observation = _LastImageArray;
            }
            else {
                step.Observation = new float[] { Car.Hull.Position.X, Car.Hull.Position.Y, Car.Speed, Car.SteerAngle };
            }
            return step; 
        }
    }
}
