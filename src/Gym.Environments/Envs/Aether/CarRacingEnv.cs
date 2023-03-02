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

namespace Gym.Environments.Envs.Aether
{
    /// <summary>
    ///## Description
    ///The easiest control task to learn from pixels - a top-down
    ///racing environment. The generated track is random every episode.

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

    ///A top-down 96x96 RGB image of the car and race track.

    ///## Rewards
    ///The reward is -0.1 every frame and +1000/N for every track tile visited,
    ///where N is the total number of tiles visited in the track. For example,
    ///if you have finished in 732 frames, your reward is
    ///1000 - 0.1*732 = 926.8 points.

    ///## Starting State
    ///The car starts at rest in the center of the road.

    ///## Episode Termination
    ///The episode finishes when all the tiles are visited. The car can also go
    ///outside the playfield - that is, far off the track, in which case it will
    ///receive -100 reward and die.

    ///## Arguments
    ///`lap_complete_percent` dictates the percentage of tiles that must be visited by
    ///the agent before a lap is considered complete.

    ///Passing `domain_randomize=True` enables the domain randomized variant of the environment.
    ///In this scenario, the background and track colours are different on every reset.

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
    ///- v1: Change track completion logic and add domain randomization (0.24.0)
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

        private IEnvironmentViewerFactoryDelegate _viewerFactory;
        private IEnvViewer _viewer;
        private bool ContinuousMode { get; set; } = true;
        private bool DomainRandomize { get; set; } = false;
        private float LapCompletePercent { get; set; } = 0.95f;
        private bool Verbose { get; set; } = false;

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

        public CarRacingEnv(IEnvironmentViewerFactoryDelegate viewerFactory, float lap_complete_percent = 0.95f, bool domain_randomize = false, bool continuous = true, bool verbose = false)
        {
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
        }

        public override void Close()
        {
            throw new NotImplementedException();
        }

        public override Image Render(string mode = "human")
        {
            throw new NotImplementedException();
        }

        public override NDArray Reset()
        {
            throw new NotImplementedException();
        }

        public override void Seed(int seed)
        {
            throw new NotImplementedException();
        }

        public override Step Step(object action)
        {
            throw new NotImplementedException();
        }
    }
}
