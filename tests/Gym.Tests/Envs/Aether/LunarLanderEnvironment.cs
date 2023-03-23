using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Rendering.Avalonia;
using Gym.Tests.Helpers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Gym.Environments.Envs.Aether;
using Gym.Rendering.WinForm;
using NumSharp;
using static System.Net.Mime.MediaTypeNames;
using System.Security.Policy;
using System.Windows.Documents;
using System.Xml.Linq;
using tainicom.Aether.Physics2D.Common;

namespace Gym.Tests.Envs.Aether {

    [TestClass]
    public class LunarLanderEnvironment {
        private Dictionary<int, float> _ExpectedScoreForRandomSeed = new Dictionary<int, float>();
        private Dictionary<int, int> _ExpectedStepsForRandomSeed = new Dictionary<int, int>();
        private int _seed;
        private const int MAX_STEPS = 5000;
        private const bool VERBOSE = false;

        public LunarLanderEnvironment()
        {
            // Total reward: 184.01764 in 1547 steps.
            _ExpectedScoreForRandomSeed[1000] = 184.01764f;
            _ExpectedStepsForRandomSeed[1000] = 1547;
        }


        public void Run(IEnvironmentViewerFactoryDelegate factory, int? seed, bool continuous = false) {
            _seed = seed ?? 0;
            LunarLanderEnv env = new LunarLanderEnv(factory, random_state: np.random.RandomState(_seed), continuous: continuous);
            env.Verbose = VERBOSE;
            try {
                // Run a PID test
                float total_reward = 0f;
                int steps = 0;
                NDArray state = env.Reset();
                while (true) {
                    object a = PID(env, (float[]) state);
                    var (observation, reward, done, information) = env.Step(a);
                    total_reward += reward;
                    steps++;
                    if (VERBOSE && (steps % 1000 == 0)) {
                        System.Diagnostics.Debug.WriteLine("{0}: a={1}, reward={2}, total={3}", steps, a, reward, total_reward);
                    }

                    if (done || steps > MAX_STEPS) {
                        break;
                    }

                    state = observation;
                    env.Render();
                }
                if (_seed != 0)
                {
                    #if DEBUG
                    float escore = _ExpectedScoreForRandomSeed[_seed];
                    int esteps = _ExpectedStepsForRandomSeed[_seed];
                    Assert.AreEqual(esteps, steps, string.Format("Expected number of steps for seed {0} did not match.", _seed));
                    Assert.AreEqual(escore, total_reward, 1e-5f, string.Format("Expected score for seed {0} did not match.", _seed));
                    #endif
                }
                Assert.IsTrue(steps < MAX_STEPS, "Too many steps.");
                System.Diagnostics.Debug.WriteLine("Total reward: {0} in {1} steps.", total_reward, steps);
            } finally {
                env.CloseEnvironment();
            }
        }

        /// <summary>
        //    The heuristic for
        //1. Testing
        //2. Demonstration rollout.

        //Args:
        //    env: The environment
        //    s (list): The state. Attributes:
        //        s[0] is the horizontal coordinate
        //        s[1] is the vertical coordinate
        //        s[2] is the horizontal speed
        //        s[3] is the vertical speed
        //        s[4] is the angle
        //        s[5] is the angular speed
        //        s[6] 1 if first leg has contact, else 0
        //        s[7] 1 if second leg has contact, else 0

        //Returns:
        //     a: The heuristic to be fed into the step function defined above to determine the next step and reward.
        /// </summary>
        /// <param name="env">The current lunar lander environment.</param>
        /// <param name="s">The state</param>
        /// <returns></returns>
        private object PID(LunarLanderEnv env, float[] s) {
            object action = null;
            float angle_targ = s[0] * 0.5f + s[2] * 1f; // Angle should point at target center
            if (angle_targ > 0.4f) {
                angle_targ = 0.4f; // more than 0.4 radians (22 degrees) is bad
            }

            if (angle_targ < -0.4) {
                angle_targ = -0.4f;
            }

            float hover_targ = 0.55f * Math.Abs(s[0]); // target y should be proportional to horizontal offset
            float angle_todo = (angle_targ - s[4]) * 0.5f - s[5] * 1f;
            float hover_todo = (hover_targ - s[1]) * 0.5f - s[3] * 0.5f;

            // override to reduce fall speed, that's all we need after contact
            if (s[6] > 0f || s[7] > 0f) {
                // legs have contact
                angle_todo = 0f;
                hover_todo = -s[3] * 0.5f;
            }

            if (env.ContinuousMode)
            {
                NDArray a = np.array(new float[] { hover_todo * 20f - 1f, -angle_todo * 20f });
                action = np.clip(a, -1f, 1f);
            }
            else
            {
                LunarLanderDiscreteActions a = LunarLanderDiscreteActions.Nothing;
                if (hover_todo > Math.Abs(angle_todo) && hover_todo > 0.05f)
                {
                    a = LunarLanderDiscreteActions.FireMainEngine;
                }
                else if (angle_todo < -0.05f)
                {
                    a = LunarLanderDiscreteActions.FireRightThruster;
                }
                else if (angle_todo > 0.05f)
                {
                    a = LunarLanderDiscreteActions.FireLeftThruster;
                }
                action = (int)a;
            }
            if(VERBOSE) 
                System.Diagnostics.Debug.WriteLine("PID: hover={0}, hover target={1}, approach angle={2}, action={3}.", hover_todo, hover_targ, angle_todo, action);

            return action;
        }

        [TestCleanup]
        public void Cleanup() {
            StaticAvaloniaApp.Shutdown();
        }

        [TestMethod]
        public void Run_Discrete_WinFormEnv() {
            Run(WinFormEnvViewer.Factory,null);
        }

        [TestMethod]
        public void Run_Discrete_AvaloniaEnv() {
            Run(AvaloniaEnvViewer.Factory,null);
        }

        [TestMethod]
        public void Run_Discrete_NullEnv() {
            Run(NullEnvViewer.Factory,null);
        }

        [TestMethod]
        public void Run_Continuous_WinFormEnv()
        {
            Run(WinFormEnvViewer.Factory, null, true);
        }

        [TestMethod]
        public void Run_Continuous_AvaloniaEnv()
        {
            Run(AvaloniaEnvViewer.Factory, null,true);
        }

        [TestMethod]
        public async Task Run_TwoInstances_Continuous_AvaloniaEnv()
        {
            var task1 = Task.Run(() => Run(AvaloniaEnvViewer.Factory, null,true));
            var task2 = Task.Run(() => Run(AvaloniaEnvViewer.Factory, null,true));
            await Task.WhenAll(task1, task2);
        }

        [TestMethod]
        public void Run_Continuous_NullEnv()
        {
            Run(NullEnvViewer.Factory, null,true);
        }

        [TestMethod]
        public void Run_Discrete_WinFormEnv_ConsistencyCheck()
        {
            Run(WinFormEnvViewer.Factory, 1000);
        }

        [TestMethod]
        public void Run_Discrete_AvaloniaEnv_ConsistencyCheck()
        {
            Run(AvaloniaEnvViewer.Factory, 1000);
        }

        [TestMethod]
        public void Run_Discrete_NullEnv_ConsistencyCheck()
        {
            Run(NullEnvViewer.Factory, 1000);
        }
    }
}