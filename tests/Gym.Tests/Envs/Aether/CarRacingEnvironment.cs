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

namespace Gym.Tests.Envs.Aether
{
    [TestClass]
    public class CarRacingEnvironment
    {
        private Dictionary<int, float> _ExpectedScoreForRandomSeed = new Dictionary<int, float>();
        private Dictionary<int, int> _ExpectedStepsForRandomSeed = new Dictionary<int, int>();
        private const int MAX_STEPS = 5000;
        private const bool VERBOSE = true;
        public void Run(IEnvironmentViewerFactoryDelegate factory, NumPyRandom random_state, bool continuous = false)
        {
            CarRacingEnv env = new CarRacingEnv(factory, random_state: random_state, continuous: continuous, verbose : VERBOSE);
            try
            {
                // Run a PID test
                float total_reward = 0f;
                int steps = 0;
                NDArray state = env.Reset();
                while (true)
                {
                    object a = Driver(env, (float[])state);
                    var (observation, reward, done, information) = env.Step(a);
                    total_reward += reward;
                    steps++;
                    if (VERBOSE && (steps % 1000 == 0))
                    {
                        System.Diagnostics.Debug.WriteLine("{0}: a={1}, reward={2}, total={3}", steps, a, reward, total_reward);
                    }

                    if (done || steps > MAX_STEPS)
                    {
                        break;
                    }

                    state = observation;
                    env.Render();
                }
                if (random_state != null)
                {
                    float escore = _ExpectedScoreForRandomSeed[random_state.Seed];
                    int esteps = _ExpectedStepsForRandomSeed[random_state.Seed];
                    Assert.AreEqual(esteps, steps, string.Format("Expected number of steps for seed {0} did not match.", random_state.Seed));
                    Assert.AreEqual(escore, total_reward, 1e-5f, string.Format("Expected score for seed {0} did not match.", random_state.Seed));
                }
                Assert.IsTrue(steps < MAX_STEPS, "Too many steps.");
                System.Diagnostics.Debug.WriteLine("Total reward: {0} in {1} steps.", total_reward, steps);
            }
            finally
            {
                env.CloseEnvironment();
            }
        }
        private object Driver(CarRacingEnv env, float[] s)
        {
            return (0);
        }
        [TestMethod]
        public void Run_Discrete_WinFormEnv()
        {
            Run(WinFormEnvViewer.Factory, null);
        }

        [TestMethod]
        public void Run_Discrete_AvaloniaEnv()
        {
            Run(AvaloniaEnvViewer.Factory, null);
        }

        [TestMethod]
        public void Run_Discrete_NullEnv()
        {
            Run(NullEnvViewer.Factory, null);
        }

        [TestMethod]
        public void Run_Continuous_WinFormEnv()
        {
            Run(WinFormEnvViewer.Factory, null, true);
        }

        [TestMethod]
        public void Run_Continuous_AvaloniaEnv()
        {
            Run(AvaloniaEnvViewer.Factory, null, true);
        }

        [TestMethod]
        public void Run_Continuous_NullEnv()
        {
            Run(NullEnvViewer.Factory, null, true);
        }

        [TestMethod]
        public void Run_Discrete_WinFormEnv_ConsistencyCheck()
        {
            Run(WinFormEnvViewer.Factory, np.random.RandomState(1000));
        }

        [TestMethod]
        public void Run_Discrete_AvaloniaEnv_ConsistencyCheck()
        {
            Run(AvaloniaEnvViewer.Factory, np.random.RandomState(1000));
        }

        [TestMethod]
        public void Run_Discrete_NullEnv_ConsistencyCheck()
        {
            Run(NullEnvViewer.Factory, np.random.RandomState(1000));
        }
    }
}
