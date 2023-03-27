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

namespace Gym.Tests.Envs.Classic
{
    [TestClass]
    public class LEMLanderEnvironment
    {
        private const bool VERBOSE = true;
        // Test1 : Fuel runs out
        private float[] _BurnsTest1 = new float[] { 35.6f, 177f, 180.2f, 8.2f, 13.3f, 121, 3f, 133.6f, 68.2f, 124.10f, 147.5f, 91f, 133.9f, 199.1f, 184.2f, 145.8f, 178.9f, 30.3f, 101.7f, 69.4f, 152.6f, 58.1f, 74.2f, 187.1f, 42.4f, 170.10f, 154.8f, 24.9f, 189.7f, 65.8f, 106.2f, 74.6f, 140.9f, 128f, 37f, 190.5f, 161f, 104.6f, 86f, 160.4f, 197.6f, 96.8f, 86f, 180.4f, 180.5f, 121, 8f, 178.2f };
        private float[] _TimeTest1 = new float[] { 3f };
        // Test2 : Crash
        private float[] _BurnsTest2 = new float[] { 64f, 195.8f, 121.3f, 120.5f, 82.8f, 68.9f, 41.2f, 63.7f, 47.2f, 25.1f, 16.5f, 12.7f, 31.5f, 73f, 143.1f, 69.7f, 51.8f, 12.5f, 183.7f, 50f, 15f, 15f, 12f, 12.5f, 12.5f, 57.95f, 0.05f };
        private float[] _TimeTest2 = new float[] { 10f };
        // Test4 : Good Landing
        private float[] _BurnsTest4 = new float[] { 164.8f,13.6f,96.1f,70.8f,64.5f,33.7f,29f,11.1f,20.3f,196f,86.5f,21.6f,27.6f,116.8f,162.8f,189f,28.9f,112.7f,110.1f,21.1f, 12.1f };
        private float[] _TimeTest4 = new float[] { 10f };
        // Test5 : Perfect Landing
        private float[] _BurnsTest5 = new float[] { 164.8f, 13.6f, 96.1f, 70.8f, 64.5f, 33.7f, 29f, 11.1f, 20.3f, 196f, 86.5f, 21.6f, 27.6f, 116.8f, 162.8f, 189f, 28.9f, 112.7f, 110.1f, 21.1f, 13.7f };
        private float[] _TimeTest5 = new float[] { 10f };
        // Test6 : So-So Landing
        private float[] _BurnsTest6 = new float[] { 164.8f, 13.6f, 96.1f, 70.8f, 64.5f, 33.7f, 29f, 11.1f, 20.3f, 196f, 86.5f, 21.6f, 27.6f, 116.8f, 162.8f, 189f, 28.9f, 112.7f, 110.1f, 23.1f, 11.5f, 7f, 2f, 0f };
        private float[] _TimeTest6 = new float[] { 10f };
        private int MAX_STEPS = 100;
        public LEMLanderEnv Run(IEnvironmentViewerFactoryDelegate factory, NumPyRandom random_state, float[] burns, float[] time)
        {
            LEMLanderEnv env = new LEMLanderEnv(factory, random_state: random_state);
            env.Verbose = VERBOSE;
            try
            {
                // Run a PID test
                float total_reward = 0f;
                int steps = 0;
                NDArray state = env.Reset();
                int ix = 0;
                while (true)
                {
                    object a = np.array(new float[] { burns[ix % burns.Length], time[ix % time.Length] });
                    ix = (ix + 1) % burns.Length;
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
                System.Diagnostics.Debug.WriteLine("Total reward: {0} in {1} steps.", total_reward, steps);
            }
            finally
            {
                env.CloseEnvironment();
            }
            return (env);
        }
        #region Test 1 - A Known Series of Burns
        [TestMethod]
        public void Run_Test1_WinFormEnv()
        {
            LEMLanderEnv env = Run(WinFormEnvViewer.Factory, null, _BurnsTest1, _TimeTest1);
            Assert.AreEqual(LanderStatus.FreeFall, env.Status);
        }

        [TestMethod]
        public void Run_Test1_AvaloniaEnv()
        {
            LEMLanderEnv env = Run(AvaloniaEnvViewer.Factory, null, _BurnsTest1, _TimeTest1);
            Assert.AreEqual(LanderStatus.FreeFall, env.Status);
        }

        [TestMethod]
        public void Run_Test1_NullEnv()
        {
            LEMLanderEnv env = Run(NullEnvViewer.Factory, null, _BurnsTest1, _TimeTest1);
            Assert.AreEqual(LanderStatus.FreeFall, env.Status);
        }
        #endregion

        #region Test 2 - Crash
        [TestMethod]
        public void Run_Test2_WinFormEnv()
        {
            LEMLanderEnv env = Run(WinFormEnvViewer.Factory, null, _BurnsTest2, _TimeTest2);
            Assert.AreEqual(LanderStatus.Crashed, env.Status);
        }

        [TestMethod]
        public void Run_Test2_AvaloniaEnv()
        {
            LEMLanderEnv env = Run(AvaloniaEnvViewer.Factory, null, _BurnsTest2, _TimeTest2);
            Assert.AreEqual(LanderStatus.Crashed, env.Status);
        }

        [TestMethod]
        public void Run_Test2_NullEnv()
        {
            LEMLanderEnv env = Run(NullEnvViewer.Factory, null, _BurnsTest2, _TimeTest2);
            Assert.AreEqual(LanderStatus.Crashed, env.Status);
        }
        #endregion

        #region Test 3 - Force The GUI Display
        [TestMethod]
        public void Run_Test3_WinFormEnv()
        {
            Run(WinFormEnvViewer.Factory, null, new float[] { 5f }, new float[] { 2f });
        }
        /// <summary>
        /// Test the avalonia viewer with the slow run to force the display of the GUI
        /// </summary>
        [TestMethod]
        public void Run_Test3_AvaloniaEnv()
        {
            Run(AvaloniaEnvViewer.Factory, null, new float[] { 5f }, new float[] { 2f });
        }

        [TestMethod]
        public void Run_Test3_NullEnv()
        {
            Run(NullEnvViewer.Factory, null, new float[] { 5f }, new float[] { 2f });
        }
        #endregion
        #region Test 4 - Good Landing Solution
        [TestMethod]
        public void Run_Test4_WinFormEnv()
        {
            LEMLanderEnv env = Run(WinFormEnvViewer.Factory, null, _BurnsTest4, _TimeTest4);
            Assert.AreEqual(LanderStatus.Landed, env.Status);
        }

        [TestMethod]
        public void Run_Test4_AvaloniaEnv()
        {
            LEMLanderEnv env = Run(AvaloniaEnvViewer.Factory, null, _BurnsTest4, _TimeTest4);
            Assert.AreEqual(LanderStatus.Landed, env.Status);
        }

        [TestMethod]
        public void Run_Test4_NullEnv()
        {
            LEMLanderEnv env = Run(NullEnvViewer.Factory, null, _BurnsTest4, _TimeTest4);
            Assert.AreEqual(LanderStatus.Landed, env.Status);
        }
        #endregion

        #region Test 5 - Perfect Landing Solution
        [TestMethod]
        public void Run_Test5_WinFormEnv()
        {
            LEMLanderEnv env = Run(WinFormEnvViewer.Factory, null, _BurnsTest5, _TimeTest5);
            Assert.AreEqual(LanderStatus.Landed, env.Status);
        }

        [TestMethod]
        public void Run_Test5_AvaloniaEnv()
        {
            LEMLanderEnv env = Run(AvaloniaEnvViewer.Factory, null, _BurnsTest5, _TimeTest5);
            Assert.AreEqual(LanderStatus.Landed, env.Status);
        }

        [TestMethod]
        public void Run_Test5_NullEnv()
        {
            LEMLanderEnv env = Run(NullEnvViewer.Factory, null, _BurnsTest5, _TimeTest5);
            Assert.AreEqual(LanderStatus.Landed, env.Status);
        }
        #endregion
        #region Test 6 - So-So Landing Solution
        [TestMethod]
        public void Run_Test6_WinFormEnv()
        {
            LEMLanderEnv env = Run(WinFormEnvViewer.Factory, null, _BurnsTest6, _TimeTest6);
            Assert.AreEqual(LanderStatus.Landed, env.Status);
        }

        [TestMethod]
        public void Run_Test6_AvaloniaEnv()
        {
            LEMLanderEnv env = Run(AvaloniaEnvViewer.Factory, null, _BurnsTest6, _TimeTest6);
            Assert.AreEqual(LanderStatus.Landed, env.Status);
        }

        [TestMethod]
        public void Run_Test6_NullEnv()
        {
            LEMLanderEnv env = Run(NullEnvViewer.Factory, null, _BurnsTest6, _TimeTest6);
            Assert.AreEqual(LanderStatus.Landed, env.Status);
        }
        #endregion
    }
}