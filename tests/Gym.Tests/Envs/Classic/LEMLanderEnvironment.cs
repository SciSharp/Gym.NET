﻿using Gym.Environments;
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
        private float[] _BurnsTest1 = new float[] { 35.6f, 177f, 180.2f, 8.2f, 13.3f, 121, 3f, 133.6f, 68.2f, 124.10f, 147.5f, 91f, 133.9f, 199.1f, 184.2f, 145.8f, 178.9f, 30.3f, 101.7f, 69.4f, 152.6f, 58.1f, 74.2f, 187.1f, 42.4f, 170.10f, 154.8f, 24.9f, 189.7f, 65.8f, 106.2f, 74.6f, 140.9f, 128f, 37f, 190.5f, 161f, 104.6f, 86f, 160.4f, 197.6f, 96.8f, 86f, 180.4f, 180.5f, 121, 8f, 178.2f };
        private float[] _TimeTest1 = new float[] { 3f };
        private int MAX_STEPS = 100;
        public void Run(IEnvironmentViewerFactoryDelegate factory, NumPyRandom random_state, float[] burns, float[] time)
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
        }
        [TestMethod]
        public void Run_Test1_WinFormEnv()
        {
            Run(WinFormEnvViewer.Factory, null, _BurnsTest1, _TimeTest1);
        }

        [TestMethod]
        public void Run_Test1_AvaloniaEnv()
        {
            Run(AvaloniaEnvViewer.Factory, null, _BurnsTest1, _TimeTest1);
        }

        [TestMethod]
        public void Run_Test1_NullEnv()
        {
            Run(NullEnvViewer.Factory, null, _BurnsTest1, _TimeTest1);
        }

    }
}