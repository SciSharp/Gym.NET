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
using NumSharp;

namespace Gym.Tests.Envs.Aether
{
    [TestClass]
    public class LunarLanderEnvironment
    {
        [TestMethod]
        public void Run()
        {
            LunarLanderEnv env = new LunarLanderEnv(AvaloniaEnvViewer.Factory);
            // Run a PID test
            float total_reward = 0f;
            int steps = 0;
            NDArray state = env.Reset();
            while (true)
            {
                int a = PID(env, (float[])state);
                var (observation, reward, done, information) = env.Step(a);
                total_reward += reward;
                steps++;
                if (steps % 1000 == 0)
                {
                    System.Diagnostics.Debug.WriteLine("{0}: a={1}, reward={2}, total={3}", steps, a, reward, total_reward);
                }
                if (done || steps > 200000)
                {
                    break;
                }
                state = observation;
                env.Render();
            }
            Assert.IsTrue(steps < 200000, "Too many steps.");
            System.Diagnostics.Debug.WriteLine("Total reward: {0} in {1} steps.", total_reward, steps);
        }

        private int PID(LunarLanderEnv env, float[] s)
        {
            float angle_targ = s[0] * 0.5f + s[2] * 1f; // Angle should point at target center
            if (angle_targ > 0.4f)
            {
                angle_targ = 0.4f;
            }
            if (angle_targ < -0.4)
            {
                angle_targ = -0.4f;
            }
            float hover_targ = 0.55f * np.abs(s[0]);
            float angle_todo = (angle_targ - s[4]) * 0.5f - s[5] * 1f;
            float hover_todo = (hover_targ - s[1]) * 0.5f - s[3] * 0.5f;

            if (s[6] > 0f || s[7] > 0f)
            {
                angle_todo = 0f;
                hover_todo = -s[3] * 0.5f;
            }
            int a = 0;
            if (hover_todo > Math.Abs(angle_todo) && hover_todo > 0.05f)
            {
                a = 2;
            }
            else if (angle_todo < -0.05f)
            {
                a = 3;
            }
            else if (angle_todo > 0.05f)
            {
                a = 1;
            }
            return a;
        }
    }
}
