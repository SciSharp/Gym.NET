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
        /// <summary>
        ///     Gym.NET regression baselines of the PID episode below, keyed by seed and action mode: the step count and the
        ///     float32 bit pattern of the total reward, both asserted exactly. Every value was identical in Debug and Release on
        ///     .NET 8 and .NET 10 (docs/plans/NUMSHARP_MIGRATION_PLAN.md §9.2), so there is no tolerance and no
        ///     configuration guard. They are not Gymnasium goldens: the physics engine is Aether, not Box2D.
        /// </summary>
        /// <remarks>
        ///     Recorded against NumSharp master, whose <c>RandomState</c> is NumPy's MT19937. NumSharp.Lite's stream differed
        ///     (seed 1000 discrete was 245 steps / 35.515747 and was asserted in Debug builds only, B-37).
        /// </remarks>
        private static readonly Dictionary<(int Seed, bool Continuous), (int Steps, uint RewardBits)> _Baselines = new Dictionary<(int Seed, bool Continuous), (int Steps, uint RewardBits)> {
            [(0, false)] = (192, 0x4189de14u),    // 17.233437
            [(1, false)] = (251, 0x42131674u),    // 36.771927
            [(42, false)] = (182, 0xc23951deu),   // -46.32995
            [(1000, false)] = (230, 0x4223bd98u), // 40.93515
            [(0, true)] = (176, 0x41be6a44u),     // 23.801888
            [(1, true)] = (157, 0x422776fcu),     // 41.866196
            [(42, true)] = (221, 0x423c7858u),    // 47.117523
            [(1000, true)] = (355, 0x420d93f0u),  // 35.39447
        };
        private int _seed;
        private const int MAX_STEPS = 5000;
        private const bool VERBOSE = false;

        /// <summary>
        ///     Runs one PID-controlled episode and, when <paramref name="seed"/> is given, asserts its recorded baseline.
        /// </summary>
        /// <param name="factory">The viewer the env renders every step to.</param>
        /// <param name="seed">
        ///     The seed of the env's generator, asserted against <see cref="_Baselines"/>; <see langword="null"/> still seeds
        ///     with 0, so the run stays deterministic, but asserts only that the episode ends within <see cref="MAX_STEPS"/>.
        /// </param>
        /// <param name="continuous">Selects the continuous action space.</param>
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
                if (seed.HasValue)
                {
                    var (esteps, erewardBits) = _Baselines[(_seed, continuous)];
                    uint rewardBits = unchecked((uint)BitConverter.SingleToInt32Bits(total_reward));
                    Assert.AreEqual(esteps, steps, string.Format("Expected number of steps for seed {0} did not match.", _seed));
                    Assert.AreEqual(erewardBits, rewardBits, string.Format("Expected score for seed {0} did not match: expected {1:R} (0x{2:x8}), got {3:R} (0x{4:x8}).", _seed, BitConverter.Int32BitsToSingle(unchecked((int)erewardBits)), erewardBits, total_reward, rewardBits));
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

        /// <summary>
        ///     Pins all eight headless baselines (four seeds, both action modes), so any change to a draw, its order or the
        ///     episode arithmetic fails here in every build configuration. Seeds 0, 1, 42 and 1000 are those of
        ///     docs/plans/NUMSHARP_MIGRATION_PLAN.md §9.2.
        /// </summary>
        /// <param name="seed">The env generator's seed.</param>
        /// <param name="continuous">Selects the continuous action space.</param>
        [TestMethod]
        [DataRow(0, false)]
        [DataRow(1, false)]
        [DataRow(42, false)]
        [DataRow(1000, false)]
        [DataRow(0, true)]
        [DataRow(1, true)]
        [DataRow(42, true)]
        [DataRow(1000, true)]
        public void Run_NullEnv_Baseline(int seed, bool continuous)
        {
            Run(NullEnvViewer.Factory, seed, continuous);
        }
    }
}