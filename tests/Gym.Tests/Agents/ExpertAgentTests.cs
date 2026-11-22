#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Gym.Agents;
using Gym.Envs;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Agents {
    /// <summary>
    ///     The published expert agents (DR-103) played in Gym.NET: the reader computes what PyTorch computes, and each agent
    ///     plays a whole expert episode on Gym.NET's own engines, as it did in the libraries it was trained on.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The episodes are the strongest end-to-end check the unit suite has on the MuJoCo engine and the Box2D port: a
    ///     Humanoid that walks 1,000 steps, a HumanoidStandup that stands, a hardcore walker that clears the whole course
    ///     all need contacts, actuators and integrators to behave as MuJoCo 4.0.0 and pybox2d 2.3.10 do over long horizons.
    ///     A broken engine sends an expert policy off its training distribution within a few dozen steps.
    ///     </para>
    ///     <para>
    ///     The minimums are the returns measured on 2026-11-22 (net10.0, the reference i9) less about 5%: an action can
    ///     differ in its last bit between CPUs and .NET runtimes (MathF.Tanh, the summation order), which shifts a long
    ///     episode's trajectory without making it worse, so an exact return would only pin the machine.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ExpertAgentTests {
        /// <summary>
        ///     Every agent with a seed and the return its episode must reach: (id, seed, minimum return, whether the
        ///     episode must run all its steps without terminating).
        /// </summary>
        /// <returns>The data rows.</returns>
        public static IEnumerable<object?[]> Episodes() => new object?[][] {
            new object?[] { "Ant-v5", 0, 5_400.0, true },
            new object?[] { "HalfCheetah-v5", 0, 16_800.0, true },
            new object?[] { "Hopper-v5", 0, 3_950.0, true },
            new object?[] { "Humanoid-v5", 0, 10_000.0, true },
            new object?[] { "HumanoidStandup-v5", 0, 380_000.0, true },
            new object?[] { "InvertedDoublePendulum-v5", 0, 9_300.0, true },
            new object?[] { "InvertedPendulum-v5", 0, 1_000.0, true },
            // Pusher's return is the distance cost summed over its 100 steps: -20 means the puck ended in the goal.
            new object?[] { "Pusher-v5", 7, -30.0, true },
            new object?[] { "Reacher-v5", 0, -6.0, true },
            new object?[] { "Swimmer-v5", 0, 340.0, true },
            new object?[] { "Walker2d-v5", 0, 6_600.0, true },
            // The hardcore course ends in termination either way: at the finish (about 300) or with a fall (-100 on top
            // of what was earned), so a return near 300 is the finish.
            new object?[] { "BipedalWalkerHardcore-v3", 7, 290.0, false },
        };

        /// <summary>The agent ids, one row each.</summary>
        /// <returns>The data rows.</returns>
        public static IEnumerable<object?[]> Ids() => Episodes().Select(row => new[] { row[0] });

        /// <summary>
        ///     The reader computes PyTorch's actions on the four observations the import recorded, to a millionth of each
        ///     action's range (measured: at most 6e-7).
        /// </summary>
        /// <param name="id">The agent's id.</param>
        [TestMethod]
        [DynamicData(nameof(Ids))]
        public void ReaderMatchesPyTorch(string id) {
            ExpertPolicy agent = Load(id);
            Assert.AreEqual(id, agent.Id);
            Assert.HasCount(4, agent.Checks);
            double error = agent.CheckError();
            Assert.IsLessThan(1e-5, error, $"{id}: the reader's actions differ from PyTorch's by {error:E2} of the range");
        }

        /// <summary>
        ///     The agent plays one seeded episode in Gym.NET as an expert: its return reaches the minimum, and the locomotion
        ///     agents stay up for all their steps.
        /// </summary>
        /// <param name="id">The agent's id.</param>
        /// <param name="seed">The reset seed.</param>
        /// <param name="minimum">The return the episode must reach.</param>
        /// <param name="mustLast">Whether the episode must end by truncation, never by termination.</param>
        [TestMethod]
        [DynamicData(nameof(Episodes))]
        public void PlaysAnExpertEpisode(string id, int seed, double minimum, bool mustLast) {
            ExpertPolicy agent = Load(id);
            var (steps, total, terminated) = Play(agent, seed);
            Assert.IsGreaterThanOrEqualTo(minimum, total, $"{id} seed {seed}: return {total:0.0} in {steps} steps");
            if (mustLast)
                Assert.IsFalse(terminated, $"{id} seed {seed}: ended early at step {steps} (return {total:0.0})");
        }

        /// <summary>
        ///     Each file names its source: the Hugging Face repository, a pinned 40-character revision, the algorithm and the
        ///     licence note, as the maintainer's decision to commit them requires (DR-103).
        /// </summary>
        /// <param name="id">The agent's id.</param>
        [TestMethod]
        [DynamicData(nameof(Ids))]
        public void NamesItsSource(string id) {
            ExpertPolicy agent = Load(id);
            StringAssert.StartsWith(agent.Source["repository"]!.GetValue<string>(), "https://huggingface.co/");
            Assert.AreEqual(40, agent.Source["revision"]!.GetValue<string>().Length);
            Assert.IsFalse(string.IsNullOrEmpty(agent.Source["algorithm"]!.GetValue<string>()));
            StringAssert.Contains(agent.Source["licence"]!.GetValue<string>(), "DR-103");
        }

        /// <summary>Plays one episode to its end.</summary>
        /// <param name="agent">The agent.</param>
        /// <param name="seed">The reset seed.</param>
        /// <returns>The steps taken, the return, and whether it ended by termination.</returns>
        internal static (int Steps, double Return, bool Terminated) Play(ExpertPolicy agent, int seed) {
            using IEnv env = Make(agent.Id, new Dictionary<string, object?>(agent.MakeKwargs));
            object? observation = env.Reset(seed: seed).Observation;
            int steps = 0;
            double total = 0;
            bool terminated = false, truncated = false;
            while (!terminated && !truncated) {
                double[] values;
                using (NDArray d = ((NDArray)observation!).astype(NPTypeCode.Double))
                    values = d.ToArray<double>();
                using NDArray action = np.array(agent.Act(values));
                var (next, reward, term, trunc, _) = env.Step(action);
                (observation as IDisposable)?.Dispose();
                observation = next;
                total += reward;
                steps++;
                terminated = term;
                truncated = trunc;
            }
            (observation as IDisposable)?.Dispose();
            return (steps, total, terminated);
        }

        /// <summary>Reads an agent from the assembly, where the project embeds <c>Agents/&lt;id&gt;.json</c>.</summary>
        /// <param name="id">The agent's id.</param>
        /// <returns>The agent.</returns>
        /// <exception cref="InvalidOperationException">No agent is embedded for that id.</exception>
        internal static ExpertPolicy Load(string id) {
            string name = $"Gym.Tests.Agents.{id}.json";
            using Stream stream = typeof(ExpertAgentTests).Assembly.GetManifestResourceStream(name)
                                  ?? throw new InvalidOperationException($"no embedded agent '{name}'");
            using var reader = new StreamReader(stream);
            return ExpertPolicy.Parse(reader.ReadToEnd(), name);
        }
    }
}
