#nullable enable
using System.Linq;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Utils {
    /// <summary>A tuple record for <see cref="GymRecordsTests"/>: an array and an int, Gym.NET's <c>(np.ndarray, int)</c>.</summary>
    /// <param name="Position">The first item.</param>
    /// <param name="Count">The second item.</param>
    [GymTuple]
    public sealed partial record PairObservation(NDArray Position, long Count);

    /// <summary>A dict record for <see cref="GymRecordsTests"/>: one key by snake_case, one by <c>[PyName]</c>.</summary>
    /// <param name="AgentPosition">The <c>agent_position</c> entry.</param>
    /// <param name="Goal">The <c>target</c> entry.</param>
    [GymDict]
    public partial record AgentObservation(NDArray AgentPosition, [property: PyName("target")] long Goal);

    /// <summary>
    ///     The source generator's records (DR-13, DR-34): options records that pass only what was set, construct the class and
    ///     refuse missing required arguments with Python's text; tuple and dict records that convert to and from the generic
    ///     forms, compare as Python does and print as Python does.
    /// </summary>
    [TestClass]
    public class GymRecordsTests {
        /// <summary>An options record passes only the properties that were set, by Python name, and <c>make</c> takes it.</summary>
        [TestMethod]
        public void OptionsRecord_PassesOnlySetProperties() {
            Assert.AreEqual(0, new CartPoleEnvOptions().ToKwargs().Count);
            Assert.AreEqual(new Info { ["render_mode"] = "rgb_array" }, new CartPoleEnvOptions { RenderMode = "rgb_array" }.ToKwargs());
            // Setting a value equal to the default still passes it: it was set.
            Assert.AreEqual(new Info { ["sutton_barto_reward"] = false }, new CartPoleEnvOptions { SuttonBartoReward = false }.ToKwargs());

            using IEnv env = Make("CartPole-v1", new CartPoleEnvOptions { RenderMode = "rgb_array" });
            Assert.AreEqual("rgb_array", env.RenderMode);
            Assert.AreEqual(new Info { ["render_mode"] = "rgb_array" }, env.Spec!.Kwargs);
        }

        /// <summary>The generated constructor builds the class from the record, defaults standing for unset properties.</summary>
        [TestMethod]
        public void OptionsRecord_GeneratedConstructor() {
            using (var env = new CartPoleEnv(new CartPoleEnvOptions { SuttonBartoReward = true })) {
                Assert.IsTrue(env.SuttonBartoReward);
                Assert.IsNull(env.RenderMode);
            }

            using var inner = new CartPoleEnv();
            using var limited = new TimeLimit<NDArray, NDArray>(inner, new TimeLimitOptions { MaxEpisodeSteps = 5 });
            Assert.AreEqual(5L, limited.MaxEpisodeSteps);
        }

        /// <summary>A required argument left unset is refused with Python's missing-argument text, before the base constructor runs.</summary>
        [TestMethod]
        public void OptionsRecord_MissingRequiredArgument() {
            using var inner = new CartPoleEnv();
            Assert.AreEqual("TimeLimit.__init__() missing 1 required positional argument: 'max_episode_steps'",
                Assert.ThrowsExactly<TypeError>(() => new TimeLimit<NDArray, NDArray>(inner, new TimeLimitOptions())).Message);
            Assert.ThrowsExactly<System.ArgumentNullException>(() => new TimeLimit<NDArray, NDArray>(inner, (TimeLimitOptions)null!));
        }

        /// <summary>A tuple record converts to and from <see langword="object"/>?[], compares as Python's tuples do, and prints as one.</summary>
        [TestMethod]
        public void TupleRecord_ConvertsComparesAndPrints() {
            using NDArray position = np.array(new[] { 1.5, 2.0 });
            using NDArray samePosition = np.array(new[] { 1.5, 2.0 });
            var record = new PairObservation(position, 3);

            object?[] items = record;
            Assert.AreEqual(2, items.Length);
            Assert.AreSame(position, items[0]);
            Assert.AreEqual(3L, items[1]);

            PairObservation back = new object?[] { samePosition, 3L };
            // Python's ==: arrays by value, so a record over equal arrays is equal.
            Assert.AreEqual(record, back);
            Assert.AreEqual(record.GetHashCode(), back.GetHashCode());
            Assert.AreNotEqual(record, new PairObservation(position, 4));
            Assert.AreEqual("(array([1.5, 2. ]), 3)", record.ToString());

            Assert.AreEqual("expected 2 items, got 3", Assert.ThrowsExactly<ValueError>(() => {
                PairObservation _ = new object?[] { position, 3L, 4L };
            }).Message);
        }

        /// <summary>
        ///     A tuple record is an <see cref="System.Runtime.CompilerServices.ITuple"/> holding its members, so code that only
        ///     sees an <see langword="object"/> reads it as a tuple: a <c>Tuple</c> space contains it, and Python's <c>==</c>
        ///     equals it to the generic form.
        /// </summary>
        [TestMethod]
        public void TupleRecord_IsAnITuple() {
            using NDArray position = np.array(new[] { 0.5f, 0.25f });
            var record = new PairObservation(position, 3);
            System.Runtime.CompilerServices.ITuple tuple = record;
            Assert.AreEqual(2, tuple.Length);
            Assert.AreSame(position, tuple[0]);
            Assert.AreEqual(3L, tuple[1]);
            Assert.ThrowsExactly<System.IndexOutOfRangeException>(() => tuple[2]);

            using var space = new Gym.Spaces.Tuple(new Gym.Spaces.Box(0.0, 1.0, new Shape(2), np.float32), new Gym.Spaces.Discrete(5));
            Assert.IsTrue(space.Contains(record), "A Tuple space contains the record as it contains its generic form.");
            Assert.IsFalse(space.Contains(new PairObservation(position, 7)), "...and checks its items.");
            Assert.IsTrue(PyEquality.AreEqual(record, new object?[] { position, 3L }), "Python's == between the record and the generic form.");
        }

        /// <summary>A dict record converts to and from <see cref="Info"/> by Python names, compares as dicts do, and prints as one.</summary>
        [TestMethod]
        public void DictRecord_ConvertsComparesAndPrints() {
            using NDArray position = np.array(new[] { 0L, 1L });
            var record = new AgentObservation(position, 7);

            Info dict = record;
            CollectionAssert.AreEqual(new[] { "agent_position", "target" }, dict.Keys.ToArray());
            Assert.AreEqual("{'agent_position': array([0, 1]), 'target': 7}", record.ToString());

            // Key order does not matter to a dict.
            AgentObservation back = new Info { ["target"] = 7L, ["agent_position"] = position };
            Assert.AreEqual(record, back);

            Assert.ThrowsExactly<KeyError>(() => {
                AgentObservation _ = new Info { ["agent_position"] = position };
            });
        }
    }
}
