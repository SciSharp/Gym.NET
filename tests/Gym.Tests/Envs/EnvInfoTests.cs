#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments.Envs.Box2D;
using Gym.Environments.Envs.MuJoCo;
using Gym.Environments.Envs.ToyText;
using Gym.Envs;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs {
    /// <summary>
    ///     Pins the typed info records (DR-91, DR-94, DR-95): every built-in env class that sends info keys has its record,
    ///     which reads a reset's and a step's info without losing a key or a value, keeps the dict's own values for its
    ///     indexer, compares by its members, and drops the dict in a <c>with</c> copy.
    /// </summary>
    [TestClass]
    public class EnvInfoTests {
        /// <summary>Every id whose env sends info keys, with its record type (DR-95: one per env class, versions suffixed).</summary>
        public static IEnumerable<object[]> IdsAndRecords => new (string Id, Type Record)[] {
            ("FrozenLake-v1", typeof(FrozenLakeInfo)), ("FrozenLake8x8-v1", typeof(FrozenLakeInfo)), ("CliffWalking-v1", typeof(CliffWalkingInfo)),
            ("CliffWalkingSlippery-v1", typeof(CliffWalkingInfo)), ("Taxi-v4", typeof(TaxiInfo)), ("CarRacing-v3", typeof(CarRacingInfo)),
            ("Ant-v4", typeof(AntInfoV4)), ("Ant-v5", typeof(AntInfoV5)), ("HalfCheetah-v4", typeof(HalfCheetahInfoV4)), ("HalfCheetah-v5", typeof(HalfCheetahInfoV5)),
            ("Hopper-v4", typeof(HopperInfoV4)), ("Hopper-v5", typeof(HopperInfoV5)), ("Humanoid-v4", typeof(HumanoidInfoV4)), ("Humanoid-v5", typeof(HumanoidInfoV5)),
            ("HumanoidStandup-v4", typeof(HumanoidStandupInfoV4)), ("HumanoidStandup-v5", typeof(HumanoidStandupInfoV5)),
            ("InvertedDoublePendulum-v5", typeof(InvertedDoublePendulumInfoV5)), ("InvertedPendulum-v5", typeof(InvertedPendulumInfoV5)),
            ("Pusher-v5", typeof(PusherInfoV5)), ("Reacher-v4", typeof(ReacherInfoV4)), ("Reacher-v5", typeof(ReacherInfoV5)),
            ("Swimmer-v4", typeof(SwimmerInfoV4)), ("Swimmer-v5", typeof(SwimmerInfoV5)), ("Walker2d-v4", typeof(Walker2dInfoV4)), ("Walker2d-v5", typeof(Walker2dInfoV5)),
        }.Select(pair => new object[] { pair.Id, pair.Record });

        /// <summary>The record conversion of an info dict, through the generated implicit operator.</summary>
        /// <param name="record">The record type.</param>
        /// <param name="info">The dict.</param>
        /// <returns>The record.</returns>
        private static EnvInfo Convert(Type record, Info info) =>
            (EnvInfo)record.GetMethods().Single(m => m.Name == "op_Implicit" && m.ReturnType == record && m.GetParameters()[0].ParameterType == typeof(Info)).Invoke(null, new object[] { info })!;

        /// <summary>
        ///     A reset's and a few steps' infos convert to the env's record, and the record's members alone (a <c>with</c> copy,
        ///     which drops the dict) equal the info as Python dicts: every key the env sends is a member, under its Gymnasium
        ///     name, with its value. The record also keeps the dict, so its own <c>ToInfo</c> is the dict, key order included.
        /// </summary>
        /// <param name="id">The env id.</param>
        /// <param name="record">The record type.</param>
        [TestMethod]
        [DynamicData(nameof(IdsAndRecords))]
        public void EveryRecord_ReadsItsEnvsInfos_WithoutLosingAKey(string id, Type record) {
            using IEnv env = Make(id);
            env.ActionSpace.Seed(0);
            using (var reset = env.Reset(seed: 0))
                Check(reset.Info, "reset");
            for (int i = 0; i < 5; i++) {
                using var step = env.Step(env.ActionSpace.Sample());
                Check(step.Info, $"step {i}");
                if (step.Terminated || step.Truncated)
                    break;
            }

            void Check(Info info, string when) {
                EnvInfo typed = Convert(record, info);
                CollectionAssert.AreEqual(info.Keys.ToArray(), typed.ToInfo().Keys.ToArray(), $"{id} {when}: the record keeps the dict, key order included.");
                EnvInfo membersOnly = typed with { };
                Assert.IsTrue(membersOnly.PyEquals(info), $"{id} {when}: the members are the info: {membersOnly} vs {info}");
                Assert.AreEqual(typed, membersOnly, $"{id} {when}: equality compares the members, never the dict.");
            }
        }

        /// <summary>
        ///     HalfCheetah-v5's record: the reset's info holds <c>x_position</c> only, so the step-only members are null; a
        ///     step's fills them; the indexer gives the values as sent (the control cost as a float32 scalar), the members as
        ///     <see langword="double"/>s.
        /// </summary>
        [TestMethod]
        public void HalfCheetahV5_ResetAndStep() {
            using IEnv env = Make("HalfCheetah-v5");
            using (var reset = env.Reset(seed: 0)) {
                HalfCheetahInfoV5 info = reset.Info;
                Assert.AreEqual(reset.Info.Get<double>("x_position"), info.XPosition);
                Assert.IsNull(info.XVelocity, "A reset's info has no x_velocity.");
                Assert.IsNull(info.RewardCtrl);
                Assert.IsFalse(info.ContainsKey("x_velocity"));
            }

            using NDArray action = np.zeros(new Shape(6), np.float32);
            using var step = env.Step(action);
            HalfCheetahInfoV5 stepInfo = step.Info;
            Assert.IsNotNull(stepInfo.XVelocity);
            Assert.AreEqual(step.Info.Get<double>("reward_ctrl"), stepInfo.RewardCtrl);
            var raw = (NDArray)stepInfo["reward_ctrl"]!;
            Assert.AreEqual(NPTypeCode.Single, raw.typecode, "The indexer gives the value as the env sent it: float32 for a float32 action.");
            Assert.AreEqual((double)(NDArray)step.Info["x_position"]!, (double)(NDArray)stepInfo["x_position"]!, "The dict's access reads as on step.Info.");
            Assert.IsTrue(stepInfo.PyEquals(step.Info));
            Assert.AreEqual(step.Info.Count, stepInfo.ToInfo().Count);
        }

        /// <summary>
        ///     A <c>with</c> copy drops the dict inside: its indexer reads its own members, so a changed member never shows the
        ///     original's stale value; records of equal members are equal (<c>==</c>), whatever dicts they came from.
        /// </summary>
        [TestMethod]
        public void With_DropsTheDict_AndEqualityIsTheMembers() {
            using NDArray x = NDArray.Scalar(0.5);
            HalfCheetahInfoV5 fromDict = new Info { ["x_position"] = x };
            HalfCheetahInfoV5 fromCode = new HalfCheetahInfoV5(0.5, null, null, null);
            Assert.IsTrue(fromDict == fromCode, "Equal members, though one holds a 0-d array and the other a double.");
            Assert.AreEqual(fromDict.GetHashCode(), fromCode.GetHashCode());
            Assert.IsInstanceOfType<NDArray>(fromDict["x_position"], "Read from a dict: the dict's value.");
            Assert.IsInstanceOfType<double>(fromCode["x_position"], "Built in code: the member.");

            HalfCheetahInfoV5 moved = fromDict with { XPosition = 2.0 };
            Assert.AreEqual(2.0, moved["x_position"], "The copy reads its own member, not the original dict's 0.5.");
            Assert.IsFalse(moved == fromDict);
            Assert.AreEqual("{'x_position': 2.0}", moved.ToString());
        }

        /// <summary>A key that a non-nullable member needs is Python's <c>KeyError</c>; a nullable member's absent key is <see langword="null"/>.</summary>
        [TestMethod]
        public void MissingKeys_RequiredRaise_OptionalAreNull() {
            KeyError missing = Assert.ThrowsExactly<KeyError>(() => { HalfCheetahInfoV5 _ = new Info(); });
            Assert.AreEqual("'x_position'", missing.Message);
            CarRacingInfo quiet = new Info();
            Assert.IsNull(quiet.LapFinished);
            CarRacingInfo lap = new Info { ["lap_finished"] = true };
            Assert.AreEqual(true, lap.LapFinished);
            Assert.ThrowsExactly<ArgumentNullException>(() => { HalfCheetahInfoV5 _ = (Info)null!; });
        }

        /// <summary>
        ///     The toy_text records inherit <see cref="ToyTextInfo.Prob"/> (DR-95: inheriting as the envs' families do): a
        ///     reset's Python int 1 and a step's float both read as doubles; Taxi's action mask is the info's own array.
        /// </summary>
        [TestMethod]
        public void ToyText_ProbIsInherited_AndTaxisMaskIsShared() {
            using IEnv frozen = Make("FrozenLake-v1");
            using (var reset = frozen.Reset(seed: 0)) {
                ToyTextInfo info = (FrozenLakeInfo)reset.Info;
                Assert.AreEqual(1.0, info.Prob);
            }

            using IEnv taxi = Make("Taxi-v4");
            using var taxiReset = taxi.Reset(seed: 0);
            TaxiInfo taxiInfo = taxiReset.Info;
            Assert.AreSame(taxiReset.Info["action_mask"], taxiInfo.ActionMask, "Array members are shared with the step result.");
            Assert.AreEqual(1.0, taxiInfo.Prob);
        }
    }
}
