#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs {
    /// <summary>
    ///     Pins the hand-written <see cref="RenderMode"/> twin constructors of DR-84 and DR-93: every env and vector env of
    ///     Gym.NET.Environments whose canonical constructor takes <c>string? renderMode</c> has a twin taking the enum first,
    ///     with the canonical constructor's other parameters unchanged, and both build the same env.
    /// </summary>
    [TestClass]
    public class RenderModeTwinTests {
        /// <summary>The public classes of Gym.NET.Environments with a public constructor taking <c>string? renderMode</c>.</summary>
        /// <returns>The classes, lazily, in the assembly's type order.</returns>
        private static IEnumerable<Type> ClassesWithARenderMode() =>
            typeof(CartPoleEnv).Assembly.GetTypes()
                .Where(t => t.IsPublic && !t.IsAbstract)
                .Where(t => Canonical(t) is not null);

        /// <summary>The canonical constructor: public, with a <c>string</c> parameter named <c>renderMode</c>.</summary>
        /// <param name="type">The class.</param>
        /// <returns>The constructor, or <see langword="null"/>.</returns>
        private static ConstructorInfo? Canonical(Type type) =>
            type.GetConstructors().FirstOrDefault(c => c.GetParameters().Any(p => p.Name == "renderMode" && p.ParameterType == typeof(string)));

        /// <summary>The twin: public, its first parameter a <see cref="RenderMode"/> named <c>renderMode</c>.</summary>
        /// <param name="type">The class.</param>
        /// <returns>The constructor, or <see langword="null"/>.</returns>
        private static ConstructorInfo? Twin(Type type) =>
            type.GetConstructors().FirstOrDefault(c => c.GetParameters() is { Length: > 0 } p && p[0].ParameterType == typeof(RenderMode) && p[0].Name == "renderMode");

        /// <summary>
        ///     Every such class has its twin, whose parameters after the mode are the canonical constructor's other parameters
        ///     in order, with the same names, types and defaults: a twin cannot drift from the constructor it stands beside.
        /// </summary>
        [TestMethod]
        public void EveryEnvWithARenderMode_HasItsTwin_WithTheSameOtherParameters() {
            Type[] classes = ClassesWithARenderMode().ToArray();
            Assert.IsTrue(classes.Length >= 34, $"Expected the 33 envs with a render mode and CartPoleVectorEnv, found {classes.Length}.");
            foreach (Type type in classes) {
                ParameterInfo[] canonical = Canonical(type)!.GetParameters().Where(p => p.Name != "renderMode").ToArray();
                ConstructorInfo? twin = Twin(type);
                Assert.IsNotNull(twin, $"{type.Name} has no RenderMode twin.");
                Assert.IsFalse(twin.GetParameters()[0].HasDefaultValue, $"{type.Name}: the twin's mode must be required, or new {type.Name}() turns ambiguous.");
                ParameterInfo[] rest = twin.GetParameters().Skip(1).ToArray();
                CollectionAssert.AreEqual(canonical.Select(p => p.Name).ToArray(), rest.Select(p => p.Name).ToArray(), $"{type.Name}: parameter names.");
                for (int i = 0; i < rest.Length; i++) {
                    Assert.AreEqual(canonical[i].ParameterType, rest[i].ParameterType, $"{type.Name}.{rest[i].Name}: type.");
                    Assert.AreEqual(canonical[i].HasDefaultValue, rest[i].HasDefaultValue, $"{type.Name}.{rest[i].Name}: optional.");
                    if (canonical[i].HasDefaultValue)
                        Assert.AreEqual(canonical[i].DefaultValue, rest[i].DefaultValue, $"{type.Name}.{rest[i].Name}: default.");
                }
            }
        }

        /// <summary>
        ///     Each twin builds the env the canonical constructor builds from the mode's string: its <c>RenderMode</c> is
        ///     Gymnasium's string, and it renders as that string does. Pusher-v4 refuses in both, as Gymnasium's does on MuJoCo 3+.
        /// </summary>
        /// <exception cref="AssertFailedException">Thrown when a class with a render mode builds neither an env nor a vector env, or a check fails.</exception>
        [TestMethod]
        public void EveryTwin_BuildsTheEnvOfItsString() {
            foreach (Type type in ClassesWithARenderMode()) {
                ConstructorInfo twin = Twin(type)!;
                object?[] arguments = twin.GetParameters().Select((p, i) => i == 0 ? RenderMode.RgbArray : p.DefaultValue is DBNull ? null : p.DefaultValue).ToArray();
                if (type.Name == "PusherEnvV4") {
                    var refused = Assert.ThrowsExactly<TargetInvocationException>(() => twin.Invoke(arguments));
                    Assert.IsInstanceOfType<ImportError>(refused.InnerException);
                    continue;
                }

                object built = twin.Invoke(arguments);
                try {
                    string? mode = built switch {
                        IEnv env => env.RenderMode,
                        IVectorEnv vector => vector.RenderMode,
                        _ => throw new AssertFailedException($"{type.Name} is neither an env nor a vector env."),
                    };
                    Assert.AreEqual(RenderModes.RgbArray, mode, $"{type.Name}: the twin passes Gymnasium's string on.");
                } finally {
                    (built as IDisposable)?.Dispose();
                }
            }
        }

        /// <summary>
        ///     Both spellings build the same CartPole: equal frames after the same seed, so the twin is the canonical constructor
        ///     with the enum's string and nothing else.
        /// </summary>
        [TestMethod]
        public void CartPole_EnumAndString_RenderTheSameFrame() {
            using var fromEnum = new CartPoleEnv(RenderMode.RgbArray);
            using var fromString = new CartPoleEnv(renderMode: "rgb_array");
            using (fromEnum.Reset(seed: 3))
            using (fromString.Reset(seed: 3))
            using (RenderResult a = fromEnum.Render())
            using (RenderResult b = fromString.Render())
                Assert.AreEqual(a, b);

            using var plain = new CartPoleEnv();
            Assert.IsNull(plain.RenderMode, "new CartPoleEnv() is still the constructor without a mode.");
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new CartPoleEnv((RenderMode)99));
        }

        /// <summary>
        ///     <c>make</c> takes the enum in kwargs too, and records Gymnasium's string in the spec (DR-84: kwargs and specs keep
        ///     the strings), as Gymnasium's <c>make(..., render_mode="rgb_array")</c> does.
        /// </summary>
        [TestMethod]
        public void Make_WithTheEnumInKwargs_RecordsTheString() {
            using IEnv env = Make("CartPole-v1", new { render_mode = RenderMode.RgbArray });
            Assert.AreEqual("rgb_array", env.RenderMode);
            Assert.AreEqual("rgb_array", env.Spec!.Kwargs["render_mode"]);
            Assert.AreEqual("rgb_array", env.Unwrapped.Spec!.Kwargs["render_mode"]);
        }
    }
}
