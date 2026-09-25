#nullable enable
using System;
using System.Collections.Generic;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Utils {
    /// <summary>
    ///     Pins <see cref="PyAttributes"/>: Python names (snake_case or <see cref="PyNameAttribute"/>), run-time attributes,
    ///     and Python's <c>AttributeError</c>/<c>TypeError</c> messages.
    /// </summary>
    [TestClass]
    public class PyAttributesTests {
        /// <summary>The naming rule read backwards (§3.8), acronyms included.</summary>
        [TestMethod]
        public void SnakeCase_ReadsThePascalCaseRuleBackwards() {
            Assert.AreEqual("np_random_seed", PyAttributes.SnakeCase("NpRandomSeed"));
            Assert.AreEqual("x_threshold", PyAttributes.SnakeCase("XThreshold"));
            Assert.AreEqual("rgb_array", PyAttributes.SnakeCase("RGBArray"));
            Assert.AreEqual("gravity", PyAttributes.SnakeCase("Gravity"));
            Assert.AreEqual("steps_beyond_terminated", PyAttributes.SnakeCase("StepsBeyondTerminated"));
            Assert.AreEqual("render_fps2", PyAttributes.SnakeCase("RenderFps2"));
        }

        /// <summary>Public members by snake_case name, non-public ones only with <see cref="PyNameAttribute"/>; missing names raise Python's error.</summary>
        [TestMethod]
        public void GetAttr_FindsDeclaredAndRunTimeAttributes() {
            var probe = new Probe();

            Assert.AreEqual(1.5, PyAttributes.GetAttr(probe, "force_mag"));
            Assert.AreEqual("hidden", PyAttributes.GetAttr(probe, "_secret"));
            Assert.AreEqual(3, PyAttributes.GetAttr(probe, "counter"));
            Assert.IsFalse(PyAttributes.HasAttr(probe, "_private_without_name"));
            Assert.AreEqual("'Probe' object has no attribute 'missing'", Assert.ThrowsExactly<AttributeError>(() => PyAttributes.GetAttr(probe, "missing")).Message);
            Assert.IsFalse(PyAttributes.HasAttr(probe, "raises"), "A getter that raises AttributeError makes the attribute absent, as hasattr sees it.");
            Assert.ThrowsExactly<InvalidOperationException>(() => PyAttributes.HasAttr(probe, "throws"), "Any other exception propagates.");
        }

        /// <summary>Assignments follow Python: read-only attributes refuse any value, numbers convert, other mismatches raise <c>TypeError</c>.</summary>
        [TestMethod]
        public void SetAttr_FollowsPython() {
            var probe = new Probe();

            PyAttributes.SetAttr(probe, "force_mag", 2);
            Assert.AreEqual(2.0, probe.ForceMag);
            PyAttributes.SetAttr(probe, "_secret", "changed");
            Assert.AreEqual("changed", PyAttributes.GetAttr(probe, "_secret"));

            Assert.AreEqual("property 'read_only' of 'Probe' object has no setter", Assert.ThrowsExactly<AttributeError>(() => PyAttributes.SetAttr(probe, "read_only", 1)).Message);
            Assert.AreEqual("property 'counter' of 'Probe' object has no setter", Assert.ThrowsExactly<AttributeError>(() => PyAttributes.SetAttr(probe, "counter", 4)).Message);
            Assert.ThrowsExactly<TypeError>(() => PyAttributes.SetAttr(probe, "force_mag", "fast"));
            Assert.ThrowsExactly<TypeError>(() => PyAttributes.SetAttr(probe, "force_mag", null));

            PyAttributes.SetAttr(probe, "created", true);
            Assert.AreEqual(true, PyAttributes.GetAttr(probe, "created"));
            Assert.IsTrue(PyAttributes.HasAttr(probe, "created"));

            Assert.AreEqual("'Plain' object has no attribute 'created'", Assert.ThrowsExactly<AttributeError>(() => PyAttributes.SetAttr(new Plain(), "created", 1)).Message);
        }

        /// <summary>Python prints a class by its name, without .NET's generic arity.</summary>
        [TestMethod]
        public void PyTypeName_DropsTheGenericArity() {
            Assert.AreEqual("List", PyAttributes.PyTypeName(typeof(List<int>)));
            Assert.AreEqual("Probe", PyAttributes.PyTypeName(typeof(Probe)));
        }

        /// <summary>A class with every kind of attribute member.</summary>
        private sealed class Probe : IDynamicAttributes {
            /// <summary>Found only through its <see cref="PyNameAttribute"/>.</summary>
            [PyName("_secret")]
            private string _secret = "hidden";

            /// <summary>Not an attribute: non-public without a Python name.</summary>
            private int _privateWithoutName = 7;

            /// <summary>A settable float attribute, <c>force_mag</c>.</summary>
            public double ForceMag { get; set; } = 1.5;

            /// <summary>A read-only attribute, <c>read_only</c>.</summary>
            public int ReadOnly => _privateWithoutName;

            /// <summary>A readonly field, which Python would model as a property without a setter.</summary>
            public readonly int Counter = 3;

            /// <summary>A getter that raises <see cref="AttributeError"/>, like a wrapper's <c>_np_random</c>.</summary>
            public int Raises => throw new AttributeError("no");

            /// <summary>A getter that raises something else.</summary>
            public int Throws => throw new InvalidOperationException("boom");

            /// <summary>The run-time attributes.</summary>
            IDictionary<string, object?> IDynamicAttributes.DynamicAttributes { get; } = new Dictionary<string, object?>();

            /// <summary>Keeps the private field used, so the compiler does not flag it.</summary>
            /// <returns>The field's value.</returns>
            public string Secret() => _secret;
        }

        /// <summary>A class that cannot hold run-time attributes.</summary>
        private sealed class Plain { }
    }
}
