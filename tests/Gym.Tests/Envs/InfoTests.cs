#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs {
    /// <summary>
    ///     Pins <see cref="Info"/> and <see cref="Metadata"/> to Python's <c>dict</c>: insertion order, <c>KeyError</c>, the
    ///     defaulting accessors, copies, <c>==</c> through <see cref="PyEquality"/>, and <c>repr</c>.
    /// </summary>
    [TestClass]
    public class InfoTests {
        /// <summary>Iteration follows insertion; re-assigning keeps the place; removing and re-adding moves the key to the end.</summary>
        [TestMethod]
        public void Order_IsInsertionOrder() {
            var info = new Info { ["b"] = 1, ["a"] = 2, ["c"] = 3 };
            info["a"] = 20;
            CollectionAssert.AreEqual(new[] { "b", "a", "c" }, info.Keys.ToArray());
            CollectionAssert.AreEqual(new object?[] { 1, 20, 3 }, info.Values.ToArray());

            info.Remove("b");
            info["b"] = 10;
            CollectionAssert.AreEqual(new[] { "a", "c", "b" }, info.Select(entry => entry.Key).ToArray());
            Assert.AreEqual(20, info["a"], "Removal re-indexes the keys after it.");
        }

        /// <summary>A missing key raises Python's <c>KeyError</c> with the key's repr; <c>get</c> and <c>pop</c> default instead.</summary>
        [TestMethod]
        public void MissingKey_RaisesKeyError_UnlessDefaulted() {
            var info = new Info { ["x"] = 1 };

            Assert.AreEqual("'missing'", Assert.ThrowsExactly<KeyError>(() => info["missing"]).Message);
            Assert.AreEqual("'missing'", Assert.ThrowsExactly<KeyError>(() => info.Pop("missing")).Message);
            Assert.IsNull(info.Get("missing"));
            Assert.AreEqual(5, info.Get("missing", 5));
            Assert.AreEqual(7, info.Pop("missing", 7));
            Assert.AreEqual(1, info.Pop("x"));
            Assert.AreEqual(0, info.Count);
            Assert.ThrowsExactly<ArgumentException>(() => new Info { { "k", 1 }, { "k", 2 } });
        }

        /// <summary><c>update</c> appends new keys in the other's order and replaces existing ones in place.</summary>
        [TestMethod]
        public void Update_KeepsPythonsOrder() {
            var info = new Info { ["a"] = 1, ["b"] = 2 };
            info.Update(new Info { ["c"] = 3, ["a"] = 10 });

            Assert.AreEqual("{'a': 10, 'b': 2, 'c': 3}", info.ToString());
            info.Update(info);
            Assert.AreEqual("{'a': 10, 'b': 2, 'c': 3}", info.ToString());
        }

        /// <summary>
        ///     <c>copy</c> shares values, <c>deepcopy</c> does not; both keep the concrete type, so a <see cref="Metadata"/>
        ///     copies to a <see cref="Metadata"/>.
        /// </summary>
        [TestMethod]
        public void Copies_ShallowShares_DeepDoesNot() {
            var list = new List<object?> { 1, 2 };
            using var array = np.array(new double[] { 1, 2 });
            var metadata = new Metadata { ["render_modes"] = new List<string> { "human" }, ["list"] = list, ["array"] = array };

            Info shallow = metadata.Copy();
            Info deep = metadata.DeepCopy();

            Assert.IsInstanceOfType<Metadata>(shallow);
            Assert.IsInstanceOfType<Metadata>(deep);
            Assert.AreSame(list, shallow["list"]);
            Assert.AreNotSame(list, deep["list"]);
            Assert.AreNotSame(array, deep["array"]);
            list.Add(3);
            array.SetDouble(10, 0);
            Assert.AreEqual(2, ((List<object?>)deep["list"]!).Count);
            Assert.AreEqual(1.0, ((NDArray)deep["array"]!).GetDouble(0));
            Assert.AreEqual(metadata, shallow);
        }

        /// <summary>
        ///     Python's dict <c>==</c>: order does not matter, numbers compare by value across types (<c>1 == 1.0 == True</c>),
        ///     lists and tuples compare element-wise and never equal each other, and arrays compare by value.
        /// </summary>
        [TestMethod]
        public void Equality_IsPythons() {
            Assert.AreEqual(new Info { ["a"] = 1, ["b"] = 2 }, new Info { ["b"] = 2L, ["a"] = 1.0 });
            Assert.AreEqual(new Info { ["t"] = true }, new Info { ["t"] = 1 });
            Assert.AreNotEqual(new Info { ["a"] = 1 }, new Info { ["a"] = 1, ["b"] = 2 });
            Assert.AreNotEqual(new Info { ["a"] = 1 }, new Info { ["a"] = "1" });
            Assert.AreEqual(new Info { ["l"] = new List<object?> { 1, "x" } }, new Info { ["l"] = new object?[] { 1.0, "x" } });
            Assert.AreNotEqual(new Info { ["l"] = new List<object?> { 1 } }, new Info { ["l"] = ValueTuple.Create(1) });
            Assert.AreEqual(new Info { ["n"] = new Info { ["x"] = 1 } }, new Info { ["n"] = new Dictionary<string, object?> { ["x"] = 1 } });
            Assert.IsTrue(new Info { ["a"] = 1 }.Equals(new Dictionary<string, int> { ["a"] = 1 }), "Any .NET dictionary is a dict.");

            using var left = np.array(new[] { 1.0, 2.0 });
            using var right = np.array(new long[] { 1, 2 });
            using var other = np.array(new[] { 1.0, 3.0 });
            Assert.AreEqual(new Info { ["a"] = left }, new Info { ["a"] = right }, "np.array_equal ignores the dtype.");
            Assert.AreNotEqual(new Info { ["a"] = left }, new Info { ["a"] = other });
            Assert.IsTrue(new Info() == new Info());
            Assert.IsTrue(new Info { ["a"] = 1 } != new Info());
        }

        /// <summary>The exact comparisons of <see cref="PyEquality"/>: no rounding between integers and floats, NaN never equal, identity first.</summary>
        [TestMethod]
        public void PyEquality_NumbersAreExact() {
            Assert.IsFalse(PyEquality.AreEqual(BigInteger.Pow(2, 53) + 1, Math.Pow(2, 53)), "Python: 2**53 + 1 != 2.0**53.");
            Assert.IsTrue(PyEquality.AreEqual(BigInteger.Pow(2, 53), Math.Pow(2, 53)));
            Assert.IsFalse(PyEquality.AreEqual(0.1f, 0.1), "np.float32(0.1) != 0.1.");
            Assert.IsTrue(PyEquality.AreEqual(-0.0, 0));
            Assert.IsFalse(PyEquality.AreEqual(double.NaN, double.NaN));
            object nan = double.NaN;
            Assert.IsTrue(PyEquality.AreEqual(nan, nan), "The same object is equal to itself, as in a Python container comparison.");
            using var scalar = NDArray.Scalar<long>(3);
            Assert.IsTrue(PyEquality.AreEqual(scalar, 3.0), "A 0-d array is the NumPy scalar it stands for.");
            using var single = np.array(new[] { 5 });
            Assert.IsTrue(PyEquality.AreEqual(single, 5));
            Assert.IsFalse(PyEquality.AreEqual(null, 0));
            Assert.IsTrue(PyEquality.AreEqual(null, null));
        }

        /// <summary>Python's <c>repr</c> of a dict and of the containers inside it.</summary>
        [TestMethod]
        public void Repr_IsPythons() {
            var info = new Info {
                ["int"] = 1, ["float"] = 0.5, ["none"] = null, ["str"] = "x", ["bool"] = true, ["list"] = new List<string> { "human", "rgb_array" },
                ["tuple"] = ValueTuple.Create(1), ["pair"] = (1, "a"), ["nested"] = new Info { ["k"] = new object?[0] }
            };

            Assert.AreEqual("{'int': 1, 'float': 0.5, 'none': None, 'str': 'x', 'bool': True, 'list': ['human', 'rgb_array'], 'tuple': (1,), 'pair': (1, 'a'), 'nested': {'k': []}}", info.ToString());
            Assert.AreEqual("{}", new Info().ToString());
            Assert.AreEqual("<class 'dict'>", PyBuiltins.TypeRepr(info));
            Assert.AreEqual("<class 'list'>", PyBuiltins.TypeRepr(new int[0]));
            Assert.AreEqual("<class 'tuple'>", PyBuiltins.TypeRepr((1, 2)));
        }

        /// <summary><see cref="Metadata"/>'s typed accessors read and write Gymnasium's keys.</summary>
        [TestMethod]
        public void Metadata_TypedAccessors() {
            var metadata = new Metadata { [Metadata.RenderModesKey] = new[] { "human" }, [Metadata.RenderFpsKey] = 50L };

            CollectionAssert.AreEqual(new[] { "human" }, metadata.RenderModes!.ToArray());
            Assert.AreEqual(50.0, metadata.RenderFps);
            metadata.RenderFps = 30.5;
            metadata.RenderModes = new[] { "rgb_array", "ansi" };
            Assert.AreEqual("{'render_modes': ['rgb_array', 'ansi'], 'render_fps': 30.5}", metadata.ToString());
            Assert.IsNull(new Metadata().RenderModes);
            Assert.IsNull(new Metadata().RenderFps);
            Assert.ThrowsExactly<TypeError>(() => new Metadata { [Metadata.RenderFpsKey] = new object() }.RenderFps);
        }
    }
}
