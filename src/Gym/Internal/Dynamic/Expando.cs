// Gym.Internal is 0.2's code, [Obsolete] since 2.0 and removed in 3.0 with Gym.Legacy (DR-56): it keeps no nullable
// context (DR-54), rather than 89 annotations for code on its way out.
#nullable disable
//#define SupportXmlSerialization

#region License

/*
 **************************************************************
 *  Author: Rick Strahl
 *          © West Wind Technologies, 2012
 *          http://www.west-wind.com/
 *
 * Created: Feb 2, 2012
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 **************************************************************
*/

#endregion

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Dynamic;
using System.Reflection;
using ICollection = System.Collections.ICollection;
using IEnumerable = System.Collections.IEnumerable;
using IEnumerator = System.Collections.IEnumerator;

namespace Gym.Dynamic {
    /// <summary>
    /// Class that provides extensible properties and methods to an
    /// existing object when cast to dynamic. This
    /// dynamic object stores 'extra' properties in a dictionary or
    /// checks the actual properties of the instance passed via
    /// constructor.
    ///
    /// This class can be subclassed to extend an existing type or
    /// you can pass in an instance to extend. Properties (both
    /// dynamic and strongly typed) can be accessed through an
    /// indexer.
    ///
    /// This type allows you three ways to access its properties:
    ///
    /// Directly: any explicitly declared properties are accessible
    /// Dynamic: dynamic cast allows access to dictionary and native properties/methods
    /// Dictionary: Any of the extended properties are accessible via IDictionary interface
    /// </summary>
    /// <remarks>
    ///     The legacy <see cref="Gym.Collections.Dict"/>'s <c>AsDynamic()</c> view is the one use in 2.0. Obsolete with it
    ///     (DR-56) and removed in 3.0: use <see cref="ExpandoObject"/>. Every member lookup goes through reflection, and
    ///     reflection failures are swallowed: a dynamic read of a member that throws reads as missing.
    /// </remarks>
    [Serializable]
    [DebuggerStepThrough]
    [Obsolete("The legacy Dict's dynamic view: use System.Dynamic.ExpandoObject. Removed in 3.0 with Gym.Legacy.")]
    public class Expando : DynamicObject, IDynamicMetaObjectProvider, IEnumerable<KeyValuePair<string, object>> {
        /// <summary>
        /// Instance of object passed in
        /// </summary>
        protected object Instance;

        /// <summary>
        /// Cached type of the instance
        /// </summary>
        Type InstanceType;

        /// <summary>
        ///     The instance's public properties declared on its own type (inherited ones are not included), read once and
        ///     cached; <see langword="null"/> without an instance.
        /// </summary>
        PropertyInfo[] InstancePropertyInfo {
            get {
                if (_InstancePropertyInfo == null && Instance != null)
                    _InstancePropertyInfo = Instance.GetType().GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly);
                return _InstancePropertyInfo;
            }
        }

        /// <summary><see cref="InstancePropertyInfo"/>'s cache.</summary>
        PropertyInfo[] _InstancePropertyInfo;


#if SupportXmlSerialization
        /// <summary>
        /// String Dictionary that contains the extra dynamic values
        /// stored on this object/instance
        /// </summary>
        /// <remarks>Using PropertyBag to support XML Serialization of the dictionary</remarks>
        public PropertyBag Properties = new PropertyBag();
#else
        /// <summary>
        /// String Dictionary that contains the extra dynamic values
        /// stored on this object/instance
        /// </summary>
        /// <remarks>
        ///     A public, replaceable field: the legacy <see cref="Gym.Collections.Dict"/>'s view points it at the dictionary
        ///     itself, so writes through the view change the dictionary.
        /// </remarks>
        public IDictionary<string, object> Properties = new Dictionary<string, object>();
#endif

        //public Dictionary<string,object> Properties = new Dictionary<string, object>();

        /// <summary>
        /// This constructor just works off the internal dictionary and any
        /// public properties of this object.
        ///
        /// Note you can subclass Expando.
        /// </summary>
        public Expando() {
            Initialize(this);
        }

        /// <summary>
        /// Allows passing in an existing instance variable to 'extend'.
        /// </summary>
        /// <remarks>
        /// You can pass in null here if you don't want to
        /// check native properties and only check the Dictionary!
        /// </remarks>
        /// <param name="instance">The object whose public properties and methods the dynamic view exposes, or <see langword="null"/>.</param>
        public Expando(object instance) {
            Initialize(instance);
        }

        /// <summary>
        /// Create an Expando from a dictionary
        /// </summary>
        /// <param name="dict">
        ///     The entries to copy in. Collections are copied into lists, their string-keyed dictionaries turned into
        ///     <see cref="Expando"/>s. Footgun: a nested dictionary value is wrapped as an instance (the
        ///     <see cref="Expando(object)"/> overload), not copied entry by entry, so its keys are not dynamic members.
        /// </param>
        /// <exception cref="NullReferenceException">When <paramref name="dict"/> is <see langword="null"/>.</exception>
        public Expando(IDictionary<string, object> dict) {
            var expando = this;

            Initialize(expando);

            foreach (var kvp in dict) {
                var kvpValue = kvp.Value;
                if (kvpValue is IDictionary<string, object>) {
                    var expandoVal = new Expando(kvpValue);
                    expando[kvp.Key] = expandoVal;
                } else if (kvp.Value is ICollection) {
                    // iterate through the collection and convert any string-object dictionaries
                    // along the way into expando objects
                    var objList = new List<object>();
                    foreach (var item in (ICollection) kvp.Value) {
                        var itemVals = item as IDictionary<string, object>;
                        if (itemVals != null) {
                            var expandoItem = new Expando(itemVals);
                            objList.Add(expandoItem);
                        } else {
                            objList.Add(item);
                        }
                    }

                    expando[kvp.Key] = objList;
                } else {
                    expando[kvp.Key] = kvpValue;
                }
            }
        }


        /// <summary>
        ///     Sets the instance whose members the dynamic view falls back to, and caches its type. Constructors call it;
        ///     a subclass may call it again to point the view at another object.
        /// </summary>
        /// <param name="instance">The instance, or <see langword="null"/> for a dictionary-only view.</param>
        protected void Initialize(object instance) {
            Instance = instance;
            if (instance != null)
                InstanceType = instance.GetType();
        }


        /// <summary>
        /// Return both instance and dynamic names.
        ///
        /// Important to return both so JSON serialization with
        /// Json.NET works.
        /// </summary>
        /// <returns>The instance's own public property names, then the dictionary's keys.</returns>
        public override IEnumerable<string> GetDynamicMemberNames() {
            foreach (var prop in GetProperties(true))
                yield return prop.Key;
        }


        /// <summary>
        /// Try to retrieve a member by name first from instance properties
        /// followed by the collection entries.
        /// </summary>
        /// <param name="binder">The dynamic binder naming the member.</param>
        /// <param name="result">The value read, or <see langword="null"/>.</param>
        /// <returns>
        ///     Whether the member was found: the dictionary is checked first, then the instance's public properties; a
        ///     property that throws reads as not found.
        /// </returns>
        public override bool TryGetMember(GetMemberBinder binder, out object result) {
            result = null;

            // first check the Properties collection for member
            if (Properties.Keys.Contains(binder.Name)) {
                result = Properties[binder.Name];
                return true;
            }


            // Next check for Public properties via Reflection
            if (Instance != null) {
                try {
                    return GetProperty(Instance, binder.Name, out result);
                } catch { }
            }

            // failed to retrieve a property
            result = null;
            return false;
        }


        /// <summary>
        /// Property setter implementation tries to retrieve value from instance
        /// first then into this object
        /// </summary>
        /// <param name="binder">The dynamic binder naming the member.</param>
        /// <param name="value">The value to store.</param>
        /// <returns>
        ///     <see langword="true"/> when stored, on the instance's writable property of that name or else in the
        ///     dictionary; <see langword="false"/> when the instance's setter threw.
        /// </returns>
        public override bool TrySetMember(SetMemberBinder binder, object value) {
            // first check to see if there's a native property to set
            if (Instance != null) {
                try {
                    bool result = SetProperty(Instance, binder.Name, value);
                    if (result)
                        return true;
                } catch {
                    return false;
                }
            }

            // no match - set or add to dictionary
            Properties[binder.Name] = value;
            return true;
        }

        /// <summary>
        /// Dynamic invocation method. Currently allows only for Reflection based
        /// operation (no ability to add methods dynamically).
        /// </summary>
        /// <param name="binder">The dynamic binder naming the method.</param>
        /// <param name="args">The arguments.</param>
        /// <param name="result">The method's return value, or <see langword="null"/>.</param>
        /// <returns>Whether a public instance method of that name was found and ran; a method that throws counts as not found.</returns>
        public override bool TryInvokeMember(InvokeMemberBinder binder, object[] args, out object result) {
            if (Instance != null) {
                try {
                    // check instance passed in for methods to invoke
                    if (InvokeMethod(Instance, binder.Name, args, out result))
                        return true;
                } catch { }
            }

            result = null;
            return false;
        }


        /// <summary>
        /// Reflection Helper method to retrieve a property
        /// </summary>
        /// <param name="instance">The object to read from; <see langword="null"/> means this object.</param>
        /// <param name="name">The property's name.</param>
        /// <param name="result">The value read, or <see langword="null"/>.</param>
        /// <returns>Whether a public instance property of that name exists on the cached instance type.</returns>
        /// <exception cref="NullReferenceException">When the view has no instance (its type was never cached).</exception>
        /// <exception cref="TargetInvocationException">When the property's getter throws.</exception>
        protected bool GetProperty(object instance, string name, out object result) {
            if (instance == null)
                instance = this;

            var miArray = InstanceType.GetMember(name, BindingFlags.Public | BindingFlags.GetProperty | BindingFlags.Instance);
            if (miArray != null && miArray.Length > 0) {
                var mi = miArray[0];
                if (mi.MemberType == MemberTypes.Property) {
                    result = ((PropertyInfo) mi).GetValue(instance, null);
                    return true;
                }
            }

            result = null;
            return false;
        }

        /// <summary>
        /// Reflection helper method to set a property value
        /// </summary>
        /// <param name="instance">Checked for <see langword="null"/> only. Footgun: the value is always set on <see cref="Instance"/>.</param>
        /// <param name="name">The property's name.</param>
        /// <param name="value">The value to set.</param>
        /// <returns>Whether a public instance property of that name exists and was set.</returns>
        /// <exception cref="NullReferenceException">When the view has no instance (its type was never cached).</exception>
        /// <exception cref="ArgumentException">When the property has no setter or the value's type does not fit.</exception>
        /// <exception cref="TargetInvocationException">When the property's setter throws.</exception>
        protected bool SetProperty(object instance, string name, object value) {
            if (instance == null)
                instance = this;

            var miArray = InstanceType.GetMember(name, BindingFlags.Public | BindingFlags.SetProperty | BindingFlags.Instance);
            if (miArray != null && miArray.Length > 0) {
                var mi = miArray[0];
                if (mi.MemberType == MemberTypes.Property) {
                    ((PropertyInfo) mi).SetValue(Instance, value, null);
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Reflection helper method to invoke a method
        /// </summary>
        /// <param name="instance">Checked for <see langword="null"/> only. Footgun: the method is always invoked on <see cref="Instance"/>.</param>
        /// <param name="name">The method's name; with overloads, the first one reflection lists is called, whatever the arguments.</param>
        /// <param name="args">The arguments.</param>
        /// <param name="result">The method's return value, or <see langword="null"/>.</param>
        /// <returns>Whether a public instance method of that name was found and invoked.</returns>
        /// <exception cref="NullReferenceException">When the view has no instance, or the member of that name is not a method.</exception>
        /// <exception cref="TargetParameterCountException">When the arguments do not match the method's parameters.</exception>
        /// <exception cref="TargetInvocationException">When the method throws.</exception>
        protected bool InvokeMethod(object instance, string name, object[] args, out object result) {
            if (instance == null)
                instance = this;

            // Look at the instanceType
            var miArray = InstanceType.GetMember(name,
                BindingFlags.InvokeMethod |
                BindingFlags.Public | BindingFlags.Instance);

            if (miArray != null && miArray.Length > 0) {
                var mi = miArray[0] as MethodInfo;
                result = mi.Invoke(Instance, args);
                return true;
            }

            result = null;
            return false;
        }

        /// <summary>
        ///     Converts the dynamic view to <paramref name="binder"/>'s type: the wrapped instance when it is of that type,
        ///     else this object when it is.
        /// </summary>
        /// <param name="binder">The conversion's target type.</param>
        /// <param name="result">The instance or this object, or <see langword="null"/>.</param>
        /// <returns>Whether either fits the target type.</returns>
        public override bool TryConvert(ConvertBinder binder, out object result) {
            bool IsSameOrInherits(Type actualType, Type expectedType) {
                if (!(actualType == expectedType))
                    return expectedType.IsAssignableFrom(actualType);
                return true;
            }

            if (Instance != null && IsSameOrInherits(Instance.GetType(), binder.Type)) {
                result = Instance;
                return true;
            }

            if (IsSameOrInherits(this.GetType(), binder.Type)) {
                result = this;
                return true;
            }

            result = null;
            return false;
        }

        /// <summary>
        /// Convenience method that provides a string Indexer
        /// to the Properties collection AND the strongly typed
        /// properties of the object by name.
        ///
        /// // dynamic
        /// exp["Address"] = "112 nowhere lane";
        /// // strong
        /// var name = exp["StronglyTypedProperty"] as string;
        /// </summary>
        /// <remarks>
        /// The getter checks the Properties dictionary first
        /// then looks in PropertyInfo for properties.
        /// The setter checks the instance properties before
        /// checking the Properties dictionary.
        /// </remarks>
        /// <param name="key">The member's name.</param>
        /// <returns>The dictionary's value, else the instance property's.</returns>
        /// <exception cref="KeyNotFoundException">When reading a name that is neither a key nor an instance property.</exception>
        /// <exception cref="NullReferenceException">When setting a new key on a view without an instance.</exception>
        public object this[string key] {
            get {
                try {
                    // try to get from properties collection first
                    return Properties[key];
                } catch (KeyNotFoundException) {
                    // try reflection on instanceType
                    object result = null;
                    if (GetProperty(Instance, key, out result))
                        return result;

                    // nope doesn't exist
                    throw;
                }
            }
            set {
                if (Properties.ContainsKey(key)) {
                    Properties[key] = value;
                    return;
                }

                // check instance for existance of type first
                var miArray = InstanceType.GetMember(key, BindingFlags.Public | BindingFlags.GetProperty | BindingFlags.Instance);
                if (miArray != null && miArray.Length > 0)
                    SetProperty(Instance, key, value);
                else
                    Properties[key] = value;
            }
        }


        /// <summary>
        /// Returns and the properties of
        /// </summary>
        /// <param name="includeInstanceProperties">Whether to list the instance's own public properties (read by reflection) before the dictionary's entries.</param>
        /// <returns>The name-value pairs, read lazily.</returns>
        public IEnumerable<KeyValuePair<string, object>> GetProperties(bool includeInstanceProperties = false) {
            if (includeInstanceProperties && Instance != null) {
                foreach (var prop in this.InstancePropertyInfo)
                    yield return new KeyValuePair<string, object>(prop.Name, prop.GetValue(Instance, null));
            }

            foreach (var key in this.Properties.Keys)
                yield return new KeyValuePair<string, object>(key, this.Properties[key]);
        }


        /// <summary>
        /// Checks whether a property exists in the Property collection
        /// or as a property on the instance
        /// </summary>
        /// <param name="item">The pair whose key to look for; its value is not compared.</param>
        /// <param name="includeInstanceProperties">Whether the instance's own public properties count too.</param>
        /// <returns>Whether the key exists.</returns>
        public bool Contains(KeyValuePair<string, object> item, bool includeInstanceProperties = false) {
            bool res = Properties.ContainsKey(item.Key);
            if (res)
                return true;

            if (includeInstanceProperties && Instance != null) {
                foreach (var prop in this.InstancePropertyInfo) {
                    if (prop.Name == item.Key)
                        return true;
                }
            }

            return false;
        }

        /// <summary>Enumerates the dictionary's entries only; <see cref="GetProperties"/> adds the instance's properties.</summary>
        /// <returns>The dictionary's enumerator.</returns>
        public IEnumerator<KeyValuePair<string, object>> GetEnumerator() {
            return this.Properties.GetEnumerator();
        }

        /// <summary>Enumerates the dictionary's entries only, untyped.</summary>
        /// <returns>The dictionary's enumerator.</returns>
        IEnumerator IEnumerable.GetEnumerator() {
            return GetEnumerator();
        }

        /// <summary>
        /// Converts an <see cref="IDictionary{TKey, TValue}"/> into an <see cref="Expando"/>
        /// </summary>
        /// <param name="dict">The entries; nested string-keyed dictionaries become nested <see cref="Expando"/>s, recursively, and collections lists.</param>
        /// <returns><see cref="Expando"/></returns>
        /// <exception cref="NullReferenceException">When <paramref name="dict"/> is <see langword="null"/>.</exception>
        public static Expando ToIndexableExpando(IDictionary<string, object> dict) {
            var expando = new Expando();


            foreach (var kvp in dict) {
                var kvpValue = kvp.Value as IDictionary<string, object>;
                if (kvpValue != null) {
                    var expandoVal = ToIndexableExpando(kvpValue);
                    expando[kvp.Key] = expandoVal;
                } else if (kvp.Value is ICollection) {
                    // iterate through the collection and convert any string-object dictionaries
                    // along the way into expando objects
                    var objList = new List<object>();
                    foreach (var item in (ICollection) kvp.Value) {
                        var itemVals = item as IDictionary<string, object>;
                        if (itemVals != null) {
                            var expandoItem = ToIndexableExpando(itemVals);
                            objList.Add(expandoItem);
                        } else {
                            objList.Add(item);
                        }
                    }

                    expando[kvp.Key] = objList;
                } else {
                    expando[kvp.Key] = kvp.Value;
                }
            }

            return expando;
        }
    }
}
