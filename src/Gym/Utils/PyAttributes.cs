#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Text;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Names a property or field by the Python attribute it stands for, where the default mapping (the snake_case form
    ///     of the C# name) would not give it: <c>[PyName("_disable_render_order_enforcing")]</c>. A non-public member takes
    ///     part in Python attribute access only when it carries this attribute. On a constructor parameter it is the keyword
    ///     <c>make</c> binds to it (<see cref="PyKwargs.Construct"/>), for the rare kwarg snake_case cannot spell. On a class it
    ///     is the name Python prints for it (DR-24): the vector wrappers renamed for unique class names
    ///     (<c>NormalizeObservationVector</c>) keep Gymnasium's (<c>NormalizeObservation</c>) in <c>repr</c> and messages. On
    ///     a static method it is the Python function the method stands for where C# cannot use that name
    ///     (<c>StepApiCompatibility.Dispatch</c> is <c>step_api_compatibility</c>, the class's own name).
    /// </summary>
    /// <remarks>
    ///     An overriding property inherits its base's name (<c>Inherited</c>), but a class's name is read from the class
    ///     itself only (<see cref="PyAttributes.PyTypeName"/>): a subclass prints its own name, as a Python subclass does.
    /// </remarks>
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field | AttributeTargets.Parameter | AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = false,
        Inherited = true)]
    public sealed class PyNameAttribute : Attribute {
        /// <summary>Maps the member to <paramref name="name"/>.</summary>
        /// <param name="name">The Python attribute name, spelled exactly as in the Gymnasium source.</param>
        public PyNameAttribute(string name) => Name = name;

        /// <summary>The Python attribute name.</summary>
        public string Name { get; }
    }

    /// <summary>
    ///     An object that, like a Python instance with a <c>__dict__</c>, can hold attributes created at run time: what
    ///     <c>setattr(obj, name, value)</c> does for a name the class does not declare. Gymnasium's
    ///     <c>set_wrapper_attr(name, value, force=True)</c> relies on it.
    /// </summary>
    public interface IDynamicAttributes {
        /// <summary>The run-time attributes by Python name, in the order they were created.</summary>
        IDictionary<string, object?> DynamicAttributes { get; }
    }

    /// <summary>
    ///     An object whose Python attributes can be absent on one instance and present on another of the same class (DR-45),
    ///     and whose Python class name can depend on the instance: the merged <c>MountainCarEnv</c> has
    ///     <c>get_keys_to_action</c> in discrete mode only and prints <c>Continuous_MountainCarEnv</c> in continuous mode, as
    ///     Gymnasium's two classes differ. <see cref="PyAttributes"/> asks it before reporting a <see cref="PyNameAttribute"/>
    ///     method, and names it in its errors.
    /// </summary>
    internal interface IPythonAttributePresence {
        /// <summary>Whether this instance has the Python attribute <paramref name="name"/>.</summary>
        /// <param name="name">The attribute, in Python spelling.</param>
        /// <returns><see langword="false"/> to hide it on this instance.</returns>
        bool HasPythonAttribute(string name);

        /// <summary>The class name Python prints for this instance (<c>'Continuous_MountainCarEnv' object has no attribute ...</c>).</summary>
        string PythonClassName { get; }
    }

    /// <summary>
    ///     Python's <c>hasattr</c>, <c>getattr</c> and <c>setattr</c> for .NET objects that mirror Python classes: the declared
    ///     attributes are the public instance properties and fields (named by <see cref="PyNameAttribute"/>, or the
    ///     snake_case of the C# name: <c>NpRandomSeed</c> is <c>np_random_seed</c>) plus non-public members marked with
    ///     <see cref="PyNameAttribute"/>; objects implementing <see cref="IDynamicAttributes"/> also hold run-time ones.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Errors are Python's, message for message: a missing attribute raises <see cref="AttributeError"/>
    ///     <c>'CartPoleEnv' object has no attribute 'x'</c>; assigning a read-only property raises
    ///     <c>property 'np_random_seed' of 'ExampleEnv' object has no setter</c>. A property whose getter itself raises
    ///     <see cref="AttributeError"/> (a wrapper's <c>_np_random</c>) is absent for <see cref="HasAttr"/>, as for Python's
    ///     <c>hasattr</c>.
    ///     </para>
    ///     <para>
    ///     Two deviations follow from static typing: a method is an attribute only when it carries
    ///     <see cref="PyNameAttribute"/> (Python's <c>hasattr(env, "step")</c> is true for every method), and a value must
    ///     fit the member's type (Python stores anything): numbers convert between numeric types, anything else that does not
    ///     fit raises <see cref="TypeError"/>.
    ///     </para>
    ///     <para>
    ///     A <see cref="PyNameAttribute"/> method (DR-45, <c>get_keys_to_action</c>) reads as the bound method, a delegate to
    ///     call. A virtual one is present only once the object's class overrides it, and only when an object with
    ///     per-instance presence agrees (<see cref="IPythonAttributePresence"/>), so a base that merely declares it reads as
    ///     Python's missing method. As in Python, where an instance's own attributes come before its class's functions, a
    ///     run-time attribute of the same name shadows the method, and assigning the name creates that attribute.
    ///     </para>
    /// </remarks>
    public static class PyAttributes {
        /// <summary>The attribute members of each type, by Python name; built once per type.</summary>
        private static readonly ConcurrentDictionary<Type, IReadOnlyDictionary<string, MemberInfo>> MembersByType = new();

        /// <summary>
        ///     The snake_case form of a PascalCase name, the port plan's naming rule read backwards (§3.8): <c>NpRandomSeed</c>
        ///     is <c>np_random_seed</c>, <c>XThreshold</c> is <c>x_threshold</c>, <c>RGBArray</c> is <c>rgb_array</c>.
        /// </summary>
        /// <param name="name">A C# member name.</param>
        /// <returns>The Python attribute name.</returns>
        public static string SnakeCase(string name) {
            var text = new StringBuilder(name.Length + 4);
            for (int i = 0; i < name.Length; i++) {
                char c = name[i];
                if (char.IsUpper(c)) {
                    // A word starts at an upper-case letter after a lower-case one or a digit ("NpRandom"), and at the last
                    // capital of an acronym when a lower-case letter follows ("XThreshold", "RGBArray").
                    bool afterLower = i > 0 && (char.IsLower(name[i - 1]) || char.IsDigit(name[i - 1]));
                    bool endsAcronym = i > 0 && char.IsUpper(name[i - 1]) && i + 1 < name.Length && char.IsLower(name[i + 1]);
                    if (afterLower || endsAcronym)
                        text.Append('_');
                    text.Append(char.ToLowerInvariant(c));
                } else {
                    text.Append(c);
                }
            }

            return text.ToString();
        }

        /// <summary>Python's <c>hasattr(target, name)</c>: whether reading the attribute succeeds.</summary>
        /// <param name="target">The object.</param>
        /// <param name="name">The Python attribute name.</param>
        /// <returns><see langword="true"/> when <see cref="GetAttr"/> would return a value.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="target"/> or <paramref name="name"/> is <see langword="null"/>.</exception>
        /// <remarks>Only <see cref="AttributeError"/> means "absent"; any other exception from a getter propagates, as in Python.</remarks>
        public static bool HasAttr(object target, string name) {
            try {
                GetAttr(target, name);
                return true;
            } catch (AttributeError) {
                return false;
            }
        }

        /// <summary>Python's <c>getattr(target, name)</c>: the declared member's value, else the run-time attribute's.</summary>
        /// <param name="target">The object.</param>
        /// <param name="name">The Python attribute name.</param>
        /// <returns>The value (boxed).</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="target"/> or <paramref name="name"/> is <see langword="null"/>.</exception>
        /// <exception cref="AttributeError">
        ///     Thrown when the object has no such attribute (<c>'CartPoleEnv' object has no attribute 'x'</c>), or when the
        ///     member's getter raises it.
        /// </exception>
        public static object? GetAttr(object target, string name) {
            if (target is null)
                throw new ArgumentNullException(nameof(target));
            if (name is null)
                throw new ArgumentNullException(nameof(name));

            if (Members(target.GetType()).TryGetValue(name, out MemberInfo? member)) {
                if (member is MethodInfo method)
                    return BoundMethod(target, method, name);
                try {
                    return member is PropertyInfo property ? property.GetValue(target) : ((FieldInfo)member).GetValue(target);
                } catch (TargetInvocationException e) when (e.InnerException is not null) {
                    // Reflection wraps what the getter threw; Python sees the getter's own exception (AttributeError included).
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException).Throw();
                    throw;
                }
            }

            if (target is IDynamicAttributes dynamic && dynamic.DynamicAttributes.TryGetValue(name, out object? value))
                return value;
            throw new AttributeError($"'{ClassNameOf(target)}' object has no attribute '{name}'");
        }

        /// <summary>
        ///     A <see cref="PyNameAttribute"/> method read as Python reads a method attribute (DR-45): the instance's own
        ///     attribute of that name when it holds one, else the method bound to <paramref name="target"/>, when present.
        /// </summary>
        /// <param name="target">The object.</param>
        /// <param name="method">The most derived declaration of the method for the object's class.</param>
        /// <param name="name">The Python attribute name.</param>
        /// <returns>The shadowing run-time attribute, or a delegate (<c>Func</c>/<c>Action</c>) that calls the method on the object.</returns>
        /// <exception cref="AttributeError">Thrown when the method is absent on this object (<see cref="MethodPresent"/>).</exception>
        private static object? BoundMethod(object target, MethodInfo method, string name) {
            // Python looks an instance's own attributes up before its class's functions: `env.get_keys_to_action = f` wins.
            if (target is IDynamicAttributes dynamic && dynamic.DynamicAttributes.TryGetValue(name, out object? shadow))
                return shadow;
            if (!MethodPresent(target, method, name))
                throw new AttributeError($"'{ClassNameOf(target)}' object has no attribute '{name}'");
            Type delegateType = System.Linq.Expressions.Expression.GetDelegateType(method.GetParameters().Select(p => p.ParameterType).Append(method.ReturnType).ToArray());
            return method.CreateDelegate(delegateType, target);
        }

        /// <summary>
        ///     Whether a <see cref="PyNameAttribute"/> method is present on <paramref name="target"/> (DR-45): a virtual method
        ///     only once the object's class overrides it (the declaring base's body is Python's missing method), and only when an
        ///     object with per-instance presence agrees.
        /// </summary>
        /// <param name="target">The object.</param>
        /// <param name="method">The most derived declaration of the method for the object's class.</param>
        /// <param name="name">The Python attribute name, for the per-instance hook.</param>
        /// <returns><see langword="true"/> when reading the attribute should return the bound method.</returns>
        private static bool MethodPresent(object target, MethodInfo method, string name) {
            // The map holds the most derived declaration, so a virtual method declared where its base definition is was never overridden.
            if (method.IsVirtual && method.GetBaseDefinition().DeclaringType == method.DeclaringType)
                return false;
            return target is not IPythonAttributePresence presence || presence.HasPythonAttribute(name);
        }

        /// <summary>The class name Python prints for <paramref name="target"/>: the instance's own when it varies by instance, else its type's.</summary>
        /// <param name="target">The object.</param>
        /// <returns>The name, as <c>'CartPoleEnv' object has no attribute 'x'</c> prints it.</returns>
        private static string ClassNameOf(object target) =>
            target is IPythonAttributePresence presence ? presence.PythonClassName : PyTypeName(target.GetType());

        /// <summary>
        ///     Python's <c>setattr(target, name, value)</c>: assigns a declared member, or creates or replaces a run-time
        ///     attribute on an object that can hold them.
        /// </summary>
        /// <param name="target">The object.</param>
        /// <param name="name">The Python attribute name.</param>
        /// <param name="value">The value; numbers are converted to a numeric member's type.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="target"/> or <paramref name="name"/> is <see langword="null"/>.</exception>
        /// <exception cref="AttributeError">
        ///     Thrown for a read-only property (<c>property 'np_random_seed' of 'ExampleEnv' object has no setter</c>), for a
        ///     setter that raises it, and for a new name on an object that cannot hold run-time attributes.
        /// </exception>
        /// <exception cref="TypeError">Thrown when the value does not fit the member's type.</exception>
        public static void SetAttr(object target, string name, object? value) {
            if (target is null)
                throw new ArgumentNullException(nameof(target));
            if (name is null)
                throw new ArgumentNullException(nameof(name));

            if (Members(target.GetType()).TryGetValue(name, out MemberInfo? member)) {
                if (member is MethodInfo) {
                    // Python's `env.get_keys_to_action = f` sets an instance attribute that shadows the class's function.
                    if (target is IDynamicAttributes shadowed) {
                        shadowed.DynamicAttributes[name] = value;
                        return;
                    }

                    throw new AttributeError($"'{ClassNameOf(target)}' object attribute '{name}' is read-only");
                }

                // A read-only attribute refuses any value, before the value is looked at, as Python's property does. A
                // readonly field counts as one: Python would model it as a property without a setter.
                if (member is PropertyInfo { SetMethod: null } or FieldInfo { IsInitOnly: true })
                    throw new AttributeError($"property '{name}' of '{ClassNameOf(target)}' object has no setter");
                Type memberType = member is PropertyInfo p ? p.PropertyType : ((FieldInfo)member).FieldType;
                object? converted = Fit(value, memberType, name);
                try {
                    if (member is PropertyInfo property)
                        property.SetValue(target, converted);
                    else
                        ((FieldInfo)member).SetValue(target, converted);
                    return;
                } catch (TargetInvocationException e) when (e.InnerException is not null) {
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException).Throw();
                    throw;
                }
            }

            if (target is IDynamicAttributes dynamic) {
                dynamic.DynamicAttributes[name] = value;
                return;
            }

            throw new AttributeError($"'{ClassNameOf(target)}' object has no attribute '{name}'");
        }

        /// <summary>
        ///     The name Python prints for a class: its <see cref="PyNameAttribute"/> when it has one (a class renamed for a
        ///     unique name, DR-24), else the .NET type name without the generic arity (<c>TimeLimit`2</c> is
        ///     <c>TimeLimit</c>); also what <c>Wrapper.class_name()</c> returns.
        /// </summary>
        /// <param name="type">The type.</param>
        /// <returns>The class name.</returns>
        public static string PyTypeName(Type type) {
            // A closed generic type carries its definition's attributes, so one lookup covers both forms.
            if (type.GetCustomAttributes(typeof(PyNameAttribute), inherit: false) is [PyNameAttribute declared, ..])
                return declared.Name;
            string name = type.Name;
            int tick = name.IndexOf('`');
            return tick < 0 ? name : name.Substring(0, tick);
        }

        /// <summary>The attribute members of a type by Python name, most derived first where names collide.</summary>
        /// <param name="type">The type.</param>
        /// <returns>The map.</returns>
        private static IReadOnlyDictionary<string, MemberInfo> Members(Type type) => MembersByType.GetOrAdd(type, Build);

        /// <summary>Collects the attribute members of a type and its bases.</summary>
        /// <param name="type">The type.</param>
        /// <returns>The map from Python name to member.</returns>
        private static IReadOnlyDictionary<string, MemberInfo> Build(Type type) {
            var members = new Dictionary<string, MemberInfo>(StringComparer.Ordinal);
            // Most derived type first: a property redeclared lower in the hierarchy (a wrapper's spec) wins over the base's.
            for (Type? level = type; level != null && level != typeof(object); level = level.BaseType) {
                const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
                foreach (PropertyInfo property in level.GetProperties(flags)) {
                    if (property.GetIndexParameters().Length > 0)
                        continue;
                    string? name = AttributeName(property, IsPublic(property));
                    if (name != null)
                        members.TryAdd(name, property);
                }

                foreach (FieldInfo field in level.GetFields(flags)) {
                    // Compiler-generated backing fields are reached through their properties.
                    if (field.IsDefined(typeof(System.Runtime.CompilerServices.CompilerGeneratedAttribute)))
                        continue;
                    string? name = AttributeName(field, field.IsPublic);
                    if (name != null)
                        members.TryAdd(name, field);
                }

                foreach (MethodInfo method in level.GetMethods(flags)) {
                    // Only methods marked [PyName] are attributes (DR-45); accessors and operators never are, and a generic
                    // method has no single delegate to bind. An override inherits its base's name, and the most derived
                    // declaration is found first, which is what the presence check reads.
                    if (method.IsSpecialName || method.ContainsGenericParameters)
                        continue;
                    if (method.GetCustomAttribute<PyNameAttribute>(inherit: true) is { } methodName)
                        members.TryAdd(methodName.Name, method);
                }
            }

            return members;
        }

        /// <summary>The Python name of a member, or <see langword="null"/> when it is not an attribute.</summary>
        /// <param name="member">The member.</param>
        /// <param name="isPublic">Whether it is public (non-public members need <see cref="PyNameAttribute"/>).</param>
        /// <returns>The name, or <see langword="null"/>.</returns>
        private static string? AttributeName(MemberInfo member, bool isPublic) {
            var explicitName = member.GetCustomAttribute<PyNameAttribute>(inherit: true);
            if (explicitName != null)
                return explicitName.Name;
            return isPublic ? SnakeCase(member.Name) : null;
        }

        /// <summary>Whether a property is public: its getter or setter is.</summary>
        /// <param name="property">The property.</param>
        /// <returns><see langword="true"/> when either accessor is public.</returns>
        private static bool IsPublic(PropertyInfo property) => property.GetMethod?.IsPublic == true || property.SetMethod?.IsPublic == true;

        /// <summary>Makes a value fit a member's type: as is when assignable, converted between numeric types, else a <see cref="TypeError"/>.</summary>
        /// <param name="value">The value.</param>
        /// <param name="type">The member's type.</param>
        /// <param name="name">The attribute, for the message.</param>
        /// <returns>The value to store.</returns>
        /// <exception cref="TypeError">Thrown when the value cannot be stored in a member of that type.</exception>
        private static object? Fit(object? value, Type type, string name) {
            if (value is null) {
                if (type.IsValueType && Nullable.GetUnderlyingType(type) is null)
                    throw new TypeError($"cannot set attribute '{name}' of type {type.Name} to None");
                return null;
            }

            if (type.IsInstanceOfType(value))
                return value;
            Type target = Nullable.GetUnderlyingType(type) ?? type;
            if (IsNumber(value.GetType()) && IsNumber(target))
                return Convert.ChangeType(value, target, CultureInfo.InvariantCulture);
            throw new TypeError($"cannot set attribute '{name}' of type {type.Name} to {PyBuiltins.Repr(value)} ({PyBuiltins.TypeRepr(value)})");
        }

        /// <summary>Whether a type is a .NET number (not <see langword="bool"/>, which Python treats as an int but .NET does not).</summary>
        /// <param name="type">The type.</param>
        /// <returns><see langword="true"/> for the integer and floating-point primitives and <see langword="decimal"/>.</returns>
        private static bool IsNumber(Type type) {
            return type == typeof(sbyte) || type == typeof(byte) || type == typeof(short) || type == typeof(ushort) || type == typeof(int) || type == typeof(uint)
                   || type == typeof(long) || type == typeof(ulong) || type == typeof(float) || type == typeof(double) || type == typeof(decimal);
        }
    }
}
