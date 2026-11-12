// Gym.Internal is 0.2's code, [Obsolete] since 1.0 and removed in 2.0 with Gym.Legacy (DR-56): it keeps no nullable
// context (DR-54), rather than 89 annotations for code on its way out.
#nullable disable
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace Gym.Reflection {
    /// <summary>
    ///     0.2's reflection extension methods on <see cref="Type"/>, <see cref="ConstructorInfo"/>, <see cref="MethodInfo"/>,
    ///     enums and tuples. In 2.0 only <see cref="DefaultConstructor{T}"/> calls two of them (<see cref="GetDefaultConstructor(Type)"/>
    ///     and <see cref="CreateDefaultConstructor{T}"/>), for the legacy deep merge.
    /// </summary>
    /// <remarks>
    ///     Obsolete with the legacy <see cref="Gym.Collections.Dict"/> (DR-56) and removed in 2.0: the BCL covers each
    ///     method (<see cref="Type.GetConstructor(Type[])"/> with <see cref="Type.EmptyTypes"/>, <see cref="Type.IsAssignableFrom"/>,
    ///     <see cref="System.Runtime.CompilerServices.ITuple"/>, <see cref="CustomAttributeExtensions"/>). Being extension
    ///     methods, they appear on every <see cref="object"/> and <see cref="Type"/> in a file that imports
    ///     <c>Gym.Reflection</c>.
    /// </remarks>
    [Obsolete("Part of the legacy Dict's merging: use the BCL's reflection (Type.GetConstructor(Type.EmptyTypes), Type.IsAssignableTo, System.Runtime.CompilerServices.ITuple). Removed in 2.0 with Gym.Legacy.")]
    public static class ReflectionHelper {
        //private static readonly Func<T> DefaultConstructor1;
        //
        //public static Func<T> DefaultConstructor {
        //    get {
        //        if ()
        //        return DefaultConstructor1;
        //    }
        //}

        /// <summary>
        ///     <typeparamref name="TType"/>'s parameterless constructor, or else its constructor with the fewest parameters
        ///     that are all optional; <see langword="null"/> when it has neither.
        /// </summary>
        /// <typeparam name="TType">The type whose constructor to find.</typeparam>
        /// <returns>The constructor, or <see langword="null"/>.</returns>
        public static ConstructorInfo GetDefaultConstructor<TType>() {
            var type = typeof(TType);
            return type.GetDefaultConstructor();
        }

        /// <summary>
        ///   Gets the default value for a type. This method should serve as
        ///   a programmatic equivalent to <c>default(T)</c>.
        /// </summary>
        /// <param name="type">The type whose default value should be retrieved.</param>
        /// <returns>
        ///     A zeroed instance for a value type. Footgun: for a reference type it is not <c>default(T)</c>'s
        ///     <see langword="null"/> but a new object from its default constructor.
        /// </returns>
        /// <exception cref="NullReferenceException">For a reference type with no default constructor.</exception>
        /// <exception cref="ArgumentNullException">When <paramref name="type"/> is <see langword="null"/>.</exception>
        public static object GetDefaultValue(this Type type) {
            if (type.IsValueType)
                return Activator.CreateInstance(type);
            return GetDefaultConstructor(type).Invoke(null, new object[0]);
        }

        /// <summary>
        ///     <paramref name="type"/>'s public parameterless constructor, or else its public constructor with the fewest
        ///     parameters that are all optional; <see langword="null"/> when it has neither.
        /// </summary>
        /// <param name="type">The type whose constructor to find.</param>
        /// <returns>The constructor, or <see langword="null"/>.</returns>
        /// <exception cref="ArgumentNullException">When <paramref name="type"/> is <see langword="null"/>.</exception>
        public static ConstructorInfo GetDefaultConstructor(this Type type) {
            if (type == null)
                throw new ArgumentNullException(nameof(type));
            var constructor = type.GetConstructor(Type.EmptyTypes);
            if (constructor == null) {
                var ctors =
                    from ctor in type.GetConstructors()
                    let prms = ctor.GetParameters()
                    where prms.All(p => p.IsOptional)
                    orderby prms.Length
                    select ctor;
                constructor = ctors.FirstOrDefault();
            }

            return constructor;
        }

        /// <summary>
        ///     Creates a <see cref="Func{TResult}"/> that returns <paramref name="ci"/>'s <see cref="ParameterInfo.ParameterType"/> to a constructor that all of it's parameters has default value.
        /// </summary>
        /// <typeparam name="T">The time of <paramref name="ci"/></typeparam>
        /// <param name="ci">The constructor to wrap.</param>
        /// <exception cref="InvalidOperationException">When <paramref name="ci"/> contains non-optional (<see cref="ParameterInfo.IsOptional"/>) parameters. Footgun: its message lists the optional parameters, not the offending ones.</exception>
        /// <returns>A delegate that calls <paramref name="ci"/> through reflection, each optional parameter at its default, and casts the result (an <see cref="InvalidCastException"/> when it is not a <typeparamref name="T"/>).</returns>
        public static Func<T> CreateDefaultConstructor<T>(this ConstructorInfo ci) {
            var parameters = ci.GetParameters();
            if (parameters.Any(p => p.IsOptional == false))
                throw new InvalidOperationException("This constructor contains non optional parameter/s; " + String.Join(",", parameters.Where(p => p.IsOptional).Select(p => $"[{p.Position}] {p.ParameterType.Name} {p.Name}").ToArray()));
            var arr = parameters.Where(p => p.IsOptional).Select(p => p.RawDefaultValue).ToArray();
            return () => (T) ci.Invoke(arr);
        }

        /// <summary>
        ///     Creates a <see cref="Func{TResult}"/> that returns <paramref name="ci"/>'s <see cref="ParameterInfo.ParameterType"/> to a constructor that all of it's parameters has default value.
        /// </summary>
        /// <param name="ci">The constructor to wrap.</param>
        /// <exception cref="InvalidOperationException">When <paramref name="ci"/> contains non-optional (<see cref="ParameterInfo.IsOptional"/>) parameters. Footgun: its message lists the optional parameters, not the offending ones.</exception>
        /// <returns>A delegate that calls <paramref name="ci"/> through reflection, each optional parameter at its default.</returns>
        public static Func<object> CreateDefaultConstructor(this ConstructorInfo ci) {
            var parameters = ci.GetParameters();
            if (parameters.Any(p => p.IsOptional == false))
                throw new InvalidOperationException("This constructor contains non optional parameter/s; " + String.Join(",", parameters.Where(p => p.IsOptional).Select(p => $"[{p.Position}] {p.ParameterType.Name} {p.Name}").ToArray()));
            var arr = parameters.Where(p => p.IsOptional).Select(p => p.RawDefaultValue).ToArray();
            return () => ci.Invoke(arr);
        }


        /// <summary>
        ///     Checks if the type/object has this interface.
        /// </summary>
        /// <typeparam name="T">The interface, or any type <paramref name="obj"/> may derive from.</typeparam>
        /// <param name="obj">The object to test; <see langword="null"/> gives <see langword="false"/>.</param>
        /// <returns>Whether <paramref name="obj"/>'s run-time type is assignable to <typeparamref name="T"/> (base classes count too).</returns>
        public static bool HasInterface<T>(this object obj) {
            if (obj == null)
                return false;
            // ReSharper disable once UseMethodIsInstanceOfType
            // ReSharper disable once UseIsOperator.1
            return typeof(T).IsAssignableFrom(obj.GetType());
        }

        /// <summary>
        ///     Checks if the type/object has this interface.
        /// </summary>
        /// <typeparam name="T">The interface, or any type <paramref name="type"/> may derive from.</typeparam>
        /// <param name="type">The type to test; <see langword="null"/> gives <see langword="false"/>.</param>
        /// <returns>Whether <paramref name="type"/> is assignable to <typeparamref name="T"/> (base classes count too).</returns>
        public static bool HasInterface<T>(this Type type) {
            return typeof(T).IsAssignableFrom(type);
        }

        /// <summary>
        ///     Checks if the type/object has this interface.
        /// </summary>
        /// <param name="obj">The object to test; <see langword="null"/> gives <see langword="false"/>.</param>
        /// <param name="interface">The interface, or any type; <see langword="null"/> gives <see langword="false"/>.</param>
        /// <returns>Whether <paramref name="obj"/>'s run-time type is assignable to <paramref name="interface"/>.</returns>
        public static bool HasInterface(this object obj, Type @interface) {
            if (@interface == null)
                return false;
            if (obj == null)
                return false;
            // ReSharper disable once UseMethodIsInstanceOfType
            return @interface.IsAssignableFrom(obj.GetType()); //HasInterface(obj.GetType(), @interface.Name);
        }

        /// <summary>
        ///     Checks if the type/object has this interface.
        /// </summary>
        /// <param name="type">The type to test; <see langword="null"/> gives <see langword="false"/>.</param>
        /// <param name="interface">The interface, or any type; <see langword="null"/> gives <see langword="false"/>.</param>
        /// <returns>Whether <paramref name="type"/> is assignable to <paramref name="interface"/>.</returns>
        public static bool HasInterface(this Type type, Type @interface) {
            if (@interface == null)
                return false;
            if (type == null)
                return false;
            return @interface.IsAssignableFrom(type);
        }


        /// <summary>
        ///   Gets a the value of a <see cref="T:System.ComponentModel.DescriptionAttribute" />
        ///   associated with a particular enumeration value.
        /// </summary>
        /// <typeparam name="T">The enumeration type.</typeparam>
        /// <param name="source">The enumeration value.</param>
        /// <returns>The string value stored in the value's description attribute.</returns>
        /// <exception cref="NullReferenceException">For a value that names no single member (a combination of flags, or an undefined number).</exception>
        public static string GetDescription<T>(this T source) where T : Enum {
            DescriptionAttribute[] customAttributes = (DescriptionAttribute[]) source.GetType().GetField(source.ToString()).GetCustomAttributes(typeof(DescriptionAttribute), false);
            if (customAttributes != null && customAttributes.Length != 0)
                return customAttributes[0].Description;
            return source.ToString();
        }

        /// <summary>The open <see cref="ValueTuple"/> types of every arity, which <see cref="IsValueTupleType"/> matches against.</summary>
        private static readonly HashSet<Type> ValueTupleTypes = new HashSet<Type>(new Type[] {
            typeof(ValueTuple<>),
            typeof(ValueTuple<,>),
            typeof(ValueTuple<,,>),
            typeof(ValueTuple<,,,>),
            typeof(ValueTuple<,,,,>),
            typeof(ValueTuple<,,,,,>),
            typeof(ValueTuple<,,,,,,>),
            typeof(ValueTuple<,,,,,,,>)
        });

        /// <summary>Whether <paramref name="obj"/> is a tuple; see <see cref="IsValueTupleType"/> for what counts.</summary>
        /// <param name="obj">The object to test.</param>
        /// <returns>Whether its run-time type is a value tuple or a <see cref="Tuple"/>.</returns>
        /// <exception cref="NullReferenceException">When <paramref name="obj"/> is <see langword="null"/>.</exception>
        public static bool IsValueTuple(this object obj) => IsValueTupleType(obj.GetType());

        /// <summary>
        ///     Whether <paramref name="type"/> is a value tuple. Footgun: despite the name, reference tuples
        ///     (<see cref="Tuple"/>, matched by name) count too.
        /// </summary>
        /// <param name="type">The type to test.</param>
        /// <returns>Whether it is a <see cref="ValueTuple"/> of any arity or a <see cref="Tuple"/>.</returns>
        /// <exception cref="NullReferenceException">When <paramref name="type"/> is <see langword="null"/>.</exception>
        public static bool IsValueTupleType(this Type type) {
            return (type.GetTypeInfo().IsGenericType && ValueTupleTypes.Contains(type.GetGenericTypeDefinition())) || type.Name.StartsWith("Tuple`");
        }

        /// <summary>The item types of a tuple type, in order, as a list.</summary>
        /// <param name="tupleType">The tuple type.</param>
        /// <returns>See <see cref="GetTuplePropertyTypes"/>.</returns>
        public static List<Type> GetValueTupleItemTypes(this Type tupleType) => GetTuplePropertyTypes(tupleType).ToList();

        /// <summary>
        ///     A tuple's items, in order: a <see cref="Tuple"/>'s through its properties, a value tuple's through its
        ///     <c>Item1</c>, <c>Item2</c>, ... fields. Footgun: an eight-or-more value tuple yields its first seven items
        ///     only, since the rest sit in its <c>Rest</c> field.
        /// </summary>
        /// <param name="tupleType">The tuple (despite the parameter's name, the value, not its type).</param>
        /// <returns>The items, read lazily by reflection.</returns>
        /// <exception cref="NullReferenceException">When <paramref name="tupleType"/> is <see langword="null"/>.</exception>
        public static IEnumerable<object> IterateTuple(this object tupleType) {
            var type = tupleType.GetType();
            if (type.Name.StartsWith("Tuple`")) {
                foreach (var prop in type.GetProperties()) {
                    yield return prop.GetMethod.Invoke(tupleType, null);
                }
            } else {
                FieldInfo field;
                int nth = 1;
                while ((field = type.GetRuntimeField($"Item{nth}")) != null) {
                    nth++;
                    yield return field.GetValue(tupleType);
                }
            }
        }

        /// <summary>
        ///     A tuple type's item types, in order: a <see cref="Tuple"/>'s property types, a value tuple's <c>ItemN</c>
        ///     field types. Footgun: any type whose name starts with <c>Tuple</c> is read through its properties.
        /// </summary>
        /// <param name="tupleType">The tuple type.</param>
        /// <returns>The item types, read lazily by reflection.</returns>
        /// <exception cref="NullReferenceException">When <paramref name="tupleType"/> is <see langword="null"/>.</exception>
        public static IEnumerable<Type> GetTuplePropertyTypes(this Type tupleType) {
            if (tupleType.Name.StartsWith("Tuple")) {
                foreach (var prop in tupleType.GetProperties()) {
                    yield return prop.PropertyType;
                }
            } else {
                FieldInfo field;
                int nth = 1;
                while ((field = tupleType.GetRuntimeField($"Item{nth}")) != null) {
                    nth++;
                    yield return field.FieldType;
                }
            }
        }

        /// <summary>
        ///     The first <typeparamref name="T"/> attribute on an enum value's member, or <see langword="null"/>.
        ///     This extension method is broken out so you can use a similar pattern with
        ///     other MetaData elements in the future. This is your base method for each.
        /// </summary>
        /// <typeparam name="T">The attribute type.</typeparam>
        /// <param name="value">The enum value.</param>
        /// <returns>The attribute, or <see langword="null"/> when the member has none.</returns>
        /// <exception cref="IndexOutOfRangeException">For a value that names no member (a combination of flags, or an undefined number).</exception>
        public static T GetAttribute<T>(this Enum value) where T : Attribute {
            var type = value.GetType();
            var memberInfo = type.GetMember(value.ToString());
            var attributes = memberInfo[0].GetCustomAttributes(typeof(T), false);
            return attributes.Length > 0
                ? (T) attributes[0]
                : null;
        }

        /// <summary>Whether <paramref name="actualType"/> is <paramref name="expectedType"/> or assignable to it.</summary>
        /// <param name="actualType">The type to test.</param>
        /// <param name="expectedType">The type it should be or derive from.</param>
        /// <returns><see langword="true"/> when the types are equal or <paramref name="actualType"/> is assignable to <paramref name="expectedType"/>.</returns>
        /// <exception cref="NullReferenceException">When the types differ and <paramref name="expectedType"/> is <see langword="null"/>.</exception>
        public static bool IsSameOrInherits(this Type actualType, Type expectedType) {
            if (!(actualType == expectedType))
                return expectedType.IsAssignableFrom(actualType);
            return true;
        }

        /// <summary>
        ///     Checks if type <paramref name="toCheck"/> ihneriets type <paramref name="generic"/>. Taken from: https://stackoverflow.com/a/457708/1481186
        /// </summary>
        /// <param name="generic">The open generic type, e.g. <c>typeof(List&lt;&gt;)</c>.</param>
        /// <param name="toCheck">The type whose base classes to walk; interfaces are not checked.</param>
        /// <returns>Whether <paramref name="toCheck"/> or one of its base classes is a construction of <paramref name="generic"/>.</returns>
        public static bool IsSubclassOfRawGeneric(Type generic, Type toCheck) {
            while (toCheck != null && toCheck != typeof(object)) {
                var cur = toCheck.IsGenericType ? toCheck.GetGenericTypeDefinition() : toCheck;
                if (generic == cur) {
                    return true;
                }

                toCheck = toCheck.BaseType;
            }

            return false;
        }


        /// <summary>
        ///     Creates a delegate to a <see cref="MethodInfo"/>, typed <see cref="Func{TResult}"/>-like or <see cref="Action"/>-like
        ///     from its signature.
        /// </summary>
        /// <param name="methodInfo">The method.</param>
        /// <param name="target">The instance an instance method binds to; ignored for a static method.</param>
        /// <returns>The delegate.</returns>
        /// <exception cref="ArgumentException">
        ///     When the method cannot bind (for instance a <c>ref</c> parameter). Footgun: an instance method binds by name,
        ///     so an overloaded name may bind the overload matching the delegate's signature rather than this one.
        /// </exception>
        public static Delegate CreateDelegate(this MethodInfo methodInfo, object target) {
            Func<Type[], Type> getType;
            var isAction = methodInfo.ReturnType.Equals((typeof(void)));
            var types = methodInfo.GetParameters().Select(p => p.ParameterType);

            if (isAction) {
                getType = Expression.GetActionType;
            } else {
                getType = Expression.GetFuncType;
                types = types.Concat(new[] {methodInfo.ReturnType});
            }

            if (methodInfo.IsStatic) {
                return Delegate.CreateDelegate(getType(types.ToArray()), methodInfo);
            }

            return Delegate.CreateDelegate(getType(types.ToArray()), target, methodInfo.Name);
        }
    }
}
