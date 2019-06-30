using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace Gym.Reflection {
    public static class ReflectionHelper {
        //private static readonly Func<T> DefaultConstructor1;
        //
        //public static Func<T> DefaultConstructor {
        //    get {
        //        if ()
        //        return DefaultConstructor1;
        //    }
        //}

        public static ConstructorInfo GetDefaultConstructor<TType>() {
            var type = typeof(TType);
            return type.GetDefaultConstructor();
        }

        /// <summary>
        ///   Gets the default value for a type. This method should serve as
        ///   a programmatic equivalent to <c>default(T)</c>.
        /// </summary>
        /// <param name="type">The type whose default value should be retrieved.</param>
        public static object GetDefaultValue(this Type type) {
            if (type.IsValueType)
                return Activator.CreateInstance(type);
            return GetDefaultConstructor(type).Invoke(null, new object[0]);
        }

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
        /// <exception cref="InvalidOperationException">When <paramref name="ci"/> contains non-optional (<see cref="ParameterInfo.IsOptional"/>) parameters.</exception>
        /// <returns></returns>
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
        /// <typeparam name="T">The time of <paramref name="ci"/></typeparam>
        /// <param name="ci">The constructor to wrap.</param>
        /// <exception cref="InvalidOperationException">When <paramref name="ci"/> contains non-optional (<see cref="ParameterInfo.IsOptional"/>) parameters.</exception>
        /// <returns></returns>
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
        public static bool HasInterface<T>(this Type type) {
            return typeof(T).IsAssignableFrom(type);
        }

        /// <summary>
        ///     Checks if the type/object has this interface.
        /// </summary>
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
        public static string GetDescription<T>(this T source) where T : Enum {
            DescriptionAttribute[] customAttributes = (DescriptionAttribute[]) source.GetType().GetField(source.ToString()).GetCustomAttributes(typeof(DescriptionAttribute), false);
            if (customAttributes != null && customAttributes.Length != 0)
                return customAttributes[0].Description;
            return source.ToString();
        }

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

        public static bool IsValueTuple(this object obj) => IsValueTupleType(obj.GetType());

        public static bool IsValueTupleType(this Type type) {
            return (type.GetTypeInfo().IsGenericType && ValueTupleTypes.Contains(type.GetGenericTypeDefinition())) || type.Name.StartsWith("Tuple`");
        }

        public static List<Type> GetValueTupleItemTypes(this Type tupleType) => GetTuplePropertyTypes(tupleType).ToList();

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

        /// This extension method is broken out so you can use a similar pattern with 
        /// other MetaData elements in the future. This is your base method for each.
        public static T GetAttribute<T>(this Enum value) where T : Attribute {
            var type = value.GetType();
            var memberInfo = type.GetMember(value.ToString());
            var attributes = memberInfo[0].GetCustomAttributes(typeof(T), false);
            return attributes.Length > 0
                ? (T) attributes[0]
                : null;
        }

        public static bool IsSameOrInherits(this Type actualType, Type expectedType) {
            if (!(actualType == expectedType))
                return expectedType.IsAssignableFrom(actualType);
            return true;
        }

        /// <summary>
        ///     Checks if type <see cref="toCheck"/> ihneriets type <see cref="generic"/>. Taken from: https://stackoverflow.com/a/457708/1481186
        /// </summary>
        /// <param name="generic"></param>
        /// <param name="toCheck"></param>
        /// <returns></returns>
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
        ///     Creates a delegate to a <see cref="MethodInfo"/>.
        /// </summary>
        /// <param name="methodInfo"></param>
        /// <param name="target"></param>
        /// <returns></returns>
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