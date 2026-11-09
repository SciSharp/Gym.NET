#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.ExceptionServices;
using Gym.Envs;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     An options record: an env's or wrapper's constructor arguments as typed properties (DR-11, DR-34), which
    ///     <c>make</c> accepts in place of a dictionary of keyword arguments.
    /// </summary>
    public interface IGymOptions {
        /// <summary>
        ///     The options as keyword arguments by Python name, holding only the properties that were set, so the record
        ///     overrides what it names and leaves the registered spec's other kwargs alone.
        /// </summary>
        /// <returns>A new dict the caller owns.</returns>
        Info ToKwargs();
    }

    /// <summary>
    ///     Python's keyword arguments for .NET constructors (DR-11): <see cref="From"/> turns what a caller passes (a dictionary,
    ///     an options record, an anonymous object) into a dict of Python names, and <see cref="Construct"/> is Python's
    ///     <c>cls(**kwargs)</c>, binding each name to a constructor parameter by its Python name with Python's
    ///     <see cref="TypeError"/> texts. <c>make</c> builds envs and wrappers through it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A parameter's Python name is its <see cref="PyNameAttribute"/>, else the snake_case of its C# name
    ///     (<c>suttonBartoReward</c> is <c>sutton_barto_reward</c>). Values are converted to the parameter's type where
    ///     Python's types map onto several .NET ones: any integer (a <see cref="BigInteger"/> from JSON included) to any
    ///     numeric type, a float to any floating type, a sequence to an array or list. Python would accept any value and
    ///     fail later, if at all; here a value that does not fit is a <see cref="TypeError"/> at the call.
    ///     </para>
    ///     <para>
    ///     A class can have several public constructors, Python one <c>__init__</c>: the constructors are tried from the one
    ///     with the most parameters down, and the first that binds every name is used. When none does, the error is the one
    ///     the constructor with the most parameters raised, the closest counterpart of Python's <c>__init__</c>.
    ///     </para>
    /// </remarks>
    public static class PyKwargs {
        /// <summary>
        ///     Python's <c>**kwargs</c> of a .NET value, as a new dict of Python names: a dictionary's entries as they are
        ///     (keys are already Python names), an <see cref="IGymOptions"/> record's set properties, or any other object's
        ///     public properties (an anonymous object), named by <see cref="PyNameAttribute"/> or snake_case.
        /// </summary>
        /// <param name="kwargs">The arguments, or <see langword="null"/> for none.</param>
        /// <returns>A new dict the caller owns; the values are shared, not copied.</returns>
        /// <exception cref="TypeError">
        ///     Thrown for a value that is no mapping (a number, a string, a list): <c>argument after ** must be a mapping, not
        ///     int</c>, Python's text; or for a dictionary key that is not a string: <c>keywords must be strings</c>.
        /// </exception>
        public static Info From(object? kwargs) {
            switch (kwargs) {
                case null:
                    return new Info();
                case IGymOptions options:
                    return options.ToKwargs();
                case IEnumerable<KeyValuePair<string, object?>> pairs:
                    return new Info(pairs);
                case IDictionary dictionary: {
                    var info = new Info();
                    foreach (DictionaryEntry entry in dictionary)
                        info[entry.Key as string ?? throw new TypeError("keywords must be strings")] = entry.Value;
                    return info;
                }
            }

            Type type = kwargs.GetType();
            // A dictionary typed by its values (Dictionary<string, int>, a read-only map) enumerates KeyValuePair<K, V>.
            if (KeyValueType(type) is { } pairType) {
                PropertyInfo key = pairType.GetProperty("Key")!, value = pairType.GetProperty("Value")!;
                var info = new Info();
                foreach (object? pair in (IEnumerable)kwargs)
                    info[key.GetValue(pair) as string ?? throw new TypeError("keywords must be strings")] = value.GetValue(pair);
                return info;
            }

            if (type.IsPrimitive || type.IsEnum || kwargs is string or decimal or BigInteger or IEnumerable or NDArray)
                throw new TypeError($"argument after ** must be a mapping, not {PyBuiltins.TypeName(kwargs)}");

            // An anonymous object or any other object: its readable public properties, in declaration order.
            var properties = new Info();
            foreach (PropertyInfo property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance)) {
                if (!property.CanRead || property.GetIndexParameters().Length > 0)
                    continue;
                string name = property.GetCustomAttribute<PyNameAttribute>(inherit: true)?.Name ?? PyAttributes.SnakeCase(property.Name);
                properties[name] = property.GetValue(kwargs);
            }

            return properties;
        }

        /// <summary>
        ///     Python's <c>cls(**kwargs)</c>: constructs <paramref name="type"/> with each keyword bound to the constructor
        ///     parameter of that Python name, defaults filling the rest. An open generic type (a wrapper's
        ///     <c>TimeLimit&lt;,&gt;</c>) is closed over the types its arguments imply: <c>env</c>'s <c>Env&lt;TObs, TAct&gt;</c>
        ///     closes a wrapper over the env's types.
        /// </summary>
        /// <param name="type">The class, closed or an open generic definition.</param>
        /// <param name="kwargs">The keyword arguments by Python name.</param>
        /// <returns>The new instance.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="type"/> or <paramref name="kwargs"/> is <see langword="null"/>.</exception>
        /// <exception cref="TypeError">
        ///     Thrown, with Python's texts, for an unexpected keyword (<c>CartPoleEnv.__init__() got an unexpected keyword
        ///     argument 'x'</c>) or a missing required one (<c>X.__init__() missing 1 required positional argument: 'y'</c>);
        ///     for an abstract class (<c>Can't instantiate abstract class X</c>); and for a value that does not fit its
        ///     parameter, or generic arguments that cannot be inferred.
        /// </exception>
        /// <exception cref="OverflowException">Thrown when a number does not fit its parameter's numeric type.</exception>
        /// <remarks>Whatever the constructor throws propagates as it was thrown, not wrapped in a reflection exception.</remarks>
        public static object Construct(Type type, IReadOnlyDictionary<string, object?> kwargs) {
            if (type is null)
                throw new ArgumentNullException(nameof(type));
            if (kwargs is null)
                throw new ArgumentNullException(nameof(kwargs));
            string owner = PyAttributes.PyTypeName(type);
            if (type.IsAbstract || type.IsInterface)
                throw new TypeError($"Can't instantiate abstract class {owner}");

            // The constructors kwargs can call (no ref, out or params), the fullest first; OrderBy is stable, so equal
            // counts keep their declaration order.
            ConstructorInfo[] constructors = type.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .Where(c => c.GetParameters().All(p => !p.ParameterType.IsByRef && !p.IsDefined(typeof(ParamArrayAttribute))))
                .OrderByDescending(c => c.GetParameters().Length)
                .ToArray();
            if (constructors.Length == 0)
                throw new TypeError(kwargs.Count == 0 ? $"cannot create '{owner}' instances" : $"{owner}() takes no arguments");

            Exception? primaryError = null;
            foreach (ConstructorInfo constructor in constructors) {
                if (TryBind(type, constructor, kwargs, owner, out ConstructorInfo? bound, out object?[]? arguments, out Exception? error))
                    return Invoke(bound!, arguments!);
                primaryError ??= error;
            }

            ExceptionDispatchInfo.Capture(primaryError!).Throw();
            throw primaryError!;
        }

        /// <summary>A constructor parameter's Python name: its <see cref="PyNameAttribute"/>, else the snake_case of its C# name.</summary>
        /// <param name="parameter">The parameter.</param>
        /// <returns>The keyword that binds to it.</returns>
        public static string ParameterName(ParameterInfo parameter) =>
            parameter.GetCustomAttribute<PyNameAttribute>()?.Name ?? PyAttributes.SnakeCase(parameter.Name ?? "");

        /// <summary>
        ///     Binds <paramref name="kwargs"/> to one constructor, as Python binds a call to <c>__init__</c>: unexpected names
        ///     first, then missing required ones, then the generic arguments and the value conversions.
        /// </summary>
        /// <param name="type">The class, possibly an open generic definition.</param>
        /// <param name="constructor">The constructor tried.</param>
        /// <param name="kwargs">The keyword arguments.</param>
        /// <param name="owner">The class name, for the messages.</param>
        /// <param name="bound">The constructor to invoke (of the closed type), when binding succeeds.</param>
        /// <param name="arguments">The converted arguments in parameter order, when binding succeeds.</param>
        /// <param name="error">The <see cref="TypeError"/> describing why binding failed.</param>
        /// <returns><see langword="true"/> when the constructor can be called with the keywords.</returns>
        private static bool TryBind(Type type, ConstructorInfo constructor, IReadOnlyDictionary<string, object?> kwargs, string owner,
            out ConstructorInfo? bound, out object?[]? arguments, out Exception? error) {
            bound = null;
            arguments = null;
            ParameterInfo[] parameters = constructor.GetParameters();
            string[] names = parameters.Select(ParameterName).ToArray();

            foreach (string key in kwargs.Keys) {
                if (Array.IndexOf(names, key) < 0) {
                    error = new TypeError($"{owner}.__init__() got an unexpected keyword argument '{key}'");
                    return false;
                }
            }

            var missing = new List<string>();
            for (int i = 0; i < parameters.Length; i++) {
                if (!kwargs.ContainsKey(names[i]) && !parameters[i].HasDefaultValue && !parameters[i].IsOptional)
                    missing.Add(names[i]);
            }

            if (missing.Count > 0) {
                error = new TypeError($"{owner}.__init__() missing {missing.Count} required positional argument{(missing.Count == 1 ? "" : "s")}: {JoinNames(missing)}");
                return false;
            }

            ConstructorInfo target = constructor;
            if (type.ContainsGenericParameters) {
                if (!TryCloseGeneric(type, parameters, names, kwargs, owner, out Type? closed, out error))
                    return false;
                target = (ConstructorInfo)MethodBase.GetMethodFromHandle(constructor.MethodHandle, closed!.TypeHandle)!;
                parameters = target.GetParameters();
            }

            var values = new object?[parameters.Length];
            for (int i = 0; i < parameters.Length; i++) {
                if (!kwargs.TryGetValue(names[i], out object? value)) {
                    values[i] = DefaultValue(parameters[i]);
                    continue;
                }

                if (!TryFit(value, parameters[i].ParameterType, out object? fitted)) {
                    error = new TypeError($"{owner}.__init__() argument '{names[i]}' must be {TypeDescription(parameters[i].ParameterType)}, not {PyBuiltins.TypeName(value)}");
                    return false;
                }

                values[i] = fitted;
            }

            bound = target;
            arguments = values;
            error = null;
            return true;
        }

        /// <summary>Invokes a constructor, rethrowing what it throws rather than reflection's wrapper.</summary>
        /// <param name="constructor">The constructor.</param>
        /// <param name="arguments">The arguments.</param>
        /// <returns>The instance.</returns>
        private static object Invoke(ConstructorInfo constructor, object?[] arguments) {
            try {
                return constructor.Invoke(arguments);
            } catch (TargetInvocationException e) when (e.InnerException is not null) {
                ExceptionDispatchInfo.Capture(e.InnerException).Throw();
                throw;
            }
        }

        /// <summary>
        ///     Closes an open generic class over the types its arguments imply: a constructed parameter type
        ///     (<c>Env&lt;TObs, TAct&gt;</c>) is matched against the argument's class and its bases first, then a bare type
        ///     parameter (<c>T value</c>) takes its argument's class if still open.
        /// </summary>
        /// <param name="definition">The open generic definition.</param>
        /// <param name="parameters">The constructor's parameters.</param>
        /// <param name="names">Their Python names.</param>
        /// <param name="kwargs">The keyword arguments.</param>
        /// <param name="owner">The class name, for the messages.</param>
        /// <param name="closed">The closed class.</param>
        /// <param name="error">Why it could not be closed.</param>
        /// <returns><see langword="true"/> when every type parameter was inferred and satisfies its constraints.</returns>
        private static bool TryCloseGeneric(Type definition, ParameterInfo[] parameters, string[] names, IReadOnlyDictionary<string, object?> kwargs, string owner,
            out Type? closed, out Exception? error) {
            closed = null;
            var bindings = new Dictionary<Type, Type>();
            for (int pass = 0; pass < 2; pass++) {
                for (int i = 0; i < parameters.Length; i++) {
                    if (!kwargs.TryGetValue(names[i], out object? value) || value is null)
                        continue;
                    Type pattern = parameters[i].ParameterType;
                    if (pass == 0 && !pattern.IsGenericParameter && pattern.ContainsGenericParameters) {
                        if (!Unify(pattern, value.GetType(), bindings)) {
                            error = new TypeError($"{owner}.__init__() argument '{names[i]}' must be {TypeDescription(pattern)}, not {PyBuiltins.TypeName(value)}");
                            return false;
                        }
                    } else if (pass == 1 && pattern.IsGenericParameter && !bindings.ContainsKey(pattern)) {
                        bindings[pattern] = value.GetType();
                    }
                }
            }

            Type[] typeParameters = definition.GetGenericArguments();
            string[] open = typeParameters.Where(t => !bindings.ContainsKey(t)).Select(t => t.Name).ToArray();
            if (open.Length > 0) {
                error = new TypeError($"{owner}.__init__() cannot infer the type arguments {string.Join(", ", open)} from the keyword arguments");
                return false;
            }

            try {
                closed = definition.MakeGenericType(typeParameters.Select(t => bindings[t]).ToArray());
            } catch (ArgumentException e) {
                // A constraint the inferred types do not meet.
                error = new TypeError($"{owner}.__init__() got arguments of types its type parameters do not accept: {e.Message}");
                return false;
            }

            error = null;
            return true;
        }

        /// <summary>
        ///     Matches a parameter type that mentions type parameters against an argument's class, recording what each type
        ///     parameter stands for; the bindings change only when the whole match succeeds.
        /// </summary>
        /// <param name="pattern">The parameter type (<c>Env&lt;TObs, TAct&gt;</c>, <c>T[]</c>, <c>T</c>, or a closed type).</param>
        /// <param name="actual">The argument's class.</param>
        /// <param name="bindings">The type parameters bound so far.</param>
        /// <returns><see langword="true"/> when the argument fits the pattern.</returns>
        private static bool Unify(Type pattern, Type actual, Dictionary<Type, Type> bindings) {
            if (pattern.IsGenericParameter) {
                if (bindings.TryGetValue(pattern, out Type? existing))
                    return existing.IsAssignableFrom(actual);
                bindings[pattern] = actual;
                return true;
            }

            if (!pattern.ContainsGenericParameters)
                return pattern.IsAssignableFrom(actual);
            if (pattern.IsArray)
                return actual.IsArray && actual.GetArrayRank() == pattern.GetArrayRank() && Unify(pattern.GetElementType()!, actual.GetElementType()!, bindings);
            if (!pattern.IsGenericType)
                return false;

            Type definition = pattern.GetGenericTypeDefinition();
            Type[] patternArguments = pattern.GetGenericArguments();
            foreach (Type candidate in SelfAndAncestors(actual)) {
                if (!candidate.IsGenericType || candidate.GetGenericTypeDefinition() != definition)
                    continue;
                var trial = new Dictionary<Type, Type>(bindings);
                Type[] actualArguments = candidate.GetGenericArguments();
                bool matched = true;
                for (int i = 0; i < patternArguments.Length && matched; i++)
                    matched = UnifyArgument(patternArguments[i], actualArguments[i], trial);

                if (!matched)
                    continue;
                foreach (KeyValuePair<Type, Type> binding in trial)
                    bindings[binding.Key] = binding.Value;
                return true;
            }

            return false;
        }

        /// <summary>
        ///     Matches one type argument of a constructed generic pattern. Type arguments are invariant, so a type parameter
        ///     takes the exact argument (or must already equal it), a closed argument must be the same type, and a nested
        ///     generic argument recurses.
        /// </summary>
        /// <param name="pattern">The pattern's type argument.</param>
        /// <param name="actual">The matching type argument of the argument's class.</param>
        /// <param name="bindings">The type parameters bound so far, extended on success.</param>
        /// <returns><see langword="true"/> when the type arguments agree.</returns>
        private static bool UnifyArgument(Type pattern, Type actual, Dictionary<Type, Type> bindings) {
            if (pattern.IsGenericParameter) {
                if (bindings.TryGetValue(pattern, out Type? bound))
                    return bound == actual;
                bindings[pattern] = actual;
                return true;
            }

            if (!pattern.ContainsGenericParameters)
                return pattern == actual;
            return Unify(pattern, actual, bindings);
        }

        /// <summary>A class, its base classes, then the interfaces it implements: where a constructed generic pattern may match.</summary>
        /// <param name="type">The class.</param>
        /// <returns>The types, most derived first.</returns>
        private static IEnumerable<Type> SelfAndAncestors(Type type) {
            for (Type? level = type; level != null; level = level.BaseType)
                yield return level;
            foreach (Type implemented in type.GetInterfaces())
                yield return implemented;
        }

        /// <summary>
        ///     Makes a keyword's value fit its parameter: as it is when assignable; Python's numbers converted between .NET's
        ///     numeric types (an integer to any number, a float to any floating type); an integer to an enum; a sequence to an
        ///     array or list of the element type.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="type">The parameter type.</param>
        /// <param name="fitted">The value to pass.</param>
        /// <returns><see langword="false"/> when the value cannot be passed for that type.</returns>
        /// <exception cref="OverflowException">Thrown when a number is out of the parameter type's range.</exception>
        private static bool TryFit(object? value, Type type, out object? fitted) {
            fitted = null;
            Type target = Nullable.GetUnderlyingType(type) ?? type;
            if (value is null)
                return !type.IsValueType || target != type;
            if (target.IsInstanceOfType(value)) {
                fitted = value;
                return true;
            }

            if (IsInteger(value)) {
                if (target.IsEnum) {
                    fitted = Enum.ToObject(target, value is BigInteger big ? (long)big : value);
                    return true;
                }

                if (IsNumberType(target)) {
                    fitted = value is BigInteger integer ? ConvertBigInteger(integer, target) : Convert.ChangeType(value, target, CultureInfo.InvariantCulture);
                    return true;
                }
            }

            if (value is double or float or Half && (target == typeof(double) || target == typeof(float) || target == typeof(Half) || target == typeof(decimal))) {
                fitted = value is Half half ? Convert.ChangeType((float)half, target, CultureInfo.InvariantCulture) : target == typeof(Half) ? (Half)Convert.ToDouble(value, CultureInfo.InvariantCulture) : Convert.ChangeType(value, target, CultureInfo.InvariantCulture);
                return true;
            }

            if (value is IEnumerable sequence and not string && CollectionElementType(target) is { } elementType) {
                var items = new List<object?>();
                foreach (object? item in sequence) {
                    if (!TryFit(item, elementType, out object? fittedItem))
                        return false;
                    items.Add(fittedItem);
                }

                if (target.IsArray) {
                    var array = Array.CreateInstance(elementType, items.Count);
                    for (int i = 0; i < items.Count; i++)
                        array.SetValue(items[i], i);
                    fitted = array;
                } else {
                    var list = (IList)Activator.CreateInstance(typeof(List<>).MakeGenericType(elementType))!;
                    foreach (object? item in items)
                        list.Add(item);
                    fitted = list;
                }

                return true;
            }

            return false;
        }

        /// <summary>The element type of an array or of a list-like type a <see cref="List{T}"/> can stand in for, or <see langword="null"/>.</summary>
        /// <param name="type">The parameter type.</param>
        /// <returns>The element type.</returns>
        private static Type? CollectionElementType(Type type) {
            if (type.IsArray && type.GetArrayRank() == 1)
                return type.GetElementType();
            if (!type.IsGenericType)
                return null;
            Type definition = type.GetGenericTypeDefinition();
            bool listLike = definition == typeof(List<>) || definition == typeof(IList<>) || definition == typeof(IReadOnlyList<>) || definition == typeof(ICollection<>)
                            || definition == typeof(IReadOnlyCollection<>) || definition == typeof(IEnumerable<>);
            return listLike ? type.GetGenericArguments()[0] : null;
        }

        /// <summary>A Python int too large for <see langword="long"/> perhaps, converted to a numeric type, range-checked.</summary>
        /// <param name="value">The integer.</param>
        /// <param name="target">The numeric type.</param>
        /// <returns>The converted value.</returns>
        /// <exception cref="OverflowException">Thrown when the value does not fit.</exception>
        private static object ConvertBigInteger(BigInteger value, Type target) {
            if (target == typeof(double))
                return (double)value;
            if (target == typeof(float))
                return (float)value;
            if (target == typeof(Half))
                return (Half)(double)value;
            if (target == typeof(decimal))
                return (decimal)value;
            if (target == typeof(ulong))
                return (ulong)value;
            return Convert.ChangeType((long)value, target, CultureInfo.InvariantCulture);
        }

        /// <summary>Whether a value is a Python int: a .NET integer or a <see cref="BigInteger"/> (not a <see langword="bool"/>, which a bool parameter takes as is).</summary>
        /// <param name="value">The value.</param>
        /// <returns><see langword="true"/> for an integer.</returns>
        private static bool IsInteger(object value) => value is sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger;

        /// <summary>Whether a type is one of .NET's numeric types a Python number converts to.</summary>
        /// <param name="type">The type.</param>
        /// <returns><see langword="true"/> for the integer and floating types and <see langword="decimal"/>.</returns>
        private static bool IsNumberType(Type type) =>
            type == typeof(sbyte) || type == typeof(byte) || type == typeof(short) || type == typeof(ushort) || type == typeof(int) || type == typeof(uint)
            || type == typeof(long) || type == typeof(ulong) || type == typeof(float) || type == typeof(double) || type == typeof(Half) || type == typeof(decimal);

        /// <summary>The value an unbound parameter takes: its declared default, or the type's default for an <c>[Optional]</c> one.</summary>
        /// <param name="parameter">The parameter.</param>
        /// <returns>The value to pass.</returns>
        private static object? DefaultValue(ParameterInfo parameter) {
            Type type = parameter.ParameterType;
            if (!parameter.HasDefaultValue)
                return type.IsValueType ? Activator.CreateInstance(type) : null;
            object? value = parameter.DefaultValue;
            // Metadata stores an enum default as its underlying integer.
            Type target = Nullable.GetUnderlyingType(type) ?? type;
            if (value is not null && target.IsEnum && !target.IsInstanceOfType(value))
                return Enum.ToObject(target, value);
            return value;
        }

        /// <summary>A .NET type as the messages name it: <c>long</c>'s <c>Int64</c>, a generic one with its arguments.</summary>
        /// <param name="type">The type.</param>
        /// <returns>The name.</returns>
        private static string TypeDescription(Type type) {
            if (Nullable.GetUnderlyingType(type) is { } underlying)
                return $"{TypeDescription(underlying)} or None";
            if (!type.IsGenericType)
                return type.Name;
            return $"{PyAttributes.PyTypeName(type)}<{string.Join(", ", type.GetGenericArguments().Select(TypeDescription))}>";
        }

        /// <summary>Python's list of missing argument names: <c>'a'</c>, <c>'a' and 'b'</c>, <c>'a', 'b', and 'c'</c>.</summary>
        /// <param name="names">The names.</param>
        /// <returns>The text.</returns>
        private static string JoinNames(IReadOnlyList<string> names) {
            string[] quoted = names.Select(n => $"'{n}'").ToArray();
            return quoted.Length switch {
                1 => quoted[0],
                2 => $"{quoted[0]} and {quoted[1]}",
                _ => $"{string.Join(", ", quoted, 0, quoted.Length - 1)}, and {quoted[^1]}",
            };
        }

        /// <summary>The <c>KeyValuePair&lt;string, V&gt;</c> a dictionary-like type enumerates, or <see langword="null"/>.</summary>
        /// <param name="type">The type.</param>
        /// <returns>The pair type.</returns>
        private static Type? KeyValueType(Type type) =>
            type.GetInterfaces()
                .Where(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IEnumerable<>))
                .Select(i => i.GetGenericArguments()[0])
                .FirstOrDefault(pair => pair.IsGenericType && pair.GetGenericTypeDefinition() == typeof(KeyValuePair<,>));
    }
}
