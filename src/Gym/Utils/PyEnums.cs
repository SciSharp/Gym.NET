#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Marks a C# enum that stands for Python values (DR-84): either the members of a Python <c>Enum</c> class
    ///     (<see cref="Gym.Vector.AutoresetMode"/> is Gymnasium's <c>AutoresetMode</c>, printed
    ///     <c>&lt;AutoresetMode.NEXT_STEP: 'NextStep'&gt;</c>), or plain Python strings that a closed set of values takes
    ///     (<see cref="Gym.Envs.RenderMode"/>, whose members are Gymnasium's render-mode strings). Each member carries its
    ///     Python value in a <see cref="PyValueAttribute"/> and, for an <c>Enum</c> class, its Python name in a
    ///     <see cref="PyNameAttribute"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <see cref="PyEnums"/> reads the attributes once per type. <see cref="PyBuiltins"/> prints such a member as Python
    ///     prints its counterpart, <see cref="PyEquality"/> compares it as Python compares its counterpart (an <c>Enum</c>
    ///     member only with itself, a string member with the equal string), and <see cref="PyKwargs.From"/> passes a string
    ///     member on as its string, so specs and kwargs keep Gymnasium's values.
    ///     </para>
    ///     <para>
    ///     A C# enum without this attribute is an integer to Python (an <c>IntEnum</c>, a <c>Discrete</c> value), as before.
    ///     </para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Enum, AllowMultiple = false, Inherited = false)]
    public sealed class PyEnumAttribute : Attribute {
        /// <summary>Marks the enum.</summary>
        /// <param name="className">
        ///     The Python <c>Enum</c> class the members belong to (<c>"AutoresetMode"</c>), printed in their <c>str</c> and
        ///     <c>repr</c>; <see langword="null"/> (the default) when each member stands for a plain Python <c>str</c>, its
        ///     <see cref="PyValueAttribute"/> text.
        /// </param>
        public PyEnumAttribute(string? className = null) => ClassName = className;

        /// <summary>The Python <c>Enum</c> class's name, or <see langword="null"/> for members that are plain strings in Python.</summary>
        public string? ClassName { get; }
    }

    /// <summary>
    ///     The Python value of a <see cref="PyEnumAttribute"/> enum's member: the <c>.value</c> of a Python <c>Enum</c> member
    ///     (<c>"NextStep"</c>), or the string a string member stands for (<c>"rgb_array"</c>).
    /// </summary>
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
    public sealed class PyValueAttribute : Attribute {
        /// <summary>Gives the member its Python value.</summary>
        /// <param name="value">The value, spelled exactly as in the Gymnasium source.</param>
        public PyValueAttribute(string value) => Value = value;

        /// <summary>The Python value.</summary>
        public string Value { get; }
    }

    /// <summary>
    ///     Python's view of the C# enums marked <see cref="PyEnumAttribute"/> (DR-84): a member's Python <c>.name</c>,
    ///     <c>.value</c>, <c>str</c> and <c>repr</c>, parsing a member from its value as Python's <c>Enum(value)</c> call does,
    ///     and the members in declaration order.
    /// </summary>
    /// <remarks>
    ///     The attributes are read by reflection once per enum type and cached, so the calls cost a dictionary lookup.
    ///     Every member of such an enum must carry a <see cref="PyValueAttribute"/>: a type missing one is refused at its
    ///     first use with an <see cref="InvalidOperationException"/>, a programming error in the enum, not in the caller.
    /// </remarks>
    public static class PyEnums {
        /// <summary>The cached description of each enum type asked about, <see langword="null"/> for one without <see cref="PyEnumAttribute"/>.</summary>
        private static readonly ConcurrentDictionary<Type, Table?> Tables = new ConcurrentDictionary<Type, Table?>();

        /// <summary>Whether <paramref name="type"/> is a C# enum marked <see cref="PyEnumAttribute"/>.</summary>
        /// <param name="type">The type; <see langword="null"/> is not one.</param>
        /// <returns><see langword="true"/> for a Python-valued enum.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the type is marked but a member has no <see cref="PyValueAttribute"/>.</exception>
        public static bool IsPyEnum(Type? type) => type is not null && TableOf(type) is not null;

        /// <summary>Whether <paramref name="type"/> is a marked enum whose members stand for plain Python strings (no <c>Enum</c> class).</summary>
        /// <param name="type">The type; <see langword="null"/> is not one.</param>
        /// <returns><see langword="true"/> for a string enum such as <see cref="Gym.Envs.RenderMode"/>.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the type is marked but a member has no <see cref="PyValueAttribute"/>.</exception>
        public static bool IsStringEnum(Type? type) => type is not null && TableOf(type) is { ClassName: null };

        /// <summary>Python's <c>.value</c> of a member: <c>"NextStep"</c>, or the string a string member stands for (<c>"rgb_array"</c>).</summary>
        /// <param name="member">The member.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="member"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the member's enum is not marked <see cref="PyEnumAttribute"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare (<c>(AutoresetMode)7</c>).</exception>
        public static string Value(Enum member) => Entry(member).Value;

        /// <summary>
        ///     Python's <c>.name</c> of a member: the constant's name in its <c>Enum</c> class (<c>NEXT_STEP</c>), from its
        ///     <see cref="PyNameAttribute"/>; a string member's own value, the only name Python has for it.
        /// </summary>
        /// <param name="member">The member.</param>
        /// <returns>The name.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="member"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the member's enum is not marked <see cref="PyEnumAttribute"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        public static string Name(Enum member) => Entry(member).Name;

        /// <summary>
        ///     Python's <c>str(member)</c>, what an f-string shows: <c>AutoresetMode.SAME_STEP</c> for an <c>Enum</c> member,
        ///     the string itself for a string member. Gymnasium's messages that format a member with an f-string use this.
        /// </summary>
        /// <param name="member">The member.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="member"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the member's enum is not marked <see cref="PyEnumAttribute"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        public static string Str(Enum member) {
            (Table table, (string Name, string Value) entry) = Lookup(member);
            return table.ClassName is null ? entry.Value : $"{table.ClassName}.{entry.Name}";
        }

        /// <summary>
        ///     Python's <c>repr(member)</c>, what a printed dict shows: <c>&lt;AutoresetMode.NEXT_STEP: 'NextStep'&gt;</c> for an
        ///     <c>Enum</c> member, the quoted string (<c>'rgb_array'</c>) for a string member.
        /// </summary>
        /// <param name="member">The member.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="member"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the member's enum is not marked <see cref="PyEnumAttribute"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        public static string Repr(Enum member) {
            (Table table, (string Name, string Value) entry) = Lookup(member);
            return table.ClassName is null ? PyBuiltins.StrRepr(entry.Value) : $"<{table.ClassName}.{entry.Name}: {PyBuiltins.StrRepr(entry.Value)}>";
        }

        /// <summary>
        ///     Python's <c>Enum(value)</c>: the member whose value is <paramref name="value"/>. Only the value is accepted
        ///     (<c>"NextStep"</c>, <c>"rgb_array"</c>), never the member's name (<c>"NEXT_STEP"</c>, <c>"RgbArray"</c>), as
        ///     Python's call refuses names.
        /// </summary>
        /// <typeparam name="TEnum">The enum, marked <see cref="PyEnumAttribute"/>.</typeparam>
        /// <param name="value">The value.</param>
        /// <returns>The member.</returns>
        /// <exception cref="ValueError">Thrown for any other value, with Python's text: <c>'x' is not a valid AutoresetMode</c>.</exception>
        /// <exception cref="ArgumentException">Thrown when <typeparamref name="TEnum"/> is not marked <see cref="PyEnumAttribute"/>.</exception>
        public static TEnum Of<TEnum>(string? value) where TEnum : struct, Enum {
            if (TryOf(value, out TEnum member))
                return member;
            throw new ValueError($"{PyBuiltins.Repr(value)} is not a valid {DisplayName(typeof(TEnum))}");
        }

        /// <summary>Python's <c>Enum(value)</c> without the exception: the member whose value is <paramref name="value"/>, if any.</summary>
        /// <typeparam name="TEnum">The enum, marked <see cref="PyEnumAttribute"/>.</typeparam>
        /// <param name="value">The value; <see langword="null"/> matches no member.</param>
        /// <param name="member">The member, when <see langword="true"/> is returned; the enum's default otherwise.</param>
        /// <returns><see langword="true"/> when a member has that value.</returns>
        /// <exception cref="ArgumentException">Thrown when <typeparamref name="TEnum"/> is not marked <see cref="PyEnumAttribute"/>.</exception>
        public static bool TryOf<TEnum>(string? value, out TEnum member) where TEnum : struct, Enum {
            Table table = RequireTable(typeof(TEnum));
            if (value is not null && table.ByValue.TryGetValue(value, out Enum? found)) {
                member = (TEnum)found;
                return true;
            }

            member = default;
            return false;
        }

        /// <summary>The members in declaration order, which is Gymnasium's order for its <c>Enum</c> classes (what <c>list(AutoresetMode)</c> gives).</summary>
        /// <typeparam name="TEnum">The enum, marked <see cref="PyEnumAttribute"/>.</typeparam>
        /// <returns>The members; a shared, read-only list.</returns>
        /// <exception cref="ArgumentException">Thrown when <typeparamref name="TEnum"/> is not marked <see cref="PyEnumAttribute"/>.</exception>
        public static IReadOnlyList<TEnum> Members<TEnum>() where TEnum : struct, Enum {
            // Checked here, before the cache class initializes: a throw from its static initializer would surface as a
            // TypeInitializationException and poison the type for the process.
            RequireTable(typeof(TEnum));
            return MembersOf<TEnum>.All;
        }

        /// <summary>
        ///     The value Python holds for <paramref name="value"/>: a string member becomes its string (<c>RenderMode.Human</c>
        ///     is <c>"human"</c>); anything else, an <c>Enum</c> member included, is returned as it is. <c>make</c>'s kwargs go
        ///     through it, so a spec records <c>render_mode='human'</c> as Gymnasium's does.
        /// </summary>
        /// <param name="value">Any value.</param>
        /// <returns>The value as Python holds it.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a string enum's value it does not declare (<c>(RenderMode)99</c>).</exception>
        public static object? ToPython(object? value) =>
            value is Enum member && TableOf(member.GetType()) is { ClassName: null } table ? EntryIn(table, member).Value : value;

        /// <summary>The cached table of a marked enum type, or <see langword="null"/> for any other type.</summary>
        /// <param name="type">The type.</param>
        /// <returns>The table.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the type is marked but a member has no <see cref="PyValueAttribute"/>.</exception>
        private static Table? TableOf(Type type) => Tables.GetOrAdd(type, static t => Build(t));

        /// <summary>The table of a type that must be a marked enum.</summary>
        /// <param name="type">The type.</param>
        /// <returns>The table.</returns>
        /// <exception cref="ArgumentException">Thrown when the type is not a marked enum.</exception>
        private static Table RequireTable(Type type) =>
            TableOf(type) ?? throw new ArgumentException($"{type.FullName} is not an enum marked [PyEnum].", nameof(type));

        /// <summary>A member's table and its name and value.</summary>
        /// <param name="member">The member.</param>
        /// <returns>The table and the entry.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="member"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the member's enum is not marked.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        private static (Table Table, (string Name, string Value) Entry) Lookup(Enum member) {
            if (member is null)
                throw new ArgumentNullException(nameof(member));
            Table table = RequireTable(member.GetType());
            return (table, EntryIn(table, member));
        }

        /// <summary>A member's name and value.</summary>
        /// <param name="member">The member.</param>
        /// <returns>The entry.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="member"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the member's enum is not marked.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare.</exception>
        private static (string Name, string Value) Entry(Enum member) => Lookup(member).Entry;

        /// <summary>A member's entry in its table.</summary>
        /// <param name="table">The member's table.</param>
        /// <param name="member">The member.</param>
        /// <returns>The entry.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value the enum does not declare: a cast integer has no Python value.</exception>
        private static (string Name, string Value) EntryIn(Table table, Enum member) =>
            table.Entries.TryGetValue(member, out (string Name, string Value) entry)
                ? entry
                : throw new ArgumentOutOfRangeException(nameof(member), member, $"{member.GetType().Name} declares no member with this value, so it has no Python value.");

        /// <summary>The name Python's messages give a marked enum: its <c>Enum</c> class, else the C# name (<c>RenderMode</c>).</summary>
        /// <param name="type">The enum type.</param>
        /// <returns>The name.</returns>
        private static string DisplayName(Type type) => TableOf(type)?.ClassName ?? type.Name;

        /// <summary>Reads a type's attributes into a table, or gives <see langword="null"/> for a type that is not a marked enum.</summary>
        /// <param name="type">The type.</param>
        /// <returns>The table.</returns>
        /// <exception cref="InvalidOperationException">Thrown when a member of a marked enum has no <see cref="PyValueAttribute"/>.</exception>
        private static Table? Build(Type type) {
            if (!type.IsEnum || type.GetCustomAttribute<PyEnumAttribute>(inherit: false) is not { } marker)
                return null;
            var members = new List<Enum>();
            var entries = new Dictionary<Enum, (string Name, string Value)>();
            var byValue = new Dictionary<string, Enum>(StringComparer.Ordinal);
            // GetFields returns the constants in declaration order in practice (metadata order), which Gymnasium's Enum
            // iteration order is: the order the members were written in.
            foreach (FieldInfo field in type.GetFields(BindingFlags.Public | BindingFlags.Static)) {
                string value = field.GetCustomAttribute<PyValueAttribute>()?.Value
                               ?? throw new InvalidOperationException($"{type.FullName}.{field.Name} has no [PyValue]: every member of a [PyEnum] enum needs its Python value.");
                string name = field.GetCustomAttribute<PyNameAttribute>()?.Name ?? value;
                var member = (Enum)field.GetValue(null)!;
                members.Add(member);
                entries[member] = (name, value);
                byValue[value] = member;
            }

            return new Table(marker.ClassName, members.ToArray(), entries, byValue);
        }

        /// <summary>A marked enum's description: its Python class (or none), its members in order, and each member's Python name and value.</summary>
        private sealed class Table {
            /// <summary>Creates the table.</summary>
            /// <param name="className">The Python <c>Enum</c> class, or <see langword="null"/> for string members.</param>
            /// <param name="members">The members in declaration order.</param>
            /// <param name="entries">Each member's Python name and value.</param>
            /// <param name="byValue">The member of each Python value.</param>
            public Table(string? className, Enum[] members, Dictionary<Enum, (string Name, string Value)> entries, Dictionary<string, Enum> byValue) {
                ClassName = className;
                Members = members;
                Entries = entries;
                ByValue = byValue;
            }

            /// <summary>The Python <c>Enum</c> class, or <see langword="null"/> for string members.</summary>
            public string? ClassName { get; }

            /// <summary>The members in declaration order.</summary>
            public Enum[] Members { get; }

            /// <summary>Each member's Python name and value.</summary>
            public Dictionary<Enum, (string Name, string Value)> Entries { get; }

            /// <summary>The member of each Python value.</summary>
            public Dictionary<string, Enum> ByValue { get; }
        }

        /// <summary>The typed member list of one enum, built once from its table.</summary>
        /// <typeparam name="TEnum">The enum.</typeparam>
        private static class MembersOf<TEnum> where TEnum : struct, Enum {
            /// <summary>The members in declaration order.</summary>
            public static readonly IReadOnlyList<TEnum> All = Array.AsReadOnly(RequireTable(typeof(TEnum)).Members.Cast<TEnum>().ToArray());
        }
    }
}
