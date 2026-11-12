// Gym.Internal is 0.2's code, [Obsolete] since 1.0 and removed in 2.0 with Gym.Legacy (DR-56): it keeps no nullable
// context (DR-54), rather than 89 annotations for code on its way out.
#nullable disable
using System;
using System.Collections.Concurrent;

namespace Gym.Reflection {
    /// <summary>
    ///     A static-cache to the default constructor of type <typeparamref name="T"/>: a parameterless constructor, or else
    ///     the one whose parameters are all optional (called with their defaults). The legacy deep merge uses it to create
    ///     empty containers.
    /// </summary>
    /// <typeparam name="T">The type to initialize a constructor for.</typeparam>
    /// <remarks>
    ///     Obsolete with the legacy <see cref="Gym.Collections.Dict"/> (DR-56) and removed in 2.0: use
    ///     <see cref="System.Activator.CreateInstance{T}()"/> or a <c>new()</c> constraint. Each call of the delegate goes
    ///     through <see cref="System.Reflection.ConstructorInfo.Invoke(object[])"/>, so it costs a reflection call per object.
    /// </remarks>
    [Obsolete("Part of the legacy Dict's merging: use Activator.CreateInstance<T>() or a new() constraint. Removed in 2.0 with Gym.Legacy.")]
    public static class DefaultConstructor<T> {
        /// <summary>The cached delegate, built on first use; a race may build it twice, harmlessly.</summary>
        private static Func<T> _constructor;

        /// <summary>
        ///     Gets default constructor or a construcotr that all of his parameters are optional.
        /// </summary>
        /// <exception cref="DefaultConstructorNotFoundException">When type <typeparamref name="T"/> has no empty constructor or constructor with all default values.</exception>
        public static Func<T> Activator {
            get {
                if (_constructor != null)
                    return _constructor;

                var constructor = typeof(T).GetDefaultConstructor();
                if (constructor == null) {
                    throw new DefaultConstructorNotFoundException($"For type {typeof(T).FullName}");
                }

                _constructor = constructor.CreateDefaultConstructor<T>();
                return _constructor;
            }
        }
    }

    /// <summary>
    ///     A static-cache to the default constructor of a type given at run time, the untyped twin of
    ///     <see cref="DefaultConstructor{T}"/>: one delegate per type, cached for the process's life.
    /// </summary>
    /// <remarks>
    ///     Obsolete with the legacy <see cref="Gym.Collections.Dict"/> (DR-56) and removed in 2.0: use
    ///     <see cref="System.Activator.CreateInstance(Type)"/>. The cache never forgets a type, so it keeps every type it saw
    ///     loaded.
    /// </remarks>
    [Obsolete("Part of the legacy Dict's merging: use Activator.CreateInstance(type) or a new() constraint. Removed in 2.0 with Gym.Legacy.")]
    public static class DefaultConstructor {
        /// <summary>The delegates built so far, by type.</summary>
        private static ConcurrentDictionary<Type, Func<object>> _maps = new ConcurrentDictionary<Type, Func<object>>();

        /// <summary>Never read or written (compiler warning CS0169): 0.2's leftover, kept as it was.</summary>
        private static Func<object> _constructor;

        /// <summary>Serializes building a type's delegate, so each type's is built once.</summary>
        private static readonly object _lock = new object();

        /// <summary>
        ///     Gets default constructor or a construcotr that all of his parameters are optional.
        /// </summary>
        /// <param name="type">The type to construct.</param>
        /// <returns>A delegate that creates a new <paramref name="type"/> each call, through reflection.</returns>
        /// <exception cref="ArgumentNullException">When <paramref name="type"/> is <see langword="null"/>.</exception>
        /// <exception cref="DefaultConstructorNotFoundException">When <paramref name="type"/> has no empty constructor or constructor with all default values.</exception>
        public static Func<object> Activator(Type type) {
            if (_maps.TryGetValue(type, out var ret))
                return ret;
            lock (_lock) {
                if (_maps.TryGetValue(type, out ret))
                    return ret;

                var constructor = type.GetDefaultConstructor();
                if (constructor == null) {
                    throw new DefaultConstructorNotFoundException($"For type {type.FullName}");
                }

                ret = _maps[type] = constructor.CreateDefaultConstructor();
            }

            return ret;
        }
    }

    /// <summary>
    ///     Thrown by <see cref="DefaultConstructor{T}"/> and <see cref="DefaultConstructor"/> for a type with neither a
    ///     parameterless constructor nor one whose parameters are all optional; the message names the type.
    /// </summary>
    /// <remarks>Obsolete with the legacy <see cref="Gym.Collections.Dict"/> (DR-56) and removed in 2.0.</remarks>
    [Serializable]
    [Obsolete("Part of the legacy Dict's merging: Activator.CreateInstance throws MissingMethodException instead. Removed in 2.0 with Gym.Legacy.")]
    public class DefaultConstructorNotFoundException : Exception {
        /// <summary>An error without a message of its own (.NET's default text).</summary>
        public DefaultConstructorNotFoundException() { }

        /// <summary>An error naming the type that cannot be constructed.</summary>
        /// <param name="message">The message, <c>For type &lt;full name&gt;</c> from the two caches.</param>
        public DefaultConstructorNotFoundException(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="inner">The underlying cause.</param>
        public DefaultConstructorNotFoundException(string message, Exception inner) : base(message, inner) { }
    }
}
