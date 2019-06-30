using System;
using System.Collections.Concurrent;

namespace Gym.Reflection {
    /// <summary>
    ///     A static-cache to the default constructor of type <typeparamref name="T"/>
    /// </summary>
    /// <typeparam name="T">The type to initialize a constructor for.</typeparam>
    /// <exception cref="DefaultConstructorNotFoundException">When type <typeparamref name="T"/> has no empty constructor or constructor with all default values.</exception>
    public static class DefaultConstructor<T> {
        private static Func<T> _constructor;

        /// <summary>
        ///     Gets default constructor or a construcotr that all of his parameters are optional.
        /// </summary>
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
    ///     A static-cache to the default constructor of type <typeparamref name="T"/>
    /// </summary>
    /// <typeparam name="T">The type to initialize a constructor for.</typeparam>
    public static class DefaultConstructor {
        private static ConcurrentDictionary<Type, Func<object>> _maps = new ConcurrentDictionary<Type, Func<object>>();
        private static Func<object> _constructor;
        private static readonly object _lock = new object();

        /// <summary>
        ///     Gets default constructor or a construcotr that all of his parameters are optional.
        /// </summary>
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

    [Serializable]
    public class DefaultConstructorNotFoundException : Exception {
        public DefaultConstructorNotFoundException() { }
        public DefaultConstructorNotFoundException(string message) : base(message) { }
        public DefaultConstructorNotFoundException(string message, Exception inner) : base(message, inner) { }
    }
}