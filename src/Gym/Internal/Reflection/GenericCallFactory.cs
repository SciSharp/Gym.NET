using System;
using System.Collections.Generic;
using System.Reflection;
using JetBrains.Annotations;

namespace Gym.Reflection {
    /// <summary>
    ///     A factory that creates a <see cref="MethodInfo"/> with a specific generic type and caches it.
    /// </summary>
    public class GenericCallFactory {
        private readonly MethodInfo _nonGeneric;
        private Dictionary<Type, MethodInfo> _map;

        /// <summary>Initializes a new instance of the <see cref="T:System.Object" /> class.</summary>
        public GenericCallFactory([NotNull] MethodInfo nonGeneric) {
            _nonGeneric = nonGeneric ?? throw new ArgumentNullException(nameof(nonGeneric));
            _map = new Dictionary<Type, MethodInfo>();
        }

        private MethodInfo Get(Type type) {
            if (!_map.TryGetValue(type, out var mi)) {
                _map.Add(type, mi = _nonGeneric.MakeGenericMethod(type));
            }

            return mi;
        }

        /// <param name="type">The T type</param>
        /// <param name="parameters">The parameters to pass to the function</param>
        /// <returns></returns>
        public object Invoke(Type type, params object[] parameters) {
            return Get(type).Invoke(null, parameters);
        }

        /// <param name="type">The T type</param>
        /// <param name="instance">The instance to invoke the method in</param>
        /// <param name="parameters">The parameters to pass to the function</param>
        /// <returns></returns>
        public object InvokeInstance(Type type, object instance, params object[] parameters) {
            return Get(type).Invoke(instance, parameters);
        }
    }
}