#nullable enable
using System;
using System.Collections.Generic;
using Gym.Envs;

namespace Gym.Utils {
    /// <summary>
    ///     Port of <c>RecordConstructorArgs</c> (<c>gymnasium/utils/record_constructor.py</c> @ 9e04324): a wrapper that
    ///     records its constructor arguments, so its <see cref="Wrapper{TWObs, TWAct, TObs, TAct}.Spec"/> can describe it
    ///     completely and <c>make</c> can re-create it from an <see cref="Gym.Envs.Registration.EnvSpec"/>. A wrapper that
    ///     does not implement it is recorded with <c>kwargs=None</c>.
    /// </summary>
    /// <remarks>
    ///     Python's mix-in holds the arguments itself; a C# interface cannot, so implementers keep them in a property set from
    ///     their constructor with <see cref="RecordConstructorArgs.Record"/>, which does the deep copy.
    /// </remarks>
    public interface IRecordConstructorArgs {
        /// <summary>
        ///     The recorded arguments by parameter name in Python spelling (<c>max_episode_steps</c>), Python's
        ///     <c>_saved_kwargs</c>. Set once per constructor; in a class hierarchy the most derived constructor's record is the
        ///     one kept (see <see cref="RecordConstructorArgs.Record"/>).
        /// </summary>
        [PyName("_saved_kwargs")]
        Info SavedKwargs { get; }
    }

    /// <summary>The recording half of Python's <c>RecordConstructorArgs.__init__</c>, for <see cref="IRecordConstructorArgs"/> implementers.</summary>
    public static class RecordConstructorArgs {
        /// <summary>
        ///     Port of <c>RecordConstructorArgs.__init__(self, *, _disable_deepcopy=False, **kwargs)</c>: the arguments to store
        ///     as <see cref="IRecordConstructorArgs.SavedKwargs"/>, deep-copied unless disabled, so later changes to an argument
        ///     object do not change the record.
        /// </summary>
        /// <param name="kwargs">The arguments by parameter name, in Python spelling and order.</param>
        /// <param name="disableDeepcopy">Store the argument values as they are (Python's <c>_disable_deepcopy</c>).</param>
        /// <returns>The record to assign.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="kwargs"/> is <see langword="null"/>.</exception>
        /// <remarks>
        ///     Python keeps the <em>first</em> call's record ("always call <c>RecordConstructorArgs.__init__</c> before the
        ///     <c>Class.__init__</c>"), so a subclass that records before calling its base's <c>__init__</c> wins. C# runs the
        ///     base constructor first, so the equivalent rule is that every constructor assigns the record and the most derived
        ///     one, running last, overwrites the others: assign unconditionally.
        /// </remarks>
        public static Info Record(IEnumerable<KeyValuePair<string, object?>> kwargs, bool disableDeepcopy = false) {
            if (kwargs is null)
                throw new ArgumentNullException(nameof(kwargs));
            var record = new Info(kwargs);
            return disableDeepcopy ? record : record.DeepCopy();
        }
    }
}
