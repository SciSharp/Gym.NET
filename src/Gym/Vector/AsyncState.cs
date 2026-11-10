#nullable enable
using System.Collections.Generic;

namespace Gym.Vector {
    /// <summary>
    ///     Which call an <see cref="AsyncVectorEnv{TObs, TAct}"/> is waiting on: none (<see cref="Default"/>), or the
    ///     <c>reset</c>, <c>step</c> or <c>call</c> sent by its <c>*_async</c> method, whose results the matching
    ///     <c>*_wait</c> collects. Starting another call or waiting on the wrong one fails with
    ///     <see cref="Exceptions.AlreadyPendingCallError"/> or <see cref="Exceptions.NoAsyncCallError"/>, whose
    ///     <c>Name</c> is the pending call's <see cref="Value"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A class with four instances rather than a C# <see langword="enum"/>, as <see cref="AutoresetMode"/> is, so it
    ///     behaves as Python's <c>Enum</c> where it shows: compared by identity, named <c>WAITING_STEP</c>, valued
    ///     <c>"step"</c> (the word Gymnasium's messages print, <c>Calling `step_async` while waiting for a pending call to
    ///     `step` to complete.</c>), printed <c>&lt;AsyncState.WAITING_STEP: 'step'&gt;</c>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/async_vector_env.py#L48">gymnasium/vector/async_vector_env.py</see>.</para>
    /// </remarks>
    public sealed class AsyncState {
        /// <summary>Gymnasium's <c>AsyncState.DEFAULT</c>: no call pending; any <c>*_async</c> may start one.</summary>
        public static readonly AsyncState Default = new("DEFAULT", "default");

        /// <summary>Gymnasium's <c>AsyncState.WAITING_RESET</c>: <c>reset_async</c> was sent; <c>reset_wait</c> collects it.</summary>
        public static readonly AsyncState WaitingReset = new("WAITING_RESET", "reset");

        /// <summary>Gymnasium's <c>AsyncState.WAITING_STEP</c>: <c>step_async</c> was sent; <c>step_wait</c> collects it.</summary>
        public static readonly AsyncState WaitingStep = new("WAITING_STEP", "step");

        /// <summary>Gymnasium's <c>AsyncState.WAITING_CALL</c>: <c>call_async</c> was sent; <c>call_wait</c> collects it.</summary>
        public static readonly AsyncState WaitingCall = new("WAITING_CALL", "call");

        /// <summary>The members in Gymnasium's declaration order.</summary>
        private static readonly AsyncState[] Members = { Default, WaitingReset, WaitingStep, WaitingCall };

        /// <summary>Creates one of the four members.</summary>
        /// <param name="name">The Python member name.</param>
        /// <param name="value">The Python value.</param>
        private AsyncState(string name, string value) {
            Name = name;
            Value = value;
        }

        /// <summary>Python's <c>.name</c>: <c>DEFAULT</c>, <c>WAITING_RESET</c>, <c>WAITING_STEP</c> or <c>WAITING_CALL</c>.</summary>
        public string Name { get; }

        /// <summary>Python's <c>.value</c>: <c>"default"</c>, <c>"reset"</c>, <c>"step"</c> or <c>"call"</c>, the word the pending-call messages print.</summary>
        public string Value { get; }

        /// <summary>Every member, in Gymnasium's order.</summary>
        public static IReadOnlyList<AsyncState> All => Members;

        /// <summary>Python's <c>str(state)</c>: <c>AsyncState.WAITING_STEP</c>.</summary>
        public string Str => $"AsyncState.{Name}";

        /// <summary>Python's <c>repr</c> of the member: <c>&lt;AsyncState.WAITING_STEP: 'step'&gt;</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => $"<AsyncState.{Name}: '{Value}'>";
    }
}
