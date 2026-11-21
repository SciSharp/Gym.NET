#nullable enable
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Gym.Envs;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Keeps the arrays a Gym.NET object stores past the caller's NumSharp scope (DR-96): a caller that steps an env inside
    ///     an <c>[NDScoped]</c> method (or <see cref="NDScope.Open"/>) has every array constructed during the call disposed
    ///     when the scope closes, except what it returns, so an array the env, wrapper or vector env keeps for a later call
    ///     must be detached from that scope where it is stored, NumSharp's rule for an array cached into a long-lived field.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Call <see cref="Keep{T}"/> on the value at the line that stores it (a field, a list, a queue). Detaching an array no
    ///     scope tracks, one built before the scope opened or with no scope open at all, does nothing and costs one
    ///     thread-static read, so the call is safe on every path.
    ///     </para>
    ///     <para>
    ///     Footgun: a detached array is the keeper's to release; nothing disposes it but the keeper (or the finalizer). Keep
    ///     only what the object stores, never what it returns: a returned array the caller's scope cannot reclaim lingers until
    ///     the finalizer. And detach on the creating thread: an array built on another thread was never tracked by this one's
    ///     scope.
    ///     </para>
    /// </remarks>
    internal static class ScopeSafety {
        /// <summary>
        ///     Detaches every array <paramref name="value"/> holds from the scope that tracks it: the value itself when it is
        ///     an <see cref="NDArray"/>, a render result's frames, and the arrays inside tuples (any <see cref="ITuple"/>, the
        ///     generated records and <see cref="Gym.Spaces.GraphInstance"/> included), lists (Python's tuples are
        ///     <see langword="object"/>?[]) and dicts, recursively.
        /// </summary>
        /// <typeparam name="T">The value's type: an observation, an action, a frame.</typeparam>
        /// <param name="value">The value being stored; <see langword="null"/> is allowed and does nothing.</param>
        /// <returns><paramref name="value"/> itself, so the store reads <c>Field = ScopeSafety.Keep(x)</c>.</returns>
        internal static T Keep<T>(T value) {
            Visit(value, NDScope.Detach, depth: 0);
            return value;
        }

        /// <summary>
        ///     Hands arrays an object kept back to the caller: the caller's innermost open scope adopts every array
        ///     <paramref name="value"/> holds (<see cref="NDScope.Attach"/>), so it is released when that scope closes, as an
        ///     array built inside the scope would be. With no scope open it does nothing, and the caller owns the arrays as
        ///     before.
        /// </summary>
        /// <typeparam name="T">The value's type.</typeparam>
        /// <param name="value">The value being returned, which the object no longer keeps; <see langword="null"/> does nothing.</param>
        /// <returns><paramref name="value"/> itself, so the return reads <c>return ScopeSafety.HandOver(frames)</c>.</returns>
        /// <remarks>Footgun: hand over only what the object stops keeping; an array still kept would be disposed under it when the caller's scope closes.</remarks>
        internal static T HandOver<T>(T value) {
            Visit(value, NDScope.Attach, depth: 0);
            return value;
        }

        /// <summary>The recursion behind <see cref="Keep{T}"/> and <see cref="HandOver{T}"/>: applies an action to every array a value holds.</summary>
        /// <param name="value">The value.</param>
        /// <param name="action">What to do with each array: detach it, or attach it to the current scope.</param>
        /// <param name="depth">How deep in nested containers; a guard against a container holding itself.</param>
        private static void Visit(object? value, System.Action<NDArray> action, int depth) {
            // Observations nest a few levels (a dict of tuples of arrays); 32 is far past any space Gymnasium defines.
            if (value is null || depth > 32)
                return;
            switch (value) {
                case NDArray array:
                    action(array);
                    return;
                case RenderResult render:
                    VisitRender(render, action, depth);
                    return;
                case string:
                    return;
                case IReadOnlyDictionary<string, object?> dict:
                    foreach (object? item in dict.Values)
                        Visit(item, action, depth + 1);
                    return;
                case ITuple tuple:
                    for (int i = 0; i < tuple.Length; i++)
                        Visit(tuple[i], action, depth + 1);
                    return;
                case IEnumerable<object?> items:
                    foreach (object? item in items)
                        Visit(item, action, depth + 1);
                    return;
            }
        }

        /// <summary>Applies an action to a render result's frames: one array, a pair, or each frame of a <c>*_list</c> collection.</summary>
        /// <param name="render">The result.</param>
        /// <param name="action">What to do with each array.</param>
        /// <param name="depth">The nesting depth so far.</param>
        private static void VisitRender(RenderResult render, System.Action<NDArray> action, int depth) {
            switch (render) {
                case RenderResult.Rgb rgb:
                    action(rgb.Frame);
                    return;
                case RenderResult.Depth depthFrame:
                    action(depthFrame.Frame);
                    return;
                case RenderResult.RgbDepth pair:
                    action(pair.RgbFrame);
                    action(pair.DepthFrame);
                    return;
                case RenderResult.Frames frames:
                    foreach (RenderResult item in frames.Items)
                        Visit(item, action, depth + 1);
                    return;
            }
        }
    }
}
