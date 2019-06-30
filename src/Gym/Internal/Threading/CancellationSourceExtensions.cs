using System;
using System.Threading;
using JetBrains.Annotations;

namespace Gym.Threading {
    public static class CancellationSourceExtensions {
        /// <summary>
        ///     Cancels the <see cref="CancellationTokenSource"/> and then disposes it. Swallows any exception, filters null.
        /// </summary>
        /// <param name="src"></param>
        public static void SafeCancelAndDispose([CanBeNull] this CancellationTokenSource src) {
            if (src != null) {
                try {
                    if (!src.IsCancellationRequested)
                        src.Cancel();
                    src.Dispose();
                } catch {
                    // ignored
                }
            }
        }

        /// <summary>
        ///     Cancels the <see cref="CancellationTokenSource"/>. Swallows any exception, filters null.
        /// </summary>
        /// <param name="src"></param>
        public static void TryCancel([CanBeNull] this CancellationTokenSource src) {
            SafeCancel(src);
        }

        /// <summary>
        ///     Cancels the <see cref="CancellationTokenSource"/> and then disposes it. Swallows any exception, filters null.
        /// </summary>
        /// <param name="src"></param>
        public static void TryCancelAndDispose([CanBeNull] this CancellationTokenSource src) {
            SafeCancelAndDispose(src);
        }

        /// <summary>
        ///     Cancels the <see cref="CancellationTokenSource"/>. Swallows any exception, filters null.
        /// </summary>
        /// <param name="src"></param>
        public static void SafeCancel([CanBeNull] this CancellationTokenSource src) {
            if (src != null) {
                try {
                    if (!src.IsCancellationRequested)
                        src.Cancel();
                } catch {
                    // ignored
                }
            }
        }

        /// <summary>
        ///     Cancels the <see cref="IDisposable"/> and then disposes it. Swallows any exception, filters null.
        /// </summary>
        /// <param name="src"></param>
        public static void TryDispose([CanBeNull] this object obj) {
            if (obj != null && obj is IDisposable src) {
                try {
                    src.Dispose();
                } catch {
                    // ignored
                }
            }
        }

        /// <summary>
        ///     Cancels the <see cref="IDisposable"/> and then disposes it. Swallows any exception, filters null.
        /// </summary>
        /// <param name="src"></param>
        public static void SafeDispose([CanBeNull] this IDisposable obj) {
            if (obj != null)
                try {
                    obj.Dispose();
                } catch {
                    // ignored
                }
        }
    }
}