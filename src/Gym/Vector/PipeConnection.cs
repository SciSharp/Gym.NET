#nullable enable
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using NumSharp;

namespace Gym.Vector {
    /// <summary>
    ///     One handle on one end of a two-way pipe (DR-38), the counterpart of Python's
    ///     <c>multiprocessing.connection.Connection</c>: what <see cref="AsyncVectorEnv{TObs, TAct}"/> and a custom
    ///     <see cref="AsyncVectorWorker"/> talk through. <see cref="Send"/> queues a value for the other end, <see cref="Recv"/>
    ///     takes the next value the other end sent, waiting for it, <see cref="Poll"/> waits for one without taking it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Values are passed by reference, not pickled: the receiver gets the sender's object, so a sender must not change
    ///     what it sent (an array, an info) until the other side is done with it, which Python's copy-on-send never needed.
    ///     </para>
    ///     <para>
    ///     An end stays open while any handle on it is open, as an OS pipe stays open while any process holds it: a worker
    ///     thread gets handles of its own (a process inherits its own), so a worker that closes its handle on the parent's
    ///     end, as Gymnasium's worker does first thing, leaves the parent's handle working. Data already sent stays readable
    ///     after its sender's end closes; then <see cref="Recv"/> reports the end of the stream with
    ///     <see cref="EndOfStreamException"/>, Python's <c>EOFError</c>.
    ///     </para>
    ///     <para>Thread-safe: any thread may send, receive or close.</para>
    ///     <para>Reference: <see href="https://github.com/python/cpython/blob/v3.12.0/Lib/multiprocessing/connection.py#L182">Lib/multiprocessing/connection.py</see> (CPython 3.12).</para>
    /// </remarks>
    public sealed class PipeConnection : IDisposable {
        /// <summary>The channel both ends share.</summary>
        private readonly Channel _channel;

        /// <summary>Which end this handle is on, 0 or 1.</summary>
        private readonly int _end;

        /// <summary>Whether this handle was closed; read and written under the channel's lock.</summary>
        private bool _closed;

        /// <summary>A handle on end <paramref name="end"/> of <paramref name="channel"/>, counted as open there.</summary>
        /// <param name="channel">The channel.</param>
        /// <param name="end">The end, 0 or 1.</param>
        private PipeConnection(Channel channel, int end) {
            _channel = channel;
            _end = end;
        }

        /// <summary>
        ///     <c>multiprocessing.Pipe()</c>: a new two-way pipe, one handle on each end; what one end sends, the other
        ///     receives, in order.
        /// </summary>
        /// <returns>The two ends' handles, as Python's <c>(conn1, conn2)</c>.</returns>
        public static (PipeConnection First, PipeConnection Second) Pipe() {
            var channel = new Channel();
            return (new PipeConnection(channel, 0), new PipeConnection(channel, 1));
        }

        /// <summary>Whether this handle was closed (<c>closed</c>); the end it is on may still be open through other handles.</summary>
        public bool Closed {
            get {
                lock (_channel.Gate)
                    return _closed;
            }
        }

        /// <summary><c>send(obj)</c>: queues a value for the other end, after the ones already queued; never waits.</summary>
        /// <param name="value">The value, passed by reference.</param>
        /// <exception cref="OSError">
        ///     Thrown when this handle is closed (<c>handle is closed</c>), or when every handle on the other end is
        ///     (<c>[Errno 32] Broken pipe</c>), as writing to a pipe nobody reads fails.
        /// </exception>
        public void Send(object? value) {
            lock (_channel.Gate) {
                if (_closed)
                    throw new OSError("handle is closed");
                if (_channel.OpenHandles[1 - _end] == 0)
                    throw new OSError("[Errno 32] Broken pipe");
                _channel.Inboxes[1 - _end].Enqueue(value);
                Monitor.PulseAll(_channel.Gate);
            }
        }

        /// <summary><c>recv()</c>: takes the next value the other end sent, waiting until one arrives.</summary>
        /// <returns>The value.</returns>
        /// <exception cref="OSError">Thrown when this handle is closed (<c>handle is closed</c>).</exception>
        /// <exception cref="EndOfStreamException">
        ///     Thrown, as Python's <c>EOFError</c>, when nothing is queued and every handle on the other end is closed: nothing
        ///     will ever arrive.
        /// </exception>
        public object? Recv() {
            lock (_channel.Gate) {
                while (true) {
                    if (_closed)
                        throw new OSError("handle is closed");
                    Queue<object?> inbox = _channel.Inboxes[_end];
                    if (inbox.Count > 0)
                        return inbox.Dequeue();
                    if (_channel.OpenHandles[1 - _end] == 0)
                        throw new EndOfStreamException("Ran out of input: the other end of the pipe is closed.");
                    Monitor.Wait(_channel.Gate);
                }
            }
        }

        /// <summary>
        ///     <c>poll(timeout)</c>: whether a <see cref="Recv"/> would return at once, waiting up to
        ///     <paramref name="timeout"/> seconds for that; an end of stream counts, as it does in Python.
        /// </summary>
        /// <param name="timeout">Seconds to wait: 0 (the default) does not wait, <see langword="null"/> waits as long as it takes.</param>
        /// <returns><see langword="true"/> when a value is queued or the other end is closed.</returns>
        /// <exception cref="OSError">Thrown when this handle is closed.</exception>
        public bool Poll(double? timeout = 0) {
            long deadline = timeout is { } seconds ? Stopwatch.GetTimestamp() + (long)(Math.Max(seconds, 0) * Stopwatch.Frequency) : long.MaxValue;
            lock (_channel.Gate) {
                while (true) {
                    if (_closed)
                        throw new OSError("handle is closed");
                    if (_channel.Inboxes[_end].Count > 0 || _channel.OpenHandles[1 - _end] == 0)
                        return true;
                    if (timeout is null) {
                        Monitor.Wait(_channel.Gate);
                        continue;
                    }

                    long left = deadline - Stopwatch.GetTimestamp();
                    if (left <= 0)
                        return false;
                    // Wakes on every send and close; the loop re-checks, so a wake for the other direction costs nothing.
                    Monitor.Wait(_channel.Gate, TimeSpan.FromSeconds((double)left / Stopwatch.Frequency));
                }
            }
        }

        /// <summary>
        ///     <c>close()</c>: closes this handle; once every handle on its end is closed, the other end's receives report
        ///     the end of the stream after the values already queued. Idempotent.
        /// </summary>
        public void Close() {
            lock (_channel.Gate) {
                if (_closed)
                    return;
                _closed = true;
                _channel.OpenHandles[_end]--;
                // A receiver on the other end may now be at the end of its stream.
                Monitor.PulseAll(_channel.Gate);
            }
        }

        /// <summary>Closes this handle, as <see cref="Close"/>.</summary>
        public void Dispose() => Close();

        /// <summary>
        ///     Another handle on this handle's end, as a child process inherits one: the end stays open until every handle on
        ///     it is closed. <see cref="AsyncVectorEnv{TObs, TAct}"/> gives each worker thread handles of its own this way.
        /// </summary>
        /// <returns>The new handle, open.</returns>
        /// <exception cref="OSError">Thrown when this handle is closed.</exception>
        internal PipeConnection Duplicate() {
            lock (_channel.Gate) {
                if (_closed)
                    throw new OSError("handle is closed");
                _channel.OpenHandles[_end]++;
                return new PipeConnection(_channel, _end);
            }
        }

        /// <summary>The state both ends share: one inbox per end and the count of open handles on each.</summary>
        private sealed class Channel {
            /// <summary>Guards everything here; waited on by receivers and pollers, pulsed by senders and closers.</summary>
            public readonly object Gate = new();

            /// <summary>The values sent to each end, oldest first: <c>Inboxes[e]</c> is what end <c>e</c> receives.</summary>
            public readonly Queue<object?>[] Inboxes = { new(), new() };

            /// <summary>The open handles on each end; an end with none is closed.</summary>
            public readonly int[] OpenHandles = { 1, 1 };
        }
    }
}
