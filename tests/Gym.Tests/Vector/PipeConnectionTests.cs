#nullable enable
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Vector {
    /// <summary>
    ///     <see cref="PipeConnection"/> (DR-38), the counterpart of Python's <c>multiprocessing</c> pipe a custom
    ///     <see cref="AsyncVectorWorker"/> talks through: values in order both ways, by reference; <c>poll</c> with and without
    ///     a wait; an end of stream after the queued values once the other end is closed; and Python's refusals.
    /// </summary>
    [TestClass]
    public class PipeConnectionTests {
        /// <summary>Each end receives what the other sent, oldest first, the very objects sent.</summary>
        [TestMethod]
        public void SendAndRecv_KeepTheOrder_BothWays() {
            (PipeConnection parent, PipeConnection child) = PipeConnection.Pipe();
            var payload = new object();
            parent.Send(("step", (object?)1L));
            parent.Send(("close", (object?)null));
            child.Send(payload);

            Assert.AreEqual(("step", (object?)1L), child.Recv());
            Assert.AreEqual(("close", (object?)null), child.Recv());
            Assert.AreSame(payload, parent.Recv(), "Values cross by reference, not pickled.");
        }

        /// <summary>
        ///     <c>poll(0)</c> answers at once, <c>poll(timeout)</c> waits for a value sent from another thread, and a receive
        ///     waits for its value too.
        /// </summary>
        /// <returns>The test.</returns>
        [TestMethod]
        public async Task PollAndRecv_WaitForTheOtherEnd() {
            (PipeConnection parent, PipeConnection child) = PipeConnection.Pipe();
            Assert.IsFalse(parent.Poll(), "Nothing was sent yet.");
            Assert.IsFalse(parent.Poll(0.02), "Nothing arrives within the wait.");

            Task sender = Task.Run(() => {
                Thread.Sleep(50);
                child.Send("late");
            });
            Assert.IsTrue(parent.Poll(10), "A value sent later arrives within a long wait.");
            Assert.AreEqual("late", parent.Recv());
            await sender;

            Task<object?> receiver = Task.Run(() => child.Recv());
            parent.Send("answer");
            Assert.AreEqual("answer", await receiver);
        }

        /// <summary>
        ///     Once every handle on one end is closed, the other end still receives what was queued, then the end of the stream
        ///     (Python's <c>EOFError</c>), which <c>poll</c> reports as readable; sending to it is a broken pipe.
        /// </summary>
        [TestMethod]
        public void ClosedEnd_DrainsThenEndsTheStream() {
            (PipeConnection parent, PipeConnection child) = PipeConnection.Pipe();
            child.Send("last words");
            child.Close();
            Assert.IsTrue(child.Closed);
            Assert.IsFalse(parent.Closed, "Closing one end leaves the other handle open.");

            Assert.IsTrue(parent.Poll());
            Assert.AreEqual("last words", parent.Recv());
            Assert.IsTrue(parent.Poll(), "The end of the stream is readable.");
            Assert.ThrowsExactly<EndOfStreamException>(() => parent.Recv());
            StringAssert.Contains(Assert.ThrowsExactly<OSError>(() => parent.Send("anyone?")).Message, "Broken pipe");
        }

        /// <summary>A closed handle refuses everything with Python's <c>handle is closed</c>, and closing it twice is harmless.</summary>
        [TestMethod]
        public void ClosedHandle_RefusesWithPythonsMessage() {
            (PipeConnection parent, _) = PipeConnection.Pipe();
            parent.Close();
            parent.Dispose();
            Assert.AreEqual("handle is closed", Assert.ThrowsExactly<OSError>(() => parent.Send(1)).Message);
            Assert.AreEqual("handle is closed", Assert.ThrowsExactly<OSError>(() => parent.Recv()).Message);
            Assert.AreEqual("handle is closed", Assert.ThrowsExactly<OSError>(() => parent.Poll()).Message);
        }
    }
}
