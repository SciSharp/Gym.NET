#nullable enable
using System;
using System.Threading;
using System.Threading.Tasks;
using Gym.Envs.Registration;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Envs.Registration {
    /// <summary>
    ///     Pins DR-57: <see cref="EnvRegistry.Namespace"/>'s scope belongs to the thread and async flow that opened it, where
    ///     Gymnasium's (and Gym.NET's before) is one process-wide global. Single-threaded behavior is Gymnasium's own test
    ///     (<c>TestRegister.TestNamespace</c>); these cover what only threads and awaits can show.
    /// </summary>
    /// <remarks>
    ///     Each test registers ids no other test uses (prefix <c>NsScope</c>) under string entry points that are never
    ///     resolved, and removes them in <c>finally</c>, so the shared registry is left as it was even when an assertion fails.
    /// </remarks>
    [TestClass]
    public class NamespaceScopeTests {
        /// <summary>The longest any test waits for another thread before it fails instead of hanging the run.</summary>
        private static readonly TimeSpan Patience = TimeSpan.FromSeconds(30);

        /// <summary>
        ///     Two threads, each inside its own scope, register while both scopes are open: thread A opens its scope first,
        ///     thread B opens one after it, then A registers. With one global, A's id would land in B's namespace (the
        ///     namespace opened last); with DR-57 each id lands in its own thread's.
        /// </summary>
        /// <remarks>
        ///     Events force the interleaving, so the test fails on a process-wide namespace every time rather than by chance.
        ///     The test thread itself never opens a scope, and must still see none afterwards.
        /// </remarks>
        /// <exception cref="AssertFailedException">Thrown when a registering thread failed or timed out, with its exception.</exception>
        [TestMethod]
        public void ConcurrentScopes_EachThreadRegistersInItsOwnNamespace() {
            using var aOpened = new ManualResetEventSlim();
            using var bOpened = new ManualResetEventSlim();
            using var aRegistered = new ManualResetEventSlim();
            Exception? failure = null;

            // Each thread records its exception instead of throwing, since an exception on a dedicated thread would end the
            // test process rather than fail this test.
            var threadA = new Thread(() => {
                try {
                    using (EnvRegistry.Namespace("NsScopeA")) {
                        aOpened.Set();
                        if (!bOpened.Wait(Patience))
                            throw new TimeoutException("thread B never opened its scope");
                        EnvRegistry.Register("NsScopeGrid-v0", "no-entry-point");
                        aRegistered.Set();
                    }
                } catch (Exception e) {
                    failure ??= e;
                    aRegistered.Set();
                }
            });
            var threadB = new Thread(() => {
                try {
                    if (!aOpened.Wait(Patience))
                        throw new TimeoutException("thread A never opened its scope");
                    using (EnvRegistry.Namespace("NsScopeB")) {
                        bOpened.Set();
                        if (!aRegistered.Wait(Patience))
                            throw new TimeoutException("thread A never registered");
                        EnvRegistry.Register("NsScopeMaze-v0", "no-entry-point");
                    }
                } catch (Exception e) {
                    failure ??= e;
                    bOpened.Set();
                }
            });

            try {
                threadA.Start();
                threadB.Start();
                Assert.IsTrue(threadA.Join(Patience), "thread A did not finish");
                Assert.IsTrue(threadB.Join(Patience), "thread B did not finish");
                if (failure is not null)
                    throw new AssertFailedException($"a registering thread failed: {failure}");

                Assert.IsTrue(EnvRegistry.Registry.ContainsKey("NsScopeA/NsScopeGrid-v0"), "thread A's id is not in A's namespace");
                Assert.IsTrue(EnvRegistry.Registry.ContainsKey("NsScopeB/NsScopeMaze-v0"), "thread B's id is not in B's namespace");
                Assert.IsFalse(EnvRegistry.Registry.ContainsKey("NsScopeB/NsScopeGrid-v0"), "thread A registered in thread B's namespace");
                Assert.IsNull(EnvRegistry.CurrentNamespace, "the test thread, which opened no scope, sees a namespace");
            } finally {
                EnvRegistry.Registry.Remove("NsScopeA/NsScopeGrid-v0");
                EnvRegistry.Registry.Remove("NsScopeB/NsScopeMaze-v0");
                EnvRegistry.Registry.Remove("NsScopeB/NsScopeGrid-v0");
            }
        }

        /// <summary>
        ///     A thread started inside a scope inherits the namespace as it was when the thread started, and the scope's end
        ///     restores the opener's namespace without touching the child's.
        /// </summary>
        [TestMethod]
        public void ThreadStartedInsideScope_InheritsItsNamespace() {
            string? seenByChild = null;
            using (EnvRegistry.Namespace("NsScopeParent")) {
                var child = new Thread(() => seenByChild = EnvRegistry.CurrentNamespace);
                child.Start();
                Assert.IsTrue(child.Join(Patience), "the child thread did not finish");
            }

            Assert.AreEqual("NsScopeParent", seenByChild);
            Assert.IsNull(EnvRegistry.CurrentNamespace);
        }

        /// <summary>
        ///     An async method's scope survives its awaits (the continuation runs in the flow that opened it, on whatever
        ///     thread), and the caller's namespace is untouched before and after.
        /// </summary>
        /// <returns>The test's completion.</returns>
        [TestMethod]
        public async Task AsyncScope_SpansAwaits_AndStaysInItsFlow() {
            try {
                await RegisterAcrossAwaitAsync();

                Assert.IsTrue(EnvRegistry.Registry.ContainsKey("NsScopeAsync/NsScopeAsyncEnv-v0"), "the id registered after an await left the scope's namespace");
                Assert.IsNull(EnvRegistry.CurrentNamespace, "the scope leaked to the caller");
            } finally {
                EnvRegistry.Registry.Remove("NsScopeAsync/NsScopeAsyncEnv-v0");
                EnvRegistry.Registry.Remove("NsScopeAsyncEnv-v0");
            }
        }

        /// <summary>
        ///     The documented footgun: a namespace set inside an async method does not flow back to the caller once the
        ///     method returns, because each async method runs on a copy of its caller's flow.
        /// </summary>
        /// <returns>The test's completion.</returns>
        [TestMethod]
        public async Task NamespaceSetInsideAsyncMethod_DoesNotFlowBackToTheCaller() {
            await SetNamespaceInsideAsync();

            Assert.IsNull(EnvRegistry.CurrentNamespace);
        }

        /// <summary>Opens a scope, yields to the scheduler, then registers inside the scope.</summary>
        /// <returns>The registration's completion.</returns>
        private static async Task RegisterAcrossAwaitAsync() {
            using (EnvRegistry.Namespace("NsScopeAsync")) {
                // Task.Yield forces a real continuation, so the registration runs after the method has returned to its
                // caller once, which is where a flow-local value could be lost.
                await Task.Yield();
                EnvRegistry.Register("NsScopeAsyncEnv-v0", "no-entry-point");
            }
        }

        /// <summary>Sets the namespace directly, with no scope to restore it, then yields.</summary>
        /// <returns>The method's completion.</returns>
        private static async Task SetNamespaceInsideAsync() {
            EnvRegistry.CurrentNamespace = "NsScopeLeak";
            await Task.Yield();
        }
    }
}
