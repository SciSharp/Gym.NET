using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using ThreadState = System.Threading.ThreadState;

namespace Gym.Threading {
    /// <summary>
    ///     A thread that can be run on any specific cpu core.
    /// </summary>
    public class DistributedThread {
        #region DLL Imports

        [DllImport("kernel32.dll")]
        private static extern int GetCurrentThreadId();

        [DllImport("kernel32.dll")]
        private static extern int GetCurrentProcessorNumber();

        #endregion

        private readonly ParameterizedThreadStart parameterizedThreadStart;

        private readonly ThreadStart threadStart;

        private DistributedThread() {
            ManagedThread = new Thread(DistributedThreadStart);
        }

        public DistributedThread(ThreadStart threadStart, int threadMaxStackSize)
            : this() {
            this.threadStart = threadStart;
        }

        public DistributedThread(ParameterizedThreadStart threadStart)
            : this() {
            parameterizedThreadStart = threadStart;
        }

        private int _processorAffinity;

        public int ProcessorAffinity {
            get => _processorAffinity;
            set {
                _validateProcessorAffinity(value);
                _processorAffinity = value;
            }
        }

        private void _validateProcessorAffinity(int affinity) {
            var count = Environment.ProcessorCount;
            var n = 1;
            for (int i = 0; i < count; i++)
                n <<= 1;
            n--;
            if (affinity > n)
                throw new ArgumentException($"Affinity can't be larger than {n} (decimal), there are {count} cpus in this machine.");
        }

        public Thread ManagedThread { get; }

        private ProcessThread CurrentThread {
            get {
                var id = GetCurrentThreadId();
                return
                    (from ProcessThread th in Process.GetCurrentProcess().Threads
                        where th.Id == id
                        select th).Single();
            }
        }

        public void Start() {
            if (threadStart == null)
                throw new InvalidOperationException();

            ManagedThread.Start(null);
        }

        public void Start(object parameter) {
            if (parameterizedThreadStart == null)
                throw new InvalidOperationException();

            ManagedThread.Start(parameter);
        }

        private void DistributedThreadStart(object parameter) {
            try {
                // fix to OS thread
                Thread.BeginThreadAffinity();

                // set affinity
                if (ProcessorAffinity != 0)
                    CurrentThread.ProcessorAffinity = new IntPtr(ProcessorAffinity);

                // call real thread
                if (threadStart != null)
                    threadStart();
                else if (parameterizedThreadStart != null)
                    parameterizedThreadStart(parameter);
                else
                    throw new InvalidOperationException();
            } finally {
                // reset affinity
                CurrentThread.ProcessorAffinity = new IntPtr(0xFFFF);
                Thread.EndThreadAffinity();
            }
        }

        /// <summary>Applies a captured <see cref="T:System.Threading.CompressedStack" /> to the current thread.</summary>
        /// <param name="stack">The <see cref="T:System.Threading.CompressedStack" /> object to be applied to the current thread.</param>
        /// <exception cref="T:System.InvalidOperationException">In all cases.</exception>
        public void SetCompressedStack(CompressedStack stack) {
            ManagedThread.SetCompressedStack(stack);
        }

        /// <summary>Returns a <see cref="T:System.Threading.CompressedStack" /> object that can be used to capture the stack for the current thread.</summary>
        /// <returns>None. </returns>
        /// <exception cref="T:System.InvalidOperationException">In all cases.</exception>
        public CompressedStack GetCompressedStack() {
            return ManagedThread.GetCompressedStack();
        }

        /// <summary>Raises a <see cref="T:System.Threading.ThreadAbortException" /> in the thread on which it is invoked, to begin the process of terminating the thread while also providing exception information about the thread termination. Calling this method usually terminates the thread.</summary>
        /// <param name="stateInfo">An object that contains application-specific information, such as state, which can be used by the thread being aborted. </param>
        /// <exception cref="T:System.Security.SecurityException">The caller does not have the required permission. </exception>
        /// <exception cref="T:System.Threading.ThreadStateException">The thread that is being aborted is currently suspended.</exception>
        public void Abort(object stateInfo) {
            ManagedThread.Abort(stateInfo);
        }

        /// <summary>Raises a <see cref="T:System.Threading.ThreadAbortException" /> in the thread on which it is invoked, to begin the process of terminating the thread. Calling this method usually terminates the thread.</summary>
        /// <exception cref="T:System.Security.SecurityException">The caller does not have the required permission. </exception>
        /// <exception cref="T:System.Threading.ThreadStateException">The thread that is being aborted is currently suspended.</exception>
        public void Abort() {
            ManagedThread.Abort();
        }

        /// <summary>Either suspends the thread, or if the thread is already suspended, has no effect.</summary>
        /// <exception cref="T:System.Threading.ThreadStateException">The thread has not been started or is dead. </exception>
        /// <exception cref="T:System.Security.SecurityException">The caller does not have the appropriate <see cref="T:System.Security.Permissions.SecurityPermission" />. </exception>
        public void Suspend() {
            ManagedThread.Suspend();
        }

        /// <summary>Resumes a thread that has been suspended.</summary>
        /// <exception cref="T:System.Threading.ThreadStateException">The thread has not been started, is dead, or is not in the suspended state. </exception>
        /// <exception cref="T:System.Security.SecurityException">The caller does not have the appropriate <see cref="T:System.Security.Permissions.SecurityPermission" />. </exception>
        public void Resume() {
            ManagedThread.Resume();
        }

        /// <summary>Interrupts a thread that is in the <see langword="WaitSleepJoin" /> thread state.</summary>
        /// <exception cref="T:System.Security.SecurityException">The caller does not have the appropriate <see cref="T:System.Security.Permissions.SecurityPermission" />. </exception>
        public void Interrupt() {
            ManagedThread.Interrupt();
        }

        /// <summary>Blocks the calling thread until the thread represented by this instance terminates, while continuing to perform standard COM and <see langword="SendMessage" /> pumping.</summary>
        /// <exception cref="T:System.Threading.ThreadStateException">The caller attempted to join a thread that is in the <see cref="F:System.Threading.ThreadState.Unstarted" /> state. </exception>
        /// <exception cref="T:System.Threading.ThreadInterruptedException">The thread is interrupted while waiting. </exception>
        public void Join() {
            ManagedThread.Join();
        }

        /// <summary>Blocks the calling thread until the thread represented by this instance terminates or the specified time elapses, while continuing to perform standard COM and SendMessage pumping.</summary>
        /// <param name="millisecondsTimeout">The number of milliseconds to wait for the thread to terminate. </param>
        /// <returns>
        /// <see langword="true" /> if the thread has terminated; <see langword="false" /> if the thread has not terminated after the amount of time specified by the <paramref name="millisecondsTimeout" /> parameter has elapsed.</returns>
        /// <exception cref="T:System.ArgumentOutOfRangeException">The value of <paramref name="millisecondsTimeout" /> is negative and is not equal to <see cref="F:System.Threading.Timeout.Infinite" /> in milliseconds. </exception>
        /// <exception cref="T:System.Threading.ThreadStateException">The thread has not been started. </exception>
        public bool Join(int millisecondsTimeout) {
            return ManagedThread.Join(millisecondsTimeout);
        }

        /// <summary>Blocks the calling thread until the thread represented by this instance terminates or the specified time elapses, while continuing to perform standard COM and SendMessage pumping.</summary>
        /// <param name="timeout">A <see cref="T:System.TimeSpan" /> set to the amount of time to wait for the thread to terminate. </param>
        /// <returns>
        /// <see langword="true" /> if the thread terminated; <see langword="false" /> if the thread has not terminated after the amount of time specified by the <paramref name="timeout" /> parameter has elapsed.</returns>
        /// <exception cref="T:System.ArgumentOutOfRangeException">The value of <paramref name="timeout" /> is negative and is not equal to <see cref="F:System.Threading.Timeout.Infinite" /> in milliseconds, or is greater than <see cref="F:System.Int32.MaxValue" /> milliseconds. </exception>
        /// <exception cref="T:System.Threading.ThreadStateException">The caller attempted to join a thread that is in the <see cref="F:System.Threading.ThreadState.Unstarted" /> state. </exception>
        public bool Join(TimeSpan timeout) {
            return ManagedThread.Join(timeout);
        }

        /// <summary>Returns an <see cref="T:System.Threading.ApartmentState" /> value indicating the apartment state.</summary>
        /// <returns>One of the <see cref="T:System.Threading.ApartmentState" /> values indicating the apartment state of the managed thread. The default is <see cref="F:System.Threading.ApartmentState.Unknown" />.</returns>
        public ApartmentState GetApartmentState() {
            return ManagedThread.GetApartmentState();
        }

        /// <summary>Sets the apartment state of a thread before it is started.</summary>
        /// <param name="state">The new apartment state.</param>
        /// <returns>
        /// <see langword="true" /> if the apartment state is set; otherwise, <see langword="false" />.</returns>
        /// <exception cref="T:System.ArgumentException">
        /// <paramref name="state" /> is not a valid apartment state.</exception>
        /// <exception cref="T:System.Threading.ThreadStateException">The thread has already been started.</exception>
        public bool TrySetApartmentState(ApartmentState state) {
            return ManagedThread.TrySetApartmentState(state);
        }

        /// <summary>Sets the apartment state of a thread before it is started.</summary>
        /// <param name="state">The new apartment state.</param>
        /// <exception cref="T:System.ArgumentException">
        /// <paramref name="state" /> is not a valid apartment state.</exception>
        /// <exception cref="T:System.Threading.ThreadStateException">The thread has already been started.</exception>
        /// <exception cref="T:System.InvalidOperationException">The apartment state has already been initialized.</exception>
        public void SetApartmentState(ApartmentState state) {
            ManagedThread.SetApartmentState(state);
        }

        /// <summary>Gets a unique identifier for the current managed thread. </summary>
        /// <returns>An integer that represents a unique identifier for this managed thread.</returns>
        public int ManagedThreadId => ManagedThread.ManagedThreadId;

        /// <summary>Gets an <see cref="T:System.Threading.ExecutionContext" /> object that contains information about the various contexts of the current thread. </summary>
        /// <returns>An <see cref="T:System.Threading.ExecutionContext" /> object that consolidates context information for the current thread.</returns>
        public ExecutionContext ExecutionContext => ManagedThread.ExecutionContext;

        /// <summary>Gets or sets a value indicating the scheduling priority of a thread.</summary>
        /// <returns>One of the <see cref="T:System.Threading.ThreadPriority" /> values. The default value is <see cref="F:System.Threading.ThreadPriority.Normal" />.</returns>
        /// <exception cref="T:System.Threading.ThreadStateException">The thread has reached a final state, such as <see cref="F:System.Threading.ThreadState.Aborted" />. </exception>
        /// <exception cref="T:System.ArgumentException">The value specified for a set operation is not a valid <see cref="T:System.Threading.ThreadPriority" /> value. </exception>
        public ThreadPriority Priority {
            get => ManagedThread.Priority;
            set => ManagedThread.Priority = value;
        }

        /// <summary>Gets a value indicating the execution status of the current thread.</summary>
        /// <returns>
        /// <see langword="true" /> if this thread has been started and has not terminated normally or aborted; otherwise, <see langword="false" />.</returns>
        public bool IsAlive => ManagedThread.IsAlive;

        /// <summary>Gets a value indicating whether or not a thread belongs to the managed thread pool.</summary>
        /// <returns>
        /// <see langword="true" /> if this thread belongs to the managed thread pool; otherwise, <see langword="false" />.</returns>
        public bool IsThreadPoolThread => ManagedThread.IsThreadPoolThread;

        /// <summary>Gets or sets a value indicating whether or not a thread is a background thread.</summary>
        /// <returns>
        /// <see langword="true" /> if this thread is or is to become a background thread; otherwise, <see langword="false" />.</returns>
        /// <exception cref="T:System.Threading.ThreadStateException">The thread is dead. </exception>
        public bool IsBackground {
            get => ManagedThread.IsBackground;
            set => ManagedThread.IsBackground = value;
        }

        /// <summary>Gets a value containing the states of the current thread.</summary>
        /// <returns>One of the <see cref="T:System.Threading.ThreadState" /> values indicating the state of the current thread. The initial value is <see langword="Unstarted" />.</returns>
        public ThreadState ThreadState => ManagedThread.ThreadState;

        /// <summary>Gets or sets the apartment state of this thread.</summary>
        /// <returns>One of the <see cref="T:System.Threading.ApartmentState" /> values. The initial value is <see langword="Unknown" />.</returns>
        /// <exception cref="T:System.ArgumentException">An attempt is made to set this property to a state that is not a valid apartment state (a state other than single-threaded apartment (<see langword="STA" />) or multithreaded apartment (<see langword="MTA" />)). </exception>
        public ApartmentState ApartmentState {
            get => ManagedThread.ApartmentState;
            set => ManagedThread.ApartmentState = value;
        }

        /// <summary>Gets or sets the current culture used by the Resource Manager to look up culture-specific resources at run time.</summary>
        /// <returns>An object that represents the current culture.</returns>
        /// <exception cref="T:System.ArgumentNullException">The property is set to <see langword="null" />. </exception>
        /// <exception cref="T:System.ArgumentException">The property is set to a culture name that cannot be used to locate a resource file. Resource filenames must include only letters, numbers, hyphens or underscores.</exception>
        public CultureInfo CurrentUICulture {
            get => ManagedThread.CurrentUICulture;
            set => ManagedThread.CurrentUICulture = value;
        }

        /// <summary>Gets or sets the culture for the current thread.</summary>
        /// <returns>An object that represents the culture for the current thread.</returns>
        /// <exception cref="T:System.ArgumentNullException">
        /// The property is set to <see langword="null" />.</exception>
        public CultureInfo CurrentCulture {
            get => ManagedThread.CurrentCulture;
            set => ManagedThread.CurrentCulture = value;
        }

        /// <summary>Gets or sets the name of the thread.</summary>
        /// <returns>A string containing the name of the thread, or <see langword="null" /> if no name was set.</returns>
        /// <exception cref="T:System.InvalidOperationException">A set operation was requested, but the <see langword="Name" /> property has already been set. </exception>
        public string Name {
            get => ManagedThread.Name;
            set => ManagedThread.Name = value;
        }
    }
}