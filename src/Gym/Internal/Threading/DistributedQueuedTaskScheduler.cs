//--------------------------------------------------------------------------
// 
//  Copyright (c) Microsoft Corporation.  All rights reserved. 
// 
//  File: QueuedTaskScheduler.cs
//
//--------------------------------------------------------------------------

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Gym.Collections;

namespace Gym.Threading {
    public enum ThreadAffinityDistrubution {
        ThreadPerCpu,
        ThreadPerTwoCPUs
    }

    /// <summary>
    ///     Provides a TaskScheduler that provides control over priorities, fairness, and the underlying threads utilized.
    /// </summary>
    [DebuggerTypeProxy(typeof(DistributedQueuedTaskScheduler))]
    [DebuggerDisplay("Id={Id}, Queues={DebugQueueCount}, ScheduledTasks = {DebugTaskCount}")]
    //[DebuggerStepThrough]
    public sealed class DistributedQueuedTaskScheduler : TaskScheduler, IDisposable {
        /// <summary>Whether we're processing tasks on the current thread.</summary>
        private static readonly ThreadLocal<bool> _taskProcessingThread = new ThreadLocal<bool>();

        /// <summary>The collection of tasks to be executed on our custom threads.</summary>
        private readonly BlockingCollection<Task> _blockingTaskQueue;

        /// <summary>
        ///     The maximum allowed concurrency level of this scheduler.  If custom threads are
        ///     used, this represents the number of created threads.
        /// </summary>
        private readonly int _concurrencyLevel;

        /// <summary>Cancellation token used for disposal.</summary>
        private readonly CancellationTokenSource _disposeCancellation;

        private readonly CancellationTokenSource _sharedCancellation;

        /// <summary>The queue of tasks to process when using an underlying target scheduler.</summary>
        private readonly Queue<Task> _nonthreadsafeTaskQueue;

        /// <summary>
        ///     A sorted list of round-robin queue lists.  Tasks with the smallest priority value
        ///     are preferred.  Priority groups are round-robin'd through in order of priority.
        /// </summary>
        private readonly SortedList<int, QueueGroup> _queueGroups = new SortedList<int, QueueGroup>();

        // ***
        // *** For when using a target scheduler
        // ***

        /// <summary>The scheduler onto which actual work is scheduled.</summary>
        private readonly TaskScheduler _targetScheduler;

        // ***
        // *** For when using our own threads
        // ***

        /// <summary>The threads used by the scheduler to process work.</summary>
        private readonly Thread[] _threads;

        /// <summary>The number of Tasks that have been queued or that are running whiel using an underlying scheduler.</summary>
        private int _delegatesQueuedOrRunning;

        /// <summary>Initializes the scheduler.</summary>
        /// <param name="threadOnCpu">how many threads will be created per cpu. recommanded is 1 or 2.</param>
        /// <param name="cpusPerThread">When creating a thread, on how many cpus should it be distributed on? When there are: 4 cpus, <paramref name="threadCount"/> is 4 then each thread will be assinged into two cpus. thread1 to cpu1 and cpu2, thread2 to cpu2 and cpu3, thread 3 to cpu3 and cpu4, thread 4 to cpu4 and cpu1.</param>
        public DistributedQueuedTaskScheduler(int cpusPerThread, int threadOnCpu, int cpus = -1, string threadName = "", ThreadPriority threadPriority = ThreadPriority.Normal, ApartmentState threadApartmentState = ApartmentState.MTA, int threadMaxStackSize = 0, Action threadInit = null, Action threadFinally = null, CancellationToken? cancellationToken = null) {
            if (cpusPerThread <= 0)
                throw new ArgumentOutOfRangeException(nameof(cpusPerThread));
            if (threadOnCpu <= 0)
                throw new ArgumentOutOfRangeException(nameof(threadOnCpu));
            if (cpus < 0)
                cpus = Environment.ProcessorCount;

            var threadCount = threadOnCpu * cpus;

            var threads = new DistributedThread[threadCount];
            for (int i = 0; i < threadCount; i++) {
                threads[i] = new DistributedThread(() => ThreadBasedDispatchLoop(threadInit, threadFinally), threadMaxStackSize) {
                    Priority = threadPriority,
                    IsBackground = true,
                    Name = $"[{(threadName ?? "QWorker ")}] {i}"
                };

                threads[i].SetApartmentState(threadApartmentState);
            }

            var possibleAffinities = new CircularQueue<int>();
            for (int i = 0; i < cpus; i++) {
                //popular
                possibleAffinities.Enqueue(1 << i);
            }

            for (int i = 0; i < threadCount; i++) {
                var thread = threads[i];
                thread.ProcessorAffinity = 0;
                var clone = (CircularQueue<int>) possibleAffinities.Clone();
                possibleAffinities.Dequeue(); //move forward by one
                for (int x = 0; x < cpusPerThread; x++) {
                    thread.ProcessorAffinity |= clone.Dequeue();
                }
            }

            _disposeCancellation = new CancellationTokenSource();
            _sharedCancellation = cancellationToken.HasValue ? CancellationTokenSource.CreateLinkedTokenSource(_disposeCancellation.Token, cancellationToken.Value) : _disposeCancellation;

            _concurrencyLevel = threadCount;

            // Initialize the queue used for storing tasks
            _blockingTaskQueue = new BlockingCollection<Task>();

            _threads = new Thread[threadCount];
            for (var i = 0; i < threads.Length; i++) {
                var dt = threads[i];
                dt.Start();
                _threads[i] = dt.ManagedThread;
            }
        }

        /// <summary>Gets the number of queues currently activated.</summary>
        private int DebugQueueCount {
            get {
                var count = 0;
                foreach (var group in _queueGroups)
                    count += group.Value.Count;
                return count;
            }
        }

        /// <summary>Gets the number of tasks currently scheduled.</summary>
        private int DebugTaskCount {
            get { return (_targetScheduler != null ? _nonthreadsafeTaskQueue : (IEnumerable<Task>) _blockingTaskQueue).Count(t => t != null); }
        }

        /// <summary>Gets the maximum concurrency level to use when processing tasks.</summary>
        public override int MaximumConcurrencyLevel => _concurrencyLevel;

        /// <summary>Initiates shutdown of the scheduler.</summary>
        public void Dispose() {
            _sharedCancellation.SafeCancelAndDispose();
            _disposeCancellation.SafeCancelAndDispose();
        }

        /// <summary>The dispatch loop run by all threads in this scheduler.</summary>
        /// <param name="threadInit">An initialization routine to run when the thread begins.</param>
        /// <param name="threadFinally">A finalization routine to run before the thread ends.</param>
        private void ThreadBasedDispatchLoop(Action threadInit, Action threadFinally) {
            _taskProcessingThread.Value = true;
            if (threadInit != null)
                threadInit();
            try {
                // If the scheduler is disposed, the cancellation token will be set and
                // we'll receive an OperationCanceledException.  That OCE should not crash the process.
                try {
                    // If a thread abort occurs, we'll try to reset it and continue running.
                    while (true)
                        try {
                            // For each task queued to the scheduler, try to execute it.
                            foreach (var task in _blockingTaskQueue.GetConsumingEnumerable(_sharedCancellation.Token)) // If the task is not null, that means it was queued to this scheduler directly.
                                // Run it.
                                if (task != null) {
                                    TryExecuteTask(task);
                                }
                                // If the task is null, that means it's just a placeholder for a task
                                // queued to one of the subschedulers.  Find the next task based on
                                // priority and fairness and run it.
                                else {
                                    // Find the next task based on our ordering rules...
                                    Task targetTask;
                                    DistributedQueuedTaskSchedulerQueue queueForTargetTask;
                                    lock (_queueGroups) {
                                        FindNextTask_NeedsLock(out targetTask, out queueForTargetTask);
                                    }

                                    // ... and if we found one, run it
                                    if (targetTask != null)
                                        queueForTargetTask.ExecuteTask(targetTask);
                                }
                        } catch (ThreadAbortException) {
                            // If we received a thread abort, and that thread abort was due to shutting down
                            // or unloading, let it pass through.  Otherwise, reset the abort so we can
                            // continue processing work items.
                            if (!Environment.HasShutdownStarted && !AppDomain.CurrentDomain.IsFinalizingForUnload())
                                Thread.ResetAbort();
                        }
                } catch (OperationCanceledException) { }
            } finally {
                // Run a cleanup routine if there was one
                if (threadFinally != null)
                    threadFinally();
                _taskProcessingThread.Value = false;
            }
        }

        /// <summary>Find the next task that should be executed, based on priorities and fairness and the like.</summary>
        /// <param name="targetTask">The found task, or null if none was found.</param>
        /// <param name="queueForTargetTask">
        ///     The scheduler associated with the found task.  Due to security checks inside of TPL,
        ///     this scheduler needs to be used to execute that task.
        /// </param>
        private void FindNextTask_NeedsLock(out Task targetTask, out DistributedQueuedTaskSchedulerQueue queueForTargetTask) {
            targetTask = null;
            queueForTargetTask = null;

            // Look through each of our queue groups in sorted order.
            // This ordering is based on the priority of the queues.
            foreach (var queueGroup in _queueGroups) {
                var queues = queueGroup.Value;

                // Within each group, iterate through the queues in a round-robin
                // fashion.  Every time we iterate again and successfully find a task, 
                // we'll start in the next location in the group.
                foreach (var i in queues.CreateSearchOrder()) {
                    queueForTargetTask = queues[i];
                    var items = queueForTargetTask._workItems;
                    if (items.Count > 0) {
                        targetTask = items.Dequeue();
                        if (queueForTargetTask._disposed && items.Count == 0)
                            RemoveQueue_NeedsLock(queueForTargetTask);

                        queues.NextQueueIndex = (queues.NextQueueIndex + 1) % queueGroup.Value.Count;
                        return;
                    }
                }
            }
        }

        /// <summary>Queues a task to the scheduler.</summary>
        /// <param name="task">The task to be queued.</param>
        protected override void QueueTask(Task task) {
            // If we've been disposed, no one should be queueing
            if (_sharedCancellation.IsCancellationRequested)
                throw new ObjectDisposedException(GetType().Name);

            // If the target scheduler is null (meaning we're using our own threads),
            // add the task to the blocking queue
            if (_targetScheduler == null) {
                _blockingTaskQueue.Add(task);
            }
            // Otherwise, add the task to the non-blocking queue,
            // and if there isn't already an executing processing task,
            // start one up
            else {
                // Queue the task and check whether we should launch a processing
                // task (noting it if we do, so that other threads don't result
                // in queueing up too many).
                var launchTask = false;
                lock (_nonthreadsafeTaskQueue) {
                    _nonthreadsafeTaskQueue.Enqueue(task);
                    if (_delegatesQueuedOrRunning < _concurrencyLevel) {
                        ++_delegatesQueuedOrRunning;
                        launchTask = true;
                    }
                }

                // If necessary, start processing asynchronously
                if (launchTask)
                    Task.Factory.StartNew(
                        ProcessPrioritizedAndBatchedTasks,
                        CancellationToken.None,
                        TaskCreationOptions.None,
                        _targetScheduler
                    );
            }
        }

        /// <summary>
        ///     Process tasks one at a time in the best order.
        ///     This should be run in a Task generated by QueueTask.
        ///     It's been separated out into its own method to show up better in Parallel Tasks.
        /// </summary>
        private void ProcessPrioritizedAndBatchedTasks() {
            var continueProcessing = true;
            while (!_sharedCancellation.IsCancellationRequested && continueProcessing)
                try {
                    // Note that we're processing tasks on this thread
                    _taskProcessingThread.Value = true;

                    // Until there are no more tasks to process
                    while (!_sharedCancellation.IsCancellationRequested) {
                        // Try to get the next task.  If there aren't any more, we're done.
                        Task targetTask;
                        lock (_nonthreadsafeTaskQueue) {
                            if (_nonthreadsafeTaskQueue.Count == 0)
                                break;
                            targetTask = _nonthreadsafeTaskQueue.Dequeue();
                        }

                        // If the task is null, it's a placeholder for a task in the round-robin queues.
                        // Find the next one that should be processed.
                        DistributedQueuedTaskSchedulerQueue queueForTargetTask = null;
                        if (targetTask == null)
                            lock (_queueGroups) {
                                FindNextTask_NeedsLock(out targetTask, out queueForTargetTask);
                            }

                        // Now if we finally have a task, run it.  If the task
                        // was associated with one of the round-robin schedulers, we need to use it
                        // as a thunk to execute its task.
                        if (targetTask != null)
                            if (queueForTargetTask != null)
                                queueForTargetTask.ExecuteTask(targetTask);
                            else
                                TryExecuteTask(targetTask);
                    }
                } finally {
                    // Now that we think we're done, verify that there really is
                    // no more work to do.  If there's not, highlight
                    // that we're now less parallel than we were a moment ago.
                    lock (_nonthreadsafeTaskQueue) {
                        if (_nonthreadsafeTaskQueue.Count == 0) {
                            _delegatesQueuedOrRunning--;
                            continueProcessing = false;
                            _taskProcessingThread.Value = false;
                        }
                    }
                }
        }

        /// <summary>Notifies the pool that there's a new item to be executed in one of the round-robin queues.</summary>
        private void NotifyNewWorkItem() {
            QueueTask(null);
        }

        /// <summary>Tries to execute a task synchronously on the current thread.</summary>
        /// <param name="task">The task to execute.</param>
        /// <param name="taskWasPreviouslyQueued">Whether the task was previously queued.</param>
        /// <returns>true if the task was executed; otherwise, false.</returns>
        protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued) {
            // If we're already running tasks on this threads, enable inlining
            return _taskProcessingThread.Value && TryExecuteTask(task);
        }

        /// <summary>Gets the tasks scheduled to this scheduler.</summary>
        /// <returns>An enumerable of all tasks queued to this scheduler.</returns>
        /// <remarks>This does not include the tasks on sub-schedulers.  Those will be retrieved by the debugger separately.</remarks>
        protected override IEnumerable<Task> GetScheduledTasks() {
            // If we're running on our own threads, get the tasks from the blocking queue...
            if (_targetScheduler == null)
                return _blockingTaskQueue.Where(t => t != null).ToList();
            // otherwise get them from the non-blocking queue...
            return _nonthreadsafeTaskQueue.Where(t => t != null).ToList();
        }

        /// <summary>Creates and activates a new scheduling queue for this scheduler.</summary>
        /// <returns>The newly created and activated queue at priority 0.</returns>
        public TaskScheduler ActivateNewQueue() {
            return ActivateNewQueue(0);
        }

        /// <summary>Creates and activates a new scheduling queue for this scheduler.</summary>
        /// <param name="priority">The priority level for the new queue.</param>
        /// <returns>The newly created and activated queue at the specified priority.</returns>
        public TaskScheduler ActivateNewQueue(int priority) {
            // Create the queue
            var createdQueue = new DistributedQueuedTaskSchedulerQueue(priority, this);

            // Add the queue to the appropriate queue group based on priority
            lock (_queueGroups) {
                QueueGroup list;
                if (!_queueGroups.TryGetValue(priority, out list)) {
                    list = new QueueGroup();
                    _queueGroups.Add(priority, list);
                }

                list.Add(createdQueue);
            }

            // Hand the new queue back
            return createdQueue;
        }

        /// <summary>Removes a scheduler from the group.</summary>
        /// <param name="queue">The scheduler to be removed.</param>
        private void RemoveQueue_NeedsLock(DistributedQueuedTaskSchedulerQueue queue) {
            // Find the group that contains the queue and the queue's index within the group
            var queueGroup = _queueGroups[queue._priority];
            var index = queueGroup.IndexOf(queue);

            // We're about to remove the queue, so adjust the index of the next
            // round-robin starting location if it'll be affected by the removal
            if (queueGroup.NextQueueIndex >= index)
                queueGroup.NextQueueIndex--;

            // Remove it
            queueGroup.RemoveAt(index);
        }

        /// <summary>Debug view for the QueuedTaskScheduler.</summary>
        private class QueuedTaskSchedulerDebugView {
            /// <summary>The scheduler.</summary>
            private readonly DistributedQueuedTaskScheduler _scheduler;

            /// <summary>Initializes the debug view.</summary>
            /// <param name="scheduler">The scheduler.</param>
            public QueuedTaskSchedulerDebugView(DistributedQueuedTaskScheduler scheduler) {
                if (scheduler == null)
                    throw new ArgumentNullException("scheduler");
                _scheduler = scheduler;
            }

            /// <summary>Gets all of the Tasks queued to the scheduler directly.</summary>
            public IEnumerable<Task> ScheduledTasks {
                get {
                    var tasks = _scheduler._targetScheduler != null ? _scheduler._nonthreadsafeTaskQueue : (IEnumerable<Task>) _scheduler._blockingTaskQueue;
                    return tasks.Where(t => t != null).ToList();
                }
            }

            /// <summary>Gets the prioritized and fair queues.</summary>
            public IEnumerable<TaskScheduler> Queues {
                get {
                    var queues = new List<TaskScheduler>();
                    foreach (var group in _scheduler._queueGroups)
                        queues.AddRange(group.Value);
                    return queues;
                }
            }
        }

        /// <summary>A group of queues a the same priority level.</summary>
        private class QueueGroup : List<DistributedQueuedTaskSchedulerQueue> {
            /// <summary>The starting index for the next round-robin traversal.</summary>
            public int NextQueueIndex;

            /// <summary>Creates a search order through this group.</summary>
            /// <returns>An enumerable of indices for this group.</returns>
            public IEnumerable<int> CreateSearchOrder() {
                for (var i = NextQueueIndex; i < Count; i++)
                    yield return i;
                for (var i = 0; i < NextQueueIndex; i++)
                    yield return i;
            }
        }

        /// <summary>Provides a scheduling queue associatd with a QueuedTaskScheduler.</summary>
        [DebuggerDisplay("QueuePriority = {_priority}, WaitingTasks = {WaitingTasks}")]
        [DebuggerTypeProxy(typeof(QueuedTaskSchedulerQueueDebugView))]
        private sealed class DistributedQueuedTaskSchedulerQueue : TaskScheduler, IDisposable {
            /// <summary>The scheduler with which this pool is associated.</summary>
            private readonly DistributedQueuedTaskScheduler _pool;

            /// <summary>The work items stored in this queue.</summary>
            internal readonly Queue<Task> _workItems;

            /// <summary>Whether this queue has been disposed.</summary>
            internal bool _disposed;

            /// <summary>Gets the priority for this queue.</summary>
            internal readonly int _priority;

            /// <summary>Initializes the queue.</summary>
            /// <param name="priority">The priority associated with this queue.</param>
            /// <param name="pool">The scheduler with which this queue is associated.</param>
            internal DistributedQueuedTaskSchedulerQueue(int priority, DistributedQueuedTaskScheduler pool) {
                _priority = priority;
                _pool = pool;
                _workItems = new Queue<Task>();
            }

            /// <summary>Gets the number of tasks waiting in this scheduler.</summary>
            internal int WaitingTasks => _workItems.Count;

            /// <summary>Gets the maximum concurrency level to use when processing tasks.</summary>
            public override int MaximumConcurrencyLevel => _pool.MaximumConcurrencyLevel;

            /// <summary>Signals that the queue should be removed from the scheduler as soon as the queue is empty.</summary>
            public void Dispose() {
                if (!_disposed) {
                    lock (_pool._queueGroups) {
                        // We only remove the queue if it's empty.  If it's not empty,
                        // we still mark it as disposed, and the associated QueuedTaskScheduler
                        // will remove the queue when its count hits 0 and its _disposed is true.
                        if (_workItems.Count == 0)
                            _pool.RemoveQueue_NeedsLock(this);
                    }

                    _disposed = true;
                }
            }

            /// <summary>Gets the tasks scheduled to this scheduler.</summary>
            /// <returns>An enumerable of all tasks queued to this scheduler.</returns>
            protected override IEnumerable<Task> GetScheduledTasks() {
                return _workItems.ToList();
            }

            /// <summary>Queues a task to the scheduler.</summary>
            /// <param name="task">The task to be queued.</param>
            protected override void QueueTask(Task task) {
                if (_disposed)
                    throw new ObjectDisposedException(GetType().Name);

                // Queue up the task locally to this queue, and then notify
                // the parent scheduler that there's work available
                lock (_pool._queueGroups) {
                    _workItems.Enqueue(task);
                }

                _pool.NotifyNewWorkItem();
            }

            /// <summary>Tries to execute a task synchronously on the current thread.</summary>
            /// <param name="task">The task to execute.</param>
            /// <param name="taskWasPreviouslyQueued">Whether the task was previously queued.</param>
            /// <returns>true if the task was executed; otherwise, false.</returns>
            protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued) {
                // If we're using our own threads and if this is being called from one of them,
                // or if we're currently processing another task on this thread, try running it inline.
                return _taskProcessingThread.Value && TryExecuteTask(task);
            }

            /// <summary>Runs the specified ask.</summary>
            /// <param name="task">The task to execute.</param>
            internal void ExecuteTask(Task task) {
                TryExecuteTask(task);
            }

            /// <summary>A debug view for the queue.</summary>
            private sealed class QueuedTaskSchedulerQueueDebugView {
                /// <summary>The queue.</summary>
                private readonly DistributedQueuedTaskSchedulerQueue _queue;

                /// <summary>Initializes the debug view.</summary>
                /// <param name="queue">The queue to be debugged.</param>
                public QueuedTaskSchedulerQueueDebugView(DistributedQueuedTaskSchedulerQueue queue) {
                    if (queue == null)
                        throw new ArgumentNullException("queue");
                    _queue = queue;
                }

                /// <summary>Gets the priority of this queue in its associated scheduler.</summary>
                public int Priority => _queue._priority;

                /// <summary>Gets the ID of this scheduler.</summary>
                public int Id => _queue.Id;

                /// <summary>Gets all of the tasks scheduled to this queue.</summary>
                public IEnumerable<Task> ScheduledTasks => _queue.GetScheduledTasks();

                /// <summary>Gets the QueuedTaskScheduler with which this queue is associated.</summary>
                public DistributedQueuedTaskScheduler AssociatedScheduler => _queue._pool;
            }
        }

        public Task Run(Action action) {
            return Task.Factory.StartNew(action, CancellationToken.None, TaskCreationOptions.None, this);
        }

        public Task Run(Action action, CancellationToken token) {
            return Task.Factory.StartNew(action, token, TaskCreationOptions.None, this);
        }

        public Task[] Run(params Action[] actions) {
            var ret = new Task[actions.Length];
            for (var i = 0; i < actions.Length; i++) {
                ret[i] = Task.Factory.StartNew(actions[i], CancellationToken.None, TaskCreationOptions.None, this);
            }

            return ret;
        }

        public Task[] Run(CancellationToken token, params Action[] actions) {
            var ret = new Task[actions.Length];
            for (var i = 0; i < actions.Length; i++) {
                ret[i] = Task.Factory.StartNew(actions[i], token, TaskCreationOptions.None, this);
            }

            return ret;
        }

        public Task Run(Action action, TaskCreationOptions opts) {
            return Task.Factory.StartNew(action, CancellationToken.None, opts, this);
        }

        public Task Run(Action action, CancellationToken token, TaskCreationOptions opts) {
            return Task.Factory.StartNew(action, token, opts, this);
        }

        public Task<T> Run<T>(Func<T> action) {
            return Task.Factory.StartNew(action, CancellationToken.None, TaskCreationOptions.None, this);
        }

        public Task<T> Run<T>(Func<T> action, CancellationToken token) {
            return Task.Factory.StartNew(action, token, TaskCreationOptions.None, this);
        }

        public Task<T> Run<T>(Func<T> action, TaskCreationOptions opts) {
            return Task.Factory.StartNew(action, CancellationToken.None, opts, this);
        }

        public Task<T> Run<T>(Func<T> action, CancellationToken token, TaskCreationOptions opts) {
            return Task.Factory.StartNew(action, token, opts, this);
        }

        public void ForEach<T>(IEnumerable<T> toIterate, Action<T> act) {
            Parallel.ForEach(
                toIterate,
                new ParallelOptions() {TaskScheduler = this, MaxDegreeOfParallelism = _threads.Length, CancellationToken = CancellationToken.None},
                act
            );
        }

        public void ForEach<T>(IEnumerable<T> toIterate, Action<T, ParallelLoopState> act) {
            Parallel.ForEach(
                toIterate,
                new ParallelOptions() {TaskScheduler = this, MaxDegreeOfParallelism = _threads.Length, CancellationToken = CancellationToken.None},
                act
            );
        }

        public void ForEach<T>(IEnumerable<T> toIterate, Action<T, ParallelLoopState, long> act) {
            Parallel.ForEach(
                toIterate,
                new ParallelOptions() {TaskScheduler = this, MaxDegreeOfParallelism = _threads.Length, CancellationToken = CancellationToken.None},
                act
            );
        }

        public void ForEach<T>(IEnumerable<T> toIterate, CancellationToken token, Action<T> act) {
            Parallel.ForEach(
                toIterate,
                new ParallelOptions() {TaskScheduler = this, MaxDegreeOfParallelism = _threads.Length, CancellationToken = token},
                act
            );
        }

        public void ForEach<T>(IEnumerable<T> toIterate, CancellationToken token, Action<T, ParallelLoopState> act) {
            Parallel.ForEach(
                toIterate,
                new ParallelOptions() {TaskScheduler = this, MaxDegreeOfParallelism = _threads.Length, CancellationToken = token},
                act
            );
        }

        public void ForEach<T>(IEnumerable<T> toIterate, CancellationToken token, Action<T, ParallelLoopState, long> act) {
            Parallel.ForEach(
                toIterate,
                new ParallelOptions() {TaskScheduler = this, MaxDegreeOfParallelism = _threads.Length, CancellationToken = token},
                act
            );
        }
    }
}