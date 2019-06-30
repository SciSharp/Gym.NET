using System;
using System.Threading.Tasks;

namespace Gym.Threading {
    /// <summary>
    ///     A shim to <see cref="TaskScheduler.Default"/> that altenatively creates a scheduler (<see cref="DistributedQueuedTaskScheduler"/>) distributed between cpus.
    /// </summary>
    public static class DistributedScheduler {
        private static readonly object _lock = new object();
        private static DistributedQueuedTaskScheduler _default;
        private static volatile bool _createdDefault = false;
        private static int CpusPerThread = 1;
        private static int ThreadOnCpu = 2;

#if DEBUG
        private static int CPUs = 8;
#else
        private static int CPUs = -1;
#endif

        /// <param name="threadOnCpu">how many threads will be created per cpu. recommanded is 1 or 2.</param>
        /// <param name="cpusPerThread">When creating a thread, on how many cpus should it be distributed on? When there are: 4 cpus, <paramref name="threadCount"/> is 4 then each thread will be assinged into two cpus. thread1 to cpu1 and cpu2, thread2 to cpu2 and cpu3, thread 3 to cpu3 and cpu4, thread 4 to cpu4 and cpu1.</param>
        public static void Configure(int cpusPerThread, int threadOnCpu, int? cpus = -1) {
            if (_createdDefault)
                throw new InvalidOperationException("Default DistributedQueuedTaskScheduler has been already created.");
            CpusPerThread = cpusPerThread;
            ThreadOnCpu = threadOnCpu;
            if (cpus.HasValue) {
                CPUs = cpus.Value;
            }
        }

        /// <summary>
        ///     Get or create a static task scheduler.
        /// </summary>
        public static DistributedQueuedTaskScheduler Default {
            get {
                if (_createdDefault)
                    return _default;
                lock (_lock) {
                    if (_createdDefault)
                        return _default;


                    var def = _default = new DistributedQueuedTaskScheduler(CpusPerThread, ThreadOnCpu, CPUs, threadName: "DistributedScheduler");

                    _createdDefault = true;
                    return def;
                }

                return _default;
            }
        }
    }
}