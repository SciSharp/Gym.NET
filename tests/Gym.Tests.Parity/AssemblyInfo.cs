using Microsoft.VisualStudio.TestTools.UnitTesting;

// The whole assembly shares one embedded CPython (it cannot be re-initialized in a process, and Gymnasium's modules keep
// state across tests), one GIL, and the process-wide LiveExports/LiveImports counters the leak gate reads. Tests running
// side by side would serialize on the GIL anyway, and would make each other's leak gates flaky, so they run one at a time.
[assembly: DoNotParallelize]
