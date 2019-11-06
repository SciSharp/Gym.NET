using System.Collections.Concurrent;

namespace ReinforcementLearning.MemoryTypes {
    public interface IConcurrentMemory<TData> {
        ConcurrentBag<Episode<TData>> Episodes { get; }
    }
}