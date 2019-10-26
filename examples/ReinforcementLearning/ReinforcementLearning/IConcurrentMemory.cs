using System.Collections.Concurrent;

namespace ReinforcementLearning
{
    public interface IConcurrentMemory<TData>
    {
        ConcurrentBag<Episode<TData>> Episodes { get; }
    }
}