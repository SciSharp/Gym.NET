using System;
using System.Linq;

namespace ReinforcementLearning {
    public class Episode<TData> {
        public Observation<TData>[] Observations { get; set; }

        public float TotalReward => Observations?.Sum(x => x.Reward) ?? throw new InvalidOperationException($"No {nameof(Observations)} set");
    }
}