using System;
using ReinforcementLearning.DataBuilders;
using ReinforcementLearning.GameConfigurations;
using ReinforcementLearning.MemoryTypes;

namespace ReinforcementLearning.PlaySessions {
    internal class TestingPlaySession<TGameConfiguration, TData> : BasePlaySession<TGameConfiguration, TData>
        where TGameConfiguration : IGameConfiguration {
        public TestingPlaySession(TGameConfiguration game, Trainer<TGameConfiguration, TData> trainer, ReplayMemory<TData> memory, DataBuilder<TGameConfiguration, TData> dataBuilder) 
            : base(game, trainer, memory, dataBuilder) {
        }

        protected override void OnEpisodeStart(int episodeIndex) {
            base.OnEpisodeStart(episodeIndex);

            Console.WriteLine($"Stage [{episodeIndex + 1}]/[{Game.Episodes}]");
        }

        protected override void OnEpisodeDone(float episodeReward) {
            base.OnEpisodeDone(episodeReward);

            Memory.Clear();
        }

        protected override void OnCompleted() {
            base.OnCompleted();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n\nPlay session completed\n\n");
            Console.ResetColor();
        }
    }
}