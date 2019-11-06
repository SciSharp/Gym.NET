using System;
using System.Threading;
using ReinforcementLearning.DataBuilders;
using ReinforcementLearning.GameConfigurations;
using ReinforcementLearning.MemoryTypes;

namespace ReinforcementLearning.PlaySessions {
    internal class TrainingPlaySession<TGameConfiguration, TData> : BasePlaySession<TGameConfiguration, TData>
        where TGameConfiguration : IGameConfiguration {
        private float _epsilon;
        private readonly CancellationTokenSource _ct;

        internal TrainingPlaySession(
            TGameConfiguration game,
            Trainer<TGameConfiguration, TData> trainer,
            ReplayMemory<TData> memory,
            DataBuilder<TGameConfiguration, TData> dataBuilder)
            : base(game, trainer, memory, dataBuilder) {
            _ct = new CancellationTokenSource();
            Trainer.StartAsyncTraining(Memory, _ct.Token);
        }

        protected override void OnEpisodeStart(int episodeIndex) {
            base.OnEpisodeStart(episodeIndex);

            _epsilon = Game.StartingEpsilon * (Game.Episodes - episodeIndex) / Game.Episodes;
            Console.WriteLine($"Stage [{episodeIndex + 1}]/[{Game.Episodes}], with exploration rate {_epsilon}");
        }

        protected override void OnEpisodeDone(float episodeReward) {
            base.OnEpisodeDone(episodeReward);

            Memory.EndEpisode(episodeReward);
        }

        protected override void OnCompleted() {
            base.OnCompleted();

            _ct.Cancel();

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\n\nTraining completed\n\n");
            Console.ResetColor();
        }

        protected override int ComposeAction(TData[] currentData) {
            if (Random.NextDouble() <= _epsilon) {
                return Game.EnvInstance.ActionSpace.Sample();
            }

            return base.ComposeAction(currentData);
        }
    }
}