using System;
using System.Threading;
using ReinforcementLearning.GameConfigurations;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning.PlaySessions
{
    internal class TrainingPlaySession<TGameConfiguration> : BasePlaySession<TGameConfiguration>
        where TGameConfiguration : IGameConfiguration
    {
        private float _epsilon;
        private readonly CancellationTokenSource _ct;

        internal TrainingPlaySession(TGameConfiguration game, Trainer trainer) : base(game, trainer)
        {
            _ct = new CancellationTokenSource();
            Trainer.StartAsyncTraining(Memory, _ct.Token);
        }

        protected override void OnEpisodeStart(int episodeIndex)
        {
            base.OnEpisodeStart(episodeIndex);

            _epsilon = Game.StartingEpsilon * (Game.Episodes - episodeIndex) / Game.Episodes;
            Console.WriteLine($"Stage [{episodeIndex + 1}]/[{Game.Episodes}], with exploration rate {_epsilon}");
        }

        protected override void OnEpisodeDone(float episodeReward)
        {
            base.OnEpisodeDone(episodeReward);

            Memory.EndEpisode(episodeReward);
        }

        protected override void OnCompleted()
        {
            base.OnCompleted();

            _ct.Cancel();

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\n\nTraining completed\n\n");
            Console.ResetColor();
        }

        protected override int ComposeAction(Image<Rgba32>[] current)
        {
            if (Random.NextDouble() <= _epsilon)
            {
                return Game.EnvIstance.ActionSpace.Sample();
            }

            return base.ComposeAction(current);
        }
    }
}