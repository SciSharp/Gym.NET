using System;
using ReinforcementLearning.GameConfigurations;

namespace ReinforcementLearning.PlaySessions
{
    internal class TestingPlaySession<TGameConfiguration> : BasePlaySession<TGameConfiguration>
        where TGameConfiguration : IGameConfiguration
    {
        public TestingPlaySession(TGameConfiguration game, Trainer trainer) : base(game, trainer)
        { }

        protected override void OnEpisodeStart(int episodeIndex)
        {
            base.OnEpisodeStart(episodeIndex);

            Console.WriteLine($"Stage [{episodeIndex + 1}]/[{Game.Episodes}]");
        }

        protected override void OnEpisodeDone(float episodeReward)
        {
            base.OnEpisodeDone(episodeReward);

            Memory.Clear();
        }

        protected override void OnCompleted()
        {
            base.OnCompleted();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n\nPlay session completed\n\n");
            Console.ResetColor();
        }
    }
}