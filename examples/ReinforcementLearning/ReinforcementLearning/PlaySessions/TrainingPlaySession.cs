using System;
using System.Threading;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning.PlaySessions
{
    internal class TrainingPlaySession : BasePlaySession
    {
        private readonly CancellationTokenSource _ct;

        internal TrainingPlaySession()
        {
            _ct = new CancellationTokenSource();
            Trainer.StartAsyncTraining(Memory, _ct.Token);
        }

        protected override void OnEpisodeDone()
        {
            base.OnEpisodeDone();

            Memory.EndEpisode();
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
            if (Random.NextDouble() <= Epsilon)
            {
                return Game.EnvIstance.ActionSpace.Sample();
            }

            return base.ComposeAction(current);
        }
    }
}