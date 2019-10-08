using System;

namespace ReinforcementLearning.PlaySessions
{
    internal class TestingPlaySession : BasePlaySession
    {
        protected override void OnEpisodeDone()
        {
            base.OnEpisodeDone();

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