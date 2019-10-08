using System;
using System.IO;
using System.Linq;
using ReinforcementLearning.GameConfigurations;
using ReinforcementLearning.PlaySessions;

namespace ReinforcementLearning
{
    public class GameEngine
    {
        private Trainer _trainer;

        public void Play<TGameConfiguration>(TGameConfiguration game)
            where TGameConfiguration : IGameConfiguration
        {
            var env = game.EnvIstance;
            _trainer = new Trainer(game, env.ActionSpace.Shape.Size);

            Console.WriteLine("Press [L] to load last saved model, any other key to skip");
            var pressed = Console.ReadKey().KeyChar;
            if (pressed == 'l')
            {
                LoadModelToTrainer(_trainer);
            }
            Console.Clear();

            while (true)
            {
                Console.WriteLine("Press [1] to play the game with the current model");
                Console.WriteLine("Press [2] to train");
                pressed = Console.ReadKey().KeyChar;

                switch (pressed)
                {
                    case '1':
                        new TestingPlaySession<TGameConfiguration>(game, _trainer).Play();
                        break;
                    case '2':
                        new TrainingPlaySession<TGameConfiguration>(game, _trainer).Play();
                        break;
                    default:
                        Console.WriteLine($"Invalid selection {pressed}");
                        break;
                }
            }
        }

        private static void LoadModelToTrainer(Trainer trainer)
        {
            var choosenFile = Directory.GetFiles("./")
                .Select(x => new FileInfo(x))
                .Where(x => x.Extension == ".modl")
                .OrderByDescending(x => x.LastWriteTime)
                .FirstOrDefault();

            if (choosenFile == null)
            {
                Console.WriteLine($"No model found in dir {Directory.GetCurrentDirectory()}");
                return;
            }

            Console.WriteLine($"Loading model {choosenFile.FullName}");
            trainer.Load(choosenFile.OpenRead());
        }
    }
}
