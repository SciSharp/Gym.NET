using System;
using System.IO;
using System.Linq;
using ReinforcementLearning.DataBuilders;
using ReinforcementLearning.GameConfigurations;
using ReinforcementLearning.MemoryTypes;
using ReinforcementLearning.PlaySessions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning.GameEngines
{
    public abstract class GameEngine<TGameConfiguration, TData>
        where TGameConfiguration : IGameConfiguration
    {
        protected Trainer<TData> Trainer;
        protected ReplayMemory<TData> Memory;
        protected DataBuilder<TGameConfiguration, TData> DataBuilder;

        public void Play(TGameConfiguration game)
        {
            Trainer = BuildTrainer(game);
            Memory = BuildMemory(game);
            DataBuilder = BuildDataBuilder(game);

            Console.WriteLine("Press [L] to load last saved model, any other key to skip");
            var pressed = Console.ReadKey().KeyChar;
            if (pressed == 'l')
            {
                LoadModelToTrainer(Trainer);
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
                        new TestingPlaySession<TGameConfiguration, TData>(game, Trainer, Memory, DataBuilder).Play();
                        break;
                    case '2':
                        new TrainingPlaySession<TGameConfiguration, TData>(game, Trainer, Memory, DataBuilder).Play();
                        break;
                    default:
                        Console.WriteLine($"Invalid selection {pressed}");
                        break;
                }
            }
        }

        protected abstract DataBuilder<TGameConfiguration, TData> BuildDataBuilder(TGameConfiguration game);
        internal abstract ReplayMemory<TData> BuildMemory(TGameConfiguration game);
        internal abstract Trainer<TData> BuildTrainer(IGameConfiguration game);

        private static void LoadModelToTrainer(Trainer<TData> trainer)
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
