using System;
using System.IO;
using System.Linq;
using ReinforcementLearning.DataBuilders;
using ReinforcementLearning.GameConfigurations;
using ReinforcementLearning.MemoryTypes;
using ReinforcementLearning.PlaySessions;

namespace ReinforcementLearning.GameEngines {
    public abstract class GameEngine<TGameConfiguration, TData>
        where TGameConfiguration : IGameConfiguration {
        private readonly TGameConfiguration _game;

        protected GameEngine(TGameConfiguration game) {
            _game = game;
            Memory = BuildMemory(game);
            DataBuilder = BuildDataBuilder(game);
            Trainer = BuildTrainer(game);
        }

        protected Trainer<TGameConfiguration, TData> Trainer;
        protected ReplayMemory<TData> Memory;
        protected DataBuilder<TGameConfiguration, TData> DataBuilder;

        public void Play() {
            Console.WriteLine("Press [L] to load last saved model");
            Console.WriteLine("Press [P] to load pre-trained model");
            Console.WriteLine("Press any other key to skip");
            var pressed = Console.ReadKey().KeyChar;
            if (pressed == 'l') {
                LoadLatestModelToTrainer(Trainer);
            }
            if (pressed == 'p'){
                LoadModelToTrainer(Trainer, new FileInfo("Pre-trained backup-network.modl"));
            }

            Console.Clear();

            while (true) {
                Console.WriteLine("Press [1] to play the game with the current model");
                Console.WriteLine("Press [2] to train");
                pressed = Console.ReadKey().KeyChar;

                switch (pressed) {
                    case '1':
                        new TestingPlaySession<TGameConfiguration, TData>(_game, Trainer, Memory, DataBuilder).Play();
                        break;
                    case '2':
                        new TrainingPlaySession<TGameConfiguration, TData>(_game, Trainer, Memory, DataBuilder).Play();
                        break;
                    default:
                        Console.WriteLine($"Invalid selection {pressed}");
                        break;
                }
            }
        }

        protected abstract DataBuilder<TGameConfiguration, TData> BuildDataBuilder(TGameConfiguration game);
        internal abstract ReplayMemory<TData> BuildMemory(TGameConfiguration game);
        internal abstract Trainer<TGameConfiguration, TData> BuildTrainer(TGameConfiguration game);

        private static void LoadLatestModelToTrainer(Trainer<TGameConfiguration, TData> trainer) {
            var chosenFile = Directory.GetFiles("./")
                .Select(x => new FileInfo(x))
                .Where(x => x.Extension == ".modl")
                .OrderByDescending(x => x.LastWriteTime)
                .FirstOrDefault();

            if (chosenFile == null){
                Console.WriteLine($"No model found in dir {Directory.GetCurrentDirectory()}");
                return;
            }

            LoadModelToTrainer(trainer, chosenFile);
        }

        private static void LoadModelToTrainer(Trainer<TGameConfiguration, TData> trainer, FileInfo file)
        {
            if (!file.Exists){
                Console.WriteLine($"Cannot load model {file.FullName} from dir {Directory.GetCurrentDirectory()}");
            }

            Console.WriteLine($"Loading model {file.FullName}");
            trainer.Load(file.OpenRead());
        }
    }
}