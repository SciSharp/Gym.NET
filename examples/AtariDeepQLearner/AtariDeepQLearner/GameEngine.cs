using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AtariDeepQLearner.GameConfigurations;
using Gym.Envs;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace AtariDeepQLearner
{
    public class GameEngine
    {
        private readonly Random _random = new Random();

        public void Play<TGame>(TGame game)
            where TGame : IGameConfiguration
        {
            var env = game.EnvIstance;
            var trainer = new Trainer(game.ScaledImageWidth, game.ScaledImageHeight, env.ActionSpace.Shape.Size, game.BatchSize, game.Epochs);
            LoadModelToTrainer(trainer);

            var memory = new ReplayMemory(game.MemoryFrames, game.FrameWidth, game.FrameHeight);
            var imager = new Imager();

            env.Seed(0);

            var rewards = new List<float>();
            Image<Rgba32> oldImage = null;

            for (var i = 0; i < game.Episodes; i++)
            {
                Console.WriteLine($"Stage [{i + 1}]/[{game.Episodes}]");

                env.Reset();
                env.Step(env.ActionSpace.Sample());
                float episodeReward = 0;
                while (true)
                {
                    var epsilon = (float)(game.Episodes - i) / game.Episodes;

                    var action = ComposeAction(game, env, trainer, memory, imager, oldImage, epsilon);

                    var currentState = env.Step(action);
                    if (oldImage != null)
                    {
                        memory.Memorize(oldImage, action, currentState.Reward);
                    }
                    oldImage = env.Render();
                    episodeReward += currentState.Reward;
                    if (currentState.Done)
                    {
                        memory.EndEpisode();
                        Console.WriteLine("Reward: " + episodeReward);
                        rewards.Add(episodeReward);
                        if (i % 10 == 0)
                        {
                            trainer.TrainOnMemory(memory);
                        }
                        break;
                    }
                }

#pragma warning disable 4014
                //Task.Run(() => memory.Save($"memory {DateTime.Now:yyyyMMdd-HH-mm-ss}.json", 10));
#pragma warning restore 4014
            }

            Console.WriteLine("Average Reward: " + rewards.Average());
        }

        private static void LoadModelToTrainer(Trainer trainer)
        {
            Console.WriteLine("Press [L] to load last saved model");
            var pressed = Console.ReadKey().KeyChar;
            if (pressed != 'l')
            {
                return;
            }

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

        private NDArray ComposeAction(IGameConfiguration configuration, IEnv env, Trainer trainer, ReplayMemory memory, Imager imager, Image<Rgba32> oldImage, float epsilon)
        {
            if (oldImage == null)
            {
                return env.ActionSpace.Sample();
            }
            var current = memory.GetCurrent();
            if (current == null || current.Length < configuration.MemoryFrames)
            {
                return env.ActionSpace.Sample();
            }

            if (_random.NextDouble() <= epsilon)
            {
                return env.ActionSpace.Sample();
            }

            var processedImage = imager.Load(current)
                .ComposeFrames(configuration.ScaledImageWidth, configuration.ScaledImageHeight)
                .InvertColors()
                .Grayscale()
                .Compile()
                .Rectify();

            return PredictionToPython(trainer.Predict(processedImage.ToArray(), epsilon));
        }

        private static int PredictionToPython(float[] prediction) =>
            prediction.ToList().IndexOf(prediction.Max());
    }
}
