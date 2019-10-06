using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Gym.Envs;
using Gym.Observations;
using ReinforcementLearning.GameConfigurations;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning
{
    public class GameEngine
    {
        private readonly Random _random = new Random();
        private Trainer _trainer;
        private readonly Imager _imager = new Imager();

        public void Play<TGame>(TGame game)
            where TGame : IGameConfiguration
        {
            var env = game.EnvIstance;
            _trainer = new Trainer(game.ScaledImageWidth, game.ScaledImageHeight, env.ActionSpace.Shape.Size, game.BatchSize, game.Epochs);
            LoadModelToTrainer(_trainer);
            PlayGame(game, env);

            var memory = new ReplayMemory(game.MemoryFrames, game.FrameWidth, game.FrameHeight);
            var ct = new CancellationTokenSource();
            _trainer.StartAsyncTraining(memory, ct.Token);

            env.Seed(0);

            var rewards = new List<float>();
            Image<Rgba32> oldImage = null;

            var skipFrameCount = 0;
            for (var i = 0; i < game.Episodes; i++)
            {
                var epsilon = (float)(game.Episodes - i) / game.Episodes;
                Console.WriteLine($"Stage [{i + 1}]/[{game.Episodes}], with exploration rate {epsilon}");

                env.Reset();
                env.Step(env.ActionSpace.Sample());
                float episodeReward = 0;
                var action = 0;
                while (true)
                {
                    Step currentState;
                    if (skipFrameCount >= game.SkippedFrames)
                    {
                        skipFrameCount = 0;
                        currentState = env.Step(action);
                    }
                    else
                    {
                        action = ComposeAction(game, env, memory, oldImage, epsilon);

                        currentState = env.Step(action);
                        if (oldImage != null)
                        {
                            memory.Memorize(oldImage, action, currentState.Reward, currentState.Done);
                        }
                    }

                    oldImage = env.Render();
                    episodeReward += currentState.Reward;
                    if (currentState.Done)
                    {
                        memory.EndEpisode();
                        rewards.Add(episodeReward);
                        Console.WriteLine($"Reward: {episodeReward}, average is {rewards.Average()}");
                        break;
                    }
                }
            }
            ct.Cancel();

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\n\nTraining completed\n\n");
            Console.ResetColor();

            PlayGame(game, env);
        }

        private void PlayGame<TGame>(TGame game, IEnv env) where TGame : IGameConfiguration
        {
            Console.WriteLine("Press [p] play the game with the current trained model, any other key to skip");
            var pressed = Console.ReadKey().KeyChar;
            if (pressed != 'p')
            {
                return;
            }

            var rewards = new List<float>();
            var episodesRewards = new List<float>();
            var imageQueue = new Queue<Image<Rgba32>>(game.MemoryFrames);
            env.Reset();

            while (true)
            {
                var image = env.Render();
                imageQueue.Enqueue(image);
                Step currentState;
                if (imageQueue.Count == game.MemoryFrames)
                {
                    var action = PredictAction(game, imageQueue.ToArray());
                    currentState = env.Step(action);
                    imageQueue.Dequeue();
                }
                else
                {
                    currentState = env.Step(env.ActionSpace.Sample());
                }

                rewards.Add(currentState.Reward);
                if (currentState.Done)
                {
                    var espisodeReward = rewards.Sum();
                    episodesRewards.Add(espisodeReward);
                    Console.WriteLine($"Reward:  {espisodeReward}, average is {episodesRewards.Average()}");
                    rewards = new List<float>();
                    env.Reset();
                    env.Step(env.ActionSpace.Sample());
                }
            }
        }

        private static void LoadModelToTrainer(Trainer trainer)
        {
            Console.WriteLine("Press [L] to load last saved model, any other key to train new one");
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

        private int ComposeAction(IGameConfiguration configuration, IEnv env, ReplayMemory memory, Image<Rgba32> oldImage, float epsilon)
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

            return PredictAction(configuration, current);
        }

        private int PredictAction(IGameConfiguration configuration, Image<Rgba32>[] current)
        {
            var processedImage = _imager.Load(current)
                .ComposeFrames(configuration.ScaledImageWidth, configuration.ScaledImageHeight)
                .InvertColors()
                .Grayscale()
                .Compile()
                .Rectify()
                .ToArray();

            var prediction = _trainer.Predict(processedImage);
            return prediction.ToList().IndexOf(prediction.Max());
        }
    }
}
