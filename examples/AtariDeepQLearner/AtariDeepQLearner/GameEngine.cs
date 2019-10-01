using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AtariDeepQLearner.GameConfigurations;
using Gym.Envs;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace AtariDeepQLearner
{
    public class GameEngine
    {
        private readonly Random _random= new Random();

        public void Play<TGame>(TGame game)
            where TGame : IGameConfiguration
        {
            var env = game.EnvIstance;
            var trainer = new Trainer(game.ScaledImageWidth, game.ScaledImageHeight, env.ActionSpace.Shape.Size, game.Epochs);
            var memory = new ReplayMemory(game.MemoryFrames, game.FrameWidth, game.FrameHeight);
            var imager = new Imager();

            env.Seed(0);

            var rewards = new List<float>();
            Image<Rgba32> oldImage = null;

            for (var i = 0; i < game.Episodes; i++)
            {
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
                        memory.Memorize(oldImage, int.Parse(action.ToString()), currentState.Reward);
                    }
                    oldImage = env.Render();
                    episodeReward += currentState.Reward;
                    if (currentState.Done)
                    {
                        memory.EndEpisode();
                        Console.WriteLine("Reward: " + episodeReward);
                        rewards.Add(episodeReward);
                        var data = trainer.BuildStuff(memory);
                        trainer.TrainOnMemory(data);
                        break;
                    }
                }

                Console.WriteLine($"Stage [{i}]/[{game.Episodes}], reward {rewards.Last()}");
#pragma warning disable 4014
                //Task.Run(() => memory.Save($"memory {DateTime.Now:yyyyMMdd-HH-mm-ss}.json", 10));
#pragma warning restore 4014
            }

            Console.WriteLine("Average Reward: " + rewards.Average());
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

        private static NDArray PredictionToPython(float[] prediction) =>
            prediction.ToList().IndexOf(prediction.Max());
    }
}
