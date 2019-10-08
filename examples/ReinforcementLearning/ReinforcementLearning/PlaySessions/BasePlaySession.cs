using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Observations;
using ReinforcementLearning.GameConfigurations;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning.PlaySessions
{
    internal abstract class BasePlaySession
    {
        protected IGameConfiguration Game;
        protected Imager Imager;
        protected Trainer Trainer;
        protected ReplayMemory Memory;
        protected int Framescount;
        protected int? Action;
        protected Step CurrentState;
        protected List<float> EpisodeRewards;
        protected Random Random = new Random();
        protected float Epsilon;

        protected BasePlaySession()
        {
            EpisodeRewards = new List<float>();
            Imager = new Imager();
            CurrentState = new Step();
            Action = null;
            Framescount = 0;
        }

        public void Play<TGameConfiguration>(TGameConfiguration game, Trainer trainer)
            where TGameConfiguration : IGameConfiguration
        {
            Game = game;
            Memory = new ReplayMemory(Game.MemoryFrames, Game.FrameWidth, Game.FrameHeight);

            game.EnvIstance.Seed(0);

            for (var i = 0; i < game.Episodes; i++)
            {
                Epsilon = game.StartingEpsilon * (game.Episodes - i) / game.Episodes;
                Console.WriteLine($"Stage [{i + 1}]/[{game.Episodes}], with exploration rate {Epsilon}");

                game.EnvIstance.Reset();
                while (true)
                {
                    var image = game.EnvIstance.Render();

                    if (Framescount < game.SkippedFrames + 1 && Action.HasValue)
                    {
                        CurrentState = game.EnvIstance.Step(Action.Value);
                    }
                    else
                    {
                        var currentFrame = Memory.Enqueue(image);
                        if (currentFrame != null && currentFrame.Length == game.MemoryFrames)
                        {
                            Action = ComposeAction(currentFrame);
                            CurrentState = game.EnvIstance.Step(Action.Value);
                            Memory.Memorize(Action.Value, CurrentState.Reward, CurrentState.Done);
                        }

                        Framescount = 0;
                    }

                    EpisodeRewards.Add(CurrentState.Reward);
                    if (CurrentState.Done || Framescount > 1000)
                    {
                        OnEpisodeDone();
                        Console.WriteLine($"Reward: {EpisodeRewards.Sum()}, average is {EpisodeRewards.Average()}");
                        EpisodeRewards = new List<float>();
                        break;
                    }

                    Framescount++;
                }
            }

            OnCompleted();
        }

        protected virtual int ComposeAction(Image<Rgba32>[] current)
        {
            var processedImage = Imager.Load(current)
                .Crop(Game.FramePadding)
                .ComposeFrames(Game.ScaledImageWidth, Game.ScaledImageHeight, Game.ImageStackLayout)
                .InvertColors()
                .Grayscale()
                .Compile()
                .Rectify()
                .ToArray();

            var prediction = Trainer.Predict(processedImage);
            return prediction.ToList().IndexOf(prediction.Max());
        }

        protected virtual void OnEpisodeDone()
        { }

        protected virtual void OnCompleted()
        { }
    }
}
