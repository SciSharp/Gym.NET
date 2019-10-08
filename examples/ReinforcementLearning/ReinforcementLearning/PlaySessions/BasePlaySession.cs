using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Observations;
using ReinforcementLearning.GameConfigurations;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning.PlaySessions
{
    internal abstract class BasePlaySession<TGameConfiguration>
        where TGameConfiguration : IGameConfiguration
    {
        protected readonly Trainer Trainer;
        protected IGameConfiguration Game;
        protected Imager Imager;
        protected ReplayMemory Memory;
        protected int Framescount;
        protected int? Action;
        protected Step CurrentState;
        protected float CurrentEpisodeReward;
        protected List<float> EpisodeRewards = new List<float>();
        protected Random Random = new Random();

        protected BasePlaySession(TGameConfiguration game, Trainer trainer)
        {
            Game = game;
            Trainer = trainer;
            Imager = new Imager();
            CurrentState = new Step();
            Memory = new ReplayMemory(Game.MemoryFrames, Game.FrameWidth, Game.FrameHeight);
        }

        public void Play()
        {
            Game.EnvIstance.Seed(0);

            for (var i = 0; i < Game.Episodes; i++)
            {
                OnEpisodeStart(i);

                Game.EnvIstance.Reset();
                while (true)
                {
                    var image = Game.EnvIstance.Render();

                    if (Framescount < Game.SkippedFrames + 1 && Action.HasValue)
                    {
                        CurrentState = Game.EnvIstance.Step(Action.Value);
                    }
                    else
                    {
                        var currentFrame = Memory.Enqueue(image);
                        if (currentFrame != null && currentFrame.Length == Game.MemoryFrames)
                        {
                            Action = ComposeAction(currentFrame);
                            CurrentState = Game.EnvIstance.Step(Action.Value);
                            Memory.Memorize(Action.Value, CurrentState.Reward, CurrentState.Done);
                        }

                        Framescount = 0;
                    }

                    CurrentEpisodeReward += CurrentState.Reward;
                    if (CurrentState.Done || Framescount > 1000)
                    {
                        OnEpisodeDone();
                        EpisodeRewards.Add(CurrentEpisodeReward);
                        Console.WriteLine($"Reward: {CurrentEpisodeReward}, average is {EpisodeRewards.Average()}");
                        CurrentEpisodeReward = 0;
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

        protected virtual void OnEpisodeStart(int episodeIndex)
        { }

        protected virtual void OnEpisodeDone()
        { }

        protected virtual void OnCompleted()
        { }
    }
}
