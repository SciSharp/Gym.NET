using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Observations;
using ReinforcementLearning.DataBuilders;
using ReinforcementLearning.GameConfigurations;
using ReinforcementLearning.MemoryTypes;

namespace ReinforcementLearning.PlaySessions {
    internal abstract class BasePlaySession<TGameConfiguration, TData>
        where TGameConfiguration : IGameConfiguration {
        protected readonly Trainer<TGameConfiguration, TData> Trainer;
        protected IGameConfiguration Game;
        protected ReplayMemory<TData> Memory;
        protected readonly DataBuilder<TGameConfiguration, TData> DataBuilder;
        protected int Framescount;
        protected int Action;
        protected Step CurrentState;
        protected float CurrentEpisodeReward;
        protected Queue<float> EpisodeRewards = new Queue<float>();
        protected Random Random = new Random();

        protected BasePlaySession(TGameConfiguration game, Trainer<TGameConfiguration, TData> trainer,
            ReplayMemory<TData> memory, DataBuilder<TGameConfiguration, TData> dataBuilder) {
            Game = game;
            Trainer = trainer;
            Memory = memory;
            DataBuilder = dataBuilder;
            CurrentState = new Step();
        }

        public void Play() {
            Game.EnvInstance.Seed(0);

            for (var i = 0; i < Game.Episodes; i++) {
                OnEpisodeStart(i);
                Framescount = Game.SkippedFrames + 1;

                Game.EnvInstance.Reset();
                CurrentState = new Step();
                while (true) {
                    var image = Game.EnvInstance.Render();

                    if (Framescount < Game.SkippedFrames + 1) {
                        CurrentState = Game.EnvInstance.Step(Action);
                    }
                    else {
                        var currentFrame = Memory.Enqueue(image, CurrentState);
                        if (currentFrame != null && currentFrame.Length == Game.MemoryStates) {
                            Action = ComposeAction(currentFrame);
                            CurrentState = Game.EnvInstance.Step(Action);
                            Memory.Memorize(Action, CurrentState.Reward, CurrentState.Done);
                        }

                        Framescount = 0;
                    }

                    CurrentEpisodeReward += CurrentState.Reward;
                    if (CurrentState.Done || Framescount > 1000) {
                        OnEpisodeDone(CurrentEpisodeReward);
                        EpisodeRewards.Enqueue(CurrentEpisodeReward);
                        if (EpisodeRewards.Count > 100) {
                            EpisodeRewards.Dequeue();
                        }

                        Console.WriteLine($"Reward: {CurrentEpisodeReward}, average on last 100 is {EpisodeRewards.Average()}");
                        CurrentEpisodeReward = 0;
                        break;
                    }

                    Framescount++;
                }
            }

            OnCompleted();
        }

        protected virtual int ComposeAction(TData[] currentData) {
            var processedImage = DataBuilder.BuildInput(currentData);
            var prediction = Trainer.Predict(processedImage);
            return prediction.ToList().IndexOf(prediction.Max());
        }

        protected virtual void OnEpisodeStart(int episodeIndex) {
        }

        protected virtual void OnEpisodeDone(float episodeReward) {
        }

        protected virtual void OnCompleted() {
        }
    }
}