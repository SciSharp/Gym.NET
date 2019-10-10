using System;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Rendering.Avalonia;
using ReinforcementLearning.GameConfigurations;

namespace ReinforcementLearning.Runner
{
    public sealed class CartPoleConfiguration : IGameConfiguration
    {
        private readonly Lazy<IEnv> _env = new Lazy<IEnv>(() => new CartPoleEnv(AvaloniaEnvViewer.Run));

        public IEnv EnvIstance => _env.Value;
        public ImageStackLayout ImageStackLayout => ImageStackLayout.Vertical;
        public int MemoryFrames => 2;
        public int MemoryCapacity => 100;
        public int SkippedFrames => 3;
        public int FrameWidth => 600;
        public int FrameHeight => 400;
        public int ScaledImageWidth => 50;
        public int ScaledImageHeight => 50;
        public FramePadding FramePadding => new FramePadding { Top = 150, Bottom = 100 };
        public float StartingEpsilon => 1F;
        public int Episodes => 2000;
        public int BatchSize => 28;
        public int Epochs => 10;
    }
}
