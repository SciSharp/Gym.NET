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

        public IEnv EnvInstance => _env.Value;
        public ImageStackLayout ImageStackLayout => ImageStackLayout.Vertical;
        public int MemoryFrames => 4;
        public int MemoryCapacity => 100;
        public int SkippedFrames => 0;
        public int FrameWidth => 600;
        public int FrameHeight => 400;
        public int ScaledImageWidth => 60;
        public int ScaledImageHeight => 60;
        public FramePadding FramePadding => new FramePadding { Top = 150, Bottom = 100, Left = 200, Right = 200};
        public float StartingEpsilon => .7F;
        public int Episodes => 4000;
        public int BatchSize => 100;
        public int Epochs => 10;
    }
}
