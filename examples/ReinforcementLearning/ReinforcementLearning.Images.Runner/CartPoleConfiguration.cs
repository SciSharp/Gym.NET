using System;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Rendering.Avalonia;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using ReinforcementLearning.GameConfigurations;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning.Images.Runner {
    public sealed class CartPoleConfiguration : IImageGameConfiguration {
        private readonly Lazy<IEnv> _env = new Lazy<IEnv>(() => new CartPoleEnv(AvaloniaEnvViewer.Run));

        public IEnv EnvInstance => _env.Value;
        public int MemoryStates => 2;
        public ImageStackLayout ImageStackLayout => ImageStackLayout.Vertical;
        public int MemoryCapacity => 100;
        public int SkippedFrames => 1;
        public int FrameWidth => 600;
        public int FrameHeight => 400;
        public int ScaledImageWidth => 40;
        public int ScaledImageHeight => 40;
        public FramePadding FramePadding => new FramePadding {Top = 150, Bottom = 100, Left = 200, Right = 200};
        public float StartingEpsilon => 1F;
        public int Episodes => 4000;
        public int BatchSize => 100;
        public int Epochs => 10;

        public INeuralNetwork BuildNeuralNetwork() =>
            NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(ScaledImageHeight, ScaledImageWidth),
                NetworkLayers.Convolutional((4, 4), 40, ActivationType.ReLU),
                NetworkLayers.FullyConnected(20, ActivationType.ReLU),
                NetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));
    }
}