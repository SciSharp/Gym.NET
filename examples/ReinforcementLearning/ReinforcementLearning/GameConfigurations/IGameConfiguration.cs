using Gym.Envs;
using NeuralNetworkNET.APIs.Interfaces;

namespace ReinforcementLearning.GameConfigurations {
    public interface IGameConfiguration {
        IEnv EnvInstance { get; } // the game instance to play
        int MemoryStates { get; } // how many states will be used as input to the NN
        int MemoryCapacity { get; } // how many episodes will the memory work with
        int SkippedFrames { get; } // used to skip frames during play (reduces memory usage and improves performances but builds the dataset slower)
        float StartingEpsilon { get; } // starting exploration rate. Max value is 1 (exploration), it decays during training up to 0 (exploitation)
        int Episodes { get; } // how many episodes to train/play on
        int BatchSize { get; } // NN batchSize for training
        int Epochs { get; } // NN epochs for training

        INeuralNetwork BuildNeuralNetwork();
    }

    public interface IImageGameConfiguration : IGameConfiguration {
        ImageStackLayout ImageStackLayout { get; } // defines whether images will be stacker vertically or horizontally, if more than one image is required to form NN input       

        int FrameWidth { get; } // width of the game panel
        int FrameHeight { get; } // height of the game panel
        int ScaledImageWidth { get; } // width of the image to feed the NN
        int ScaledImageHeight { get; } // height of the image to feed the NN

        FramePadding FramePadding { get; } // what padding to apply to every frame, used to focus the NN to a specific area of the image
    }

    public interface IParametersGameConfiguration : IGameConfiguration {
        int ParametersLength { get; } // length of the parameter to be expected by the env
    }

    public enum ImageStackLayout {
        Horizontal,
        Vertical
    }

    public class FramePadding {
        public int Left { get; set; }
        public int Top { get; set; }
        public int Right { get; set; }
        public int Bottom { get; set; }
    }
}