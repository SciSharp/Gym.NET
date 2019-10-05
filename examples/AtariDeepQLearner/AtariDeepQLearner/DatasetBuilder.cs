using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces.Data;

namespace AtariDeepQLearner
{
    internal class DatasetBuilder
    {
        private readonly int _batchSize;
        private readonly int _inputImgHeight;
        private readonly int _inputImgWidth;
        private readonly int _outputs;
        private readonly Imager _imager = new Imager();
        private readonly Dictionary<int, (float[] x, float[] y)> _observationDictionary = new Dictionary<int, (float[] x, float[] y)>();

        public DatasetBuilder(int inputImgWidth, int inputImgHeight, int outputs, int batchSize)
        {
            _inputImgWidth = inputImgWidth;
            _inputImgHeight = inputImgHeight;
            _outputs = outputs;
            _batchSize = batchSize;
        }

        public ITrainingDataset BuildDataset(ReplayMemory memory)
        {
            var bestEpisodes = memory
                .Episodes
                .OrderByDescending(x => x.TotalReward)
                .Take(memory.Episodes.Count / 10)
                .ToList();

            var observations = bestEpisodes
                .SelectMany(x => x.Observations)
                .ToList();

            Console.WriteLine($"Training on best {bestEpisodes.Count} episodes out of {memory.Episodes.Count}. {observations.Count} observations");

            return DatasetLoader.Training(BuildRawData(observations), _batchSize);
        }

        private IEnumerable<(float[] x, float[] y)> BuildRawData(IEnumerable<Observation> bestObservations)
        {
            foreach (var observation in bestObservations)
            {
                if (_observationDictionary.ContainsKey(observation.Id))
                {
                    yield return _observationDictionary[observation.Id];
                    continue;
                }

                var x = _imager
                    .Load(observation.Images)
                    .ComposeFrames(_inputImgWidth, _inputImgHeight)
                    .InvertColors()
                    .Grayscale()
                    .Compile()
                    .Rectify()
                    .ToArray();

                var y = new float[_outputs];
                y[observation.ActionTaken] = 1;
                _observationDictionary.Add(observation.Id, (x, y));
                yield return (x, y);
            }
        }
    }
}
