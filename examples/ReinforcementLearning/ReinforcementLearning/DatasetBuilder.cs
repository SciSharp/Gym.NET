using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces.Data;
using ReinforcementLearning.GameConfigurations;

namespace ReinforcementLearning
{
    internal class DatasetBuilder
    {
        private readonly IGameConfiguration _configuration;
        private readonly int _outputs;
        private readonly Imager _imager = new Imager();
        private readonly Dictionary<int, (float[] x, float[] y)> _observationDictionary = new Dictionary<int, (float[] x, float[] y)>();

        public DatasetBuilder(IGameConfiguration configuration, int outputs)
        {
            _configuration = configuration;
            _outputs = outputs;
        }

        public ITrainingDataset BuildDataset(IConcurrentMemory memory)
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

            return DatasetLoader.Training(BuildRawData(observations), _configuration.BatchSize);
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
                    .ComposeFrames(_configuration.ScaledImageWidth, _configuration.ScaledImageHeight, _configuration.ImageStackLayout)
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
