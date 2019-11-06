
# How to run

To run the sample:
- build the solution
- set one of the two available project as startup project in VisualStudio solution explorer (ReinforcementLearning.Parameters.Runner or ReinforcementLearning.Images.Runner)
- run
- choose [L] to load the latest trained NN or [P] to load the pre-trained NN.
- choose 1 to skip trainin and run the agent.

To train a new agent:
- build the solution 
- set one of the two available project as startup project in VisualStudio solution explorer
- run
- press any key to avoid loading the model
- choose 2 to start training


# Structure
The example is splitted into two different .Net Core 3.0 projects both training the CartPole classic game using two different approaches.

All the hyperparameters can be found and changed in the configuration file named CartPoleConfiguration.cs found in both projects.

## ReinforcementLearning.Parameters.Runner
Uses the parameters (es: position and speed) given by the running environment to train the neural network. 
The training converges at ~2000 games played.

## ReinforcementLearning.Images.Runner
Uses the raw image given by the running environment to train the neural network.
The pre trained NN does not perform as well as the parameter version and at the moment is capable of keeping the average at ~ 60/65 reward per episodes.
