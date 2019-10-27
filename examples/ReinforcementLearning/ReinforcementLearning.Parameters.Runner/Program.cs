using ReinforcementLearning.GameEngines;

namespace ReinforcementLearning.Parameters.Runner
{
    class Program
    {
        static void Main(string[] args)
        {
            new ParameterGameEngine().Play(new CartPoleConfiguration());
        }
    }
}
