using ReinforcementLearning.GameEngines;

namespace ReinforcementLearning.Images.Runner
{
    class Program
    {
        static void Main(string[] args)
        {
            new ImageGameEngine().Play(new CartPoleConfiguration());
        }
    }
}
