namespace AtariDeepQLearner.Runner
{
    class Program
    {
        static void Main(string[] args)
        {
            new GameEngine().Play(new CartPoleConfiguration());
        }
    }
}
