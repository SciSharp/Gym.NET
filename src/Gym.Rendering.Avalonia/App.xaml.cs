using Avalonia;
using Avalonia.Markup.Xaml;

namespace Gym.Rendering.Avalonia
{
    public class App : Application
    {
        public override void Initialize()
        {
            AvaloniaXamlLoader.Load(this);
        }
   }
}