using Avalonia;
using Avalonia.Logging.Serilog;

namespace Gym.Rendering.Avalonia
{
    class Program
    {
        //public static void Main(string[] args) => BuildAvaloniaApp().Start(AppMain, args);
        public static void Main(string[] args) => AvaloniaEnvViewer.Run(200, 200, "ssss");

        public static AppBuilder BuildAvaloniaApp()
            => AppBuilder.Configure<App>()
                .UsePlatformDetect()
                .LogToDebug();

        private static void AppMain(Application app, string[] args)
        {
            app.Run(new AvaloniaEnvViewer());
        }
    }
}
