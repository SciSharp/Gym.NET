using System;
using System.Threading;
using System.Threading.Tasks;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Rendering.Avalonia;
using Gym.Rendering.WinForm;
using Gym.Tests.Helpers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Envs.Classic {
    [TestClass]
    public class CartpoleEnvironment {
        public void Run(IEnvironmentViewerFactoryDelegate viewer) {
            var cp = new CartPoleEnv(viewer); //or AvaloniaEnvViewer.Factory
            var done = true;
            try {
                using (new StopwatchMeasurer("time it took to run all steps in ms")) {
                    for (int i = 0; i < 1_000; i++) {
                        if (done) {
                            cp.Reset();
                            done = false;
                        } else {
                            var (observation, reward, _done, information) = cp.Step((i % 2));
                            done = _done;
                        }

                        cp.Render();
                        //Thread.Sleep(5); //this is to prevent it from finishing instantly !
                    }
                }
            } finally {
                cp.CloseEnvironment();
            }
        }

        [TestMethod]
        public void Run_WinFormEnv() {
            Run(WinFormEnvViewer.Factory);
        }

        [TestMethod]
        public void Run_AvaloniaEnv() {
            Run(AvaloniaEnvViewer.Factory);
        }

        [TestMethod]
        public void Run_NullEnv() {
            Run(NullEnvViewer.Factory);
        }
    }
}