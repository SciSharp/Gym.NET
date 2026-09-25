using System;
using System.Threading;
using System.Threading.Tasks;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Rendering.Avalonia;
using Gym.Rendering.WinForm;
using Gym.Tests.Helpers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Envs.Classic {
    /// <summary>
    ///     Viewer smoke tests: a thousand CartPole steps in <c>human</c> render mode, which renders into the viewer after every
    ///     reset and step (Gymnasium's human mode), with each viewer backend.
    /// </summary>
    [TestClass]
    public class CartpoleEnvironment {
        /// <summary>
        ///     Runs a thousand alternating pushes in human mode, resetting whenever an episode ends. The frame rate is raised
        ///     from CartPole's 50 fps so the pacing clock does not stretch the test to 20 seconds.
        /// </summary>
        /// <param name="viewer">The viewer backend to render into.</param>
        public void Run(IEnvironmentViewerFactoryDelegate viewer) {
            var cp = new CartPoleEnv(renderMode: "human", viewerFactory: viewer); //or AvaloniaEnvViewer.Factory
            cp.Metadata[Metadata.RenderFpsKey] = 1000L;
            var done = true;
            try {
                using (new StopwatchMeasurer("time it took to run all steps in ms")) {
                    for (int i = 0; i < 1_000; i++) {
                        if (done) {
                            cp.Reset().Observation.Dispose();
                            done = false;
                        } else {
                            var (observation, reward, terminated, truncated, information) = cp.Step(i % 2);
                            observation.Dispose();
                            done = terminated || truncated;
                        }
                        //Thread.Sleep(5); //this is to prevent it from finishing instantly !
                    }
                }
            } finally {
                cp.Close();
            }
        }

        /// <summary>Renders into a Windows Forms window.</summary>
        [TestMethod]
        public void Run_WinFormEnv() {
            Run(WinFormEnvViewer.Factory);
        }

        /// <summary>Renders into an Avalonia window.</summary>
        [TestMethod]
        public void Run_AvaloniaEnv() {
            Run(AvaloniaEnvViewer.Factory);
        }

        /// <summary>Renders into the headless viewer.</summary>
        [TestMethod]
        public void Run_NullEnv() {
            Run(NullEnvViewer.Factory);
        }
    }
}
