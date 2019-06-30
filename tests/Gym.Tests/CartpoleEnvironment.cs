using System;
using System.Threading;
using Ebby.Gym.Envs.Classic;
using Gym.Tests.Helpers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests {
    [TestClass]
    public class CartpoleEnvironment {
        [TestMethod]
        public void Run() {
            //Environment.SetEnvironmentVariable("PATH", Environment.GetEnvironmentVariable("PATH") + @";C:\Program Files\ArrayFire\v3\lib");
            //Device.SetBackend(Backend.CPU);

            //var v = Viewer.Run(600, 500, "heythere");
            //v.TestRendering();
            //Thread.Sleep(300);

            var cp = new CartPoleEnv();
            var rnd = new Random();
            var done = true;
            using (var sw = new StopwatchMeasurer("time it took to run all steps in ms"))
                for (int i = 0; i < 100_000; i++) {
                    if (done) {
                        cp.Reset();
                        done = false;
                    } else {
                        var (observation, reward, _done, information) = cp.Step((i % 2));
                        done = _done;
                    }

                    cp.Render();
                    Thread.Sleep(30);
                }

            Console.ReadLine();
        }
    }
}