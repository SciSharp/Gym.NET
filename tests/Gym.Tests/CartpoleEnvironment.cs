﻿using System;
using System.Threading;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Rendering.Avalonia;
using Gym.Rendering.WinForm;
using Gym.Tests.Helpers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests {
    [TestClass]
    public class CartpoleEnvironment {
        [TestMethod]
        public void Run() {
            var cp = new CartPoleEnv(WinFormEnvViewer.Factory); //or AvaloniaEnvViewer.Factory
            var rnd = new Random(42);
            var done = true;
            using (new StopwatchMeasurer("time it took to run all steps in ms"))
                for (int i = 0; i < 100_000; i++) {
                    if (done) {
                        cp.Reset();
                        done = false;
                    } else {
                        var (observation, reward, _done, information) = cp.Step((i % 2));
                        done = _done;
                    }

                    cp.Render();
                    Thread.Sleep(15); //this is to prevent it from finishing instantly !
                }
        }
    }
}