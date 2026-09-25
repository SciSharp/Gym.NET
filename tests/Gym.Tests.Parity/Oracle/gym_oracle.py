"""The Python half of Gym.NET's live parity suite (tests/Gym.Tests.Parity).

GymPythonSession loads this file into the embedded interpreter once per test run (it is an embedded resource of the
test assembly) and registers it in ``sys.modules`` as ``gym_oracle``. The C# side calls into it through pythonnet and
never interprets a Python object itself: every value that crosses back is first reduced by :func:`describe` to nested
tuples of ``str``, ``int``, ``bytes`` and ``bool``. The C# comparer therefore sees exactly the bits Gymnasium produced
(dtype strings, shapes and C-order bytes for arrays, IEEE-754 bit patterns for floats) with none of pythonnet's
conversion rules in between.

Calls that may legitimately raise go through :func:`invoke`, which returns the exception as data, so a Gymnasium error
is compared with Gym.NET's instead of aborting the test.

Every function expects the caller to hold the GIL.
"""

from __future__ import annotations

import platform
import struct
import sys
from typing import Any, Callable

import numpy as np

import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding

# The tags describe() puts first in every tuple. ValueFacts.FromDescriptor (Oracle/ValueFacts.cs) reads the same tags;
# change both together.
NONE = "none"
BOOL = "bool"
INT = "int"
FLOAT = "float"
ARRAY = "ndarray"
STR = "str"
DICT = "dict"
SEQ = "seq"
OTHER = "other"

# The tags invoke() puts first in its result.
OK = "ok"
RAISED = "raised"


def type_name(value: Any) -> str:
    """The value's type as reports print it: ``float``, ``numpy.float32``, ``numpy.ndarray``, ``gymnasium.error.Error``.

    Builtins go unqualified because ``builtins.float`` only adds noise; everything else keeps its module, because
    ``Error`` alone would not say which one.
    """
    kind = type(value)
    return kind.__qualname__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__qualname__}"


def describe(value: Any) -> tuple:
    """Reduces a value to a tagged tuple that the C# side reads without guessing.

    Shapes of the result (``type`` is :func:`type_name`):

    - ``None`` -> ``("none", type)``
    - ``bool``, ``numpy.bool`` -> ``("bool", type, value)``
    - ``int`` -> ``("int", type, "", decimal)``: a Python int is unbounded, so it has no dtype
    - numpy integer scalar -> ``("int", type, dtype.str, decimal)``
    - numpy floating scalar -> ``("float", type, dtype.str, bytes)``: its own width and byte order
    - ``float`` -> ``("float", type, "<f8", bytes)``: little-endian IEEE-754 double
    - ``numpy.ndarray`` -> ``("ndarray", type, dtype.str, shape, bytes)``: C-order bytes whatever the layout
    - ``str`` -> ``("str", type, value)``
    - ``dict`` -> ``("dict", type, ((key, value), ...))``: insertion order, both sides described
    - ``tuple``, ``list`` -> ``("seq", type, (item, ...))``
    - anything else -> ``("other", type, repr(value))``
    """
    name = type_name(value)
    if value is None:
        return (NONE, name)
    # bool first: Python's bool subclasses int, and numpy.bool is not a numpy.integer.
    if isinstance(value, (bool, np.bool_)):
        return (BOOL, name, bool(value))
    if isinstance(value, np.ndarray):
        return (ARRAY, name, value.dtype.str, tuple(int(n) for n in value.shape), value.tobytes(order="C"))
    if isinstance(value, np.integer):
        return (INT, name, value.dtype.str, str(int(value)))
    # numpy.floating before float: numpy.float64 subclasses Python's float, and must keep its numpy type name.
    if isinstance(value, np.floating):
        return (FLOAT, name, value.dtype.str, value.tobytes())
    if isinstance(value, int):
        return (INT, name, "", str(value))
    if isinstance(value, float):
        return (FLOAT, name, "<f8", struct.pack("<d", value))
    if isinstance(value, str):
        return (STR, name, value)
    if isinstance(value, dict):
        return (DICT, name, tuple((describe(key), describe(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return (SEQ, name, tuple(describe(item) for item in value))
    return (OTHER, name, repr(value))


def invoke(function: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple:
    """Calls ``function`` and returns ``("ok", describe(result))``, or ``("raised", type, message)`` if it raised.

    Returning the exception as data lets a test compare Gymnasium's error (type and verbatim message) with the one
    Gym.NET throws for the same input. ``message`` is ``str(exception)``, the text Python prints after the type.
    """
    try:
        return (OK, describe(function(*args, **kwargs)))
    except Exception as error:  # noqa: BLE001 - every exception is a comparable outcome here
        return (RAISED, type_name(error), str(error))


def evaluate(expression: str, **names: Any) -> Any:
    """Evaluates a Python expression with ``np``, ``gymnasium``, ``gym``, ``spaces`` and ``seeding`` bound, plus ``names``.

    Returns the raw object, for the C# side to keep as a handle (a space, a generator). It is the escape hatch for
    anything a C# builder cannot spell, like NumSharp's ``Python.np.eval``.
    """
    scope = {"np": np, "gymnasium": gymnasium, "gym": gymnasium, "spaces": spaces, "seeding": seeding}
    scope.update(names)
    return eval(expression, scope)  # noqa: S307 - the expressions are the test suite's own literals


def stack() -> tuple:
    """The live versions the session compares with tools/parity/manifest.json, as sorted ``(key, value)`` pairs.

    A package that fails to import is reported as ``missing (...)`` instead of raising, so the session can list every
    drift at once rather than stopping at the first.
    """
    info = {
        "gymnasium.version": gymnasium.__version__,
        "gymnasium.file": gymnasium.__file__,
        "python.version": platform.python_version(),
        "python.hash_randomization": str(sys.flags.hash_randomization),
        "numpy.version": np.__version__,
        "numpy.file": np.__file__,
    }
    try:
        import pygame

        info["pygame.version"] = pygame.version.ver
        info["pygame.sdl"] = str(pygame.version.SDL)
    except Exception as error:  # noqa: BLE001
        info["pygame.version"] = info["pygame.sdl"] = f"missing ({type_name(error)}: {error})"
    try:
        import Box2D

        info["box2d.version"] = Box2D.__version__
    except Exception as error:  # noqa: BLE001
        info["box2d.version"] = f"missing ({type_name(error)}: {error})"
    return tuple(sorted(info.items()))


# ---------------------------------------------------------------------------------------------------------------------
# Policies: actions computed from Python's own observation and fed to both sides, so a lockstep run compares dynamics
# only. Each takes (env, observation), where env is the env being stepped, and returns the action as Gymnasium's
# action space holds it.
# ---------------------------------------------------------------------------------------------------------------------


def cartpole_balance(env: gymnasium.Env, observation: np.ndarray) -> int:
    """A proportional-derivative controller that keeps CartPole-v1 upright for all 500 steps of an episode.

    Pushes towards the side the pole falls to, damped by the angular velocity, with the cart's position and velocity
    as a weak restoring term so the cart does not drift off the track. Verified on seeds 0, 1, 7, 42, 1000 and 1234: no
    termination in 500 steps. It reads the float32 observation Gymnasium returned, as an agent would.
    """
    x, x_dot, theta, theta_dot = (float(v) for v in observation)
    return 1 if 0.1 * x + 0.5 * x_dot + 10.0 * theta + 2.0 * theta_dot > 0.0 else 0


def lunar_lander_heuristic(env: gymnasium.Env, observation: np.ndarray) -> Any:
    """Gymnasium's own LunarLander controller (``gymnasium/envs/box2d/lunar_lander.py``, ``heuristic``).

    Returns a Python int for the discrete lander and a float64 array of two throttles for the continuous one, the
    same objects Gymnasium's demo feeds to ``step``.
    """
    from gymnasium.envs.box2d.lunar_lander import heuristic

    return heuristic(env.unwrapped, observation)


POLICIES: dict[str, Callable[[gymnasium.Env, np.ndarray], Any]] = {
    "cartpole_balance": cartpole_balance,
    "lunar_lander_heuristic": lunar_lander_heuristic,
}


class OracleEnv:
    """One Gymnasium env driven by a GymPair.

    ``env`` is what reset/step go to: the ``gymnasium.make`` result itself (``via_make``, with its wrapper stack), or its
    ``unwrapped`` env, for comparisons with a Gym.NET env that has no wrappers yet. Either way the env is created
    through ``gymnasium.make``, so registration kwargs and ``render_mode`` apply exactly as for a user.

    The methods return raw values; the C# side calls them through :func:`invoke`, which describes the result or the
    exception in one place. The last observation is kept here so that named policies act on Gymnasium's own values.
    """

    def __init__(self, env_id: str, kwargs: dict[str, Any], via_make: bool, render_mode: str | None) -> None:
        self.made = gymnasium.make(env_id, render_mode=render_mode, **kwargs)
        self.env = self.made if via_make else self.made.unwrapped
        self.last_observation: Any = None

    def reset(self, seed: int) -> tuple:
        """``env.reset(seed=seed)``: ``(observation, info)``."""
        observation, info = self.env.reset(seed=seed)
        self.last_observation = observation
        return observation, info

    def step(self, action: Any) -> tuple:
        """``env.step(action)``: ``(observation, reward, terminated, truncated, info)``."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.last_observation = observation
        return observation, reward, terminated, truncated, info

    def policy(self, name: str) -> Any:
        """The action the named policy picks for the last observation (an int or an ndarray)."""
        return POLICIES[name](self.env, self.last_observation)

    def rng_state(self) -> dict:
        """The env generator's full state, ``env.unwrapped.np_random.bit_generator.state``."""
        return self.env.unwrapped.np_random.bit_generator.state

    def render(self) -> Any:
        """``env.render()``; an ``rgb_array`` env returns an ``(H, W, 3)`` uint8 array."""
        return self.env.render()

    def set_attribute(self, name: str, value: Any) -> None:
        """Sets an attribute of the unwrapped env, for state injection (``state`` for the classic-control envs)."""
        setattr(self.env.unwrapped, name, value)

    def get_attribute(self, name: str) -> Any:
        """An attribute of the unwrapped env (``state``, ``observation_space``, ...)."""
        return getattr(self.env.unwrapped, name)

    def seed_action_space(self, seed: int) -> None:
        """``env.action_space.seed(seed)``: the sampled action source seeds both sides' action spaces alike."""
        self.env.action_space.seed(seed)

    def sample_action(self) -> Any:
        """``env.action_space.sample()``, fed to ``step`` as is."""
        return self.env.action_space.sample()

    def close(self) -> None:
        """Closes the made env (and with it the unwrapped one), releasing pygame surfaces."""
        self.made.close()
