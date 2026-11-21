#nullable enable
using Gym.Envs;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     The base of the MuJoCo envs' typed info records (DR-91, DR-95), as <see cref="MujocoEnv"/> is the base of the envs:
    ///     each env class has its record (<c>HalfCheetahInfoV5 info = step.Info;</c> for <see cref="HalfCheetahEnvV5"/>). It
    ///     holds no member of its own, since no key is common to every MuJoCo env.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The keys and their order are Gymnasium v1.4.0's, measured on every MuJoCo id (a reset with seed 0, then 300 random
    ///     steps). A v5 env's reset sends a subset of its step's keys, a v4 env's reset none, so the keys only a step sends are
    ///     nullable members. The scalars are <see langword="double"/>: NumPy's float64 values, the float32 control cost of a
    ///     float32 action and Python's floats and ints all widen to it exactly; the indexer still gives each value as sent.
    ///     </para>
    ///     <para>
    ///     Pusher-v4, InvertedPendulum-v4 and InvertedDoublePendulum-v4 have no record: the first refuses to build on MuJoCo 3
    ///     and later, as Gymnasium's does, and the two others always send <c>{}</c>.
    ///     </para>
    /// </remarks>
    public abstract record MujocoInfo : EnvInfo;

    /// <summary>Ant-v4's info (<see cref="AntEnvV4"/>), typed (DR-91): a step's nine keys; a v4 reset sends <c>{}</c>, so every member is nullable.</summary>
    /// <param name="RewardForward">The forward term of the reward, the weighted x velocity (<c>reward_forward</c>).</param>
    /// <param name="RewardCtrl">The control term, the negative weighted squared action, or the negative contact cost with <c>use_contact_forces</c> (<c>reward_ctrl</c>, as v4 overwrites it).</param>
    /// <param name="RewardSurvive">The healthy term (<c>reward_survive</c>), given every step in v4.</param>
    /// <param name="XPosition">The torso's x position after the step (<c>x_position</c>).</param>
    /// <param name="YPosition">The torso's y position after the step (<c>y_position</c>).</param>
    /// <param name="DistanceFromOrigin">The torso's distance from the origin in the x-y plane (<c>distance_from_origin</c>).</param>
    /// <param name="XVelocity">The torso's x velocity over the step (<c>x_velocity</c>).</param>
    /// <param name="YVelocity">The torso's y velocity over the step (<c>y_velocity</c>).</param>
    /// <param name="ForwardReward">The forward term again (<c>forward_reward</c>), the value of <paramref name="RewardForward"/>; v5 drops it.</param>
    [GymDict]
    public sealed partial record AntInfoV4(double? RewardForward, double? RewardCtrl, double? RewardSurvive, double? XPosition, double? YPosition,
                                           double? DistanceFromOrigin, double? XVelocity, double? YVelocity, double? ForwardReward) : MujocoInfo;

    /// <summary>Ant-v5's info (<see cref="AntEnvV5"/>), typed (DR-91): a reset sends the first three keys, a step all nine.</summary>
    /// <param name="XPosition">The torso's x position (<c>x_position</c>, <c>qpos[0]</c>).</param>
    /// <param name="YPosition">The torso's y position (<c>y_position</c>, <c>qpos[1]</c>).</param>
    /// <param name="DistanceFromOrigin">The torso's distance from the origin in the x-y plane (<c>distance_from_origin</c>).</param>
    /// <param name="XVelocity">The torso's x velocity over the step (<c>x_velocity</c>); <see langword="null"/> on a reset's info.</param>
    /// <param name="YVelocity">The torso's y velocity over the step (<c>y_velocity</c>); <see langword="null"/> on a reset's info.</param>
    /// <param name="RewardForward">The forward term of the reward, the weighted x velocity (<c>reward_forward</c>).</param>
    /// <param name="RewardCtrl">The control term, the negative weighted squared action, sent in the action's dtype (<c>reward_ctrl</c>).</param>
    /// <param name="RewardContact">The contact term, the negative weighted squared clipped contact forces (<c>reward_contact</c>).</param>
    /// <param name="RewardSurvive">The healthy term (<c>reward_survive</c>): the healthy reward while healthy, 0 otherwise.</param>
    [GymDict]
    public sealed partial record AntInfoV5(double XPosition, double YPosition, double DistanceFromOrigin, double? XVelocity, double? YVelocity,
                                           double? RewardForward, double? RewardCtrl, double? RewardContact, double? RewardSurvive) : MujocoInfo;

    /// <summary>HalfCheetah-v4's info (<see cref="HalfCheetahEnvV4"/>), typed (DR-91): a step's four keys; a v4 reset sends <c>{}</c>.</summary>
    /// <param name="XPosition">The front tip's x position after the step (<c>x_position</c>).</param>
    /// <param name="XVelocity">The x velocity over the step (<c>x_velocity</c>).</param>
    /// <param name="RewardRun">The forward term of the reward (<c>reward_run</c>, v5's <c>reward_forward</c>).</param>
    /// <param name="RewardCtrl">The control term, the negative weighted squared action, sent in the action's dtype (<c>reward_ctrl</c>).</param>
    [GymDict]
    public sealed partial record HalfCheetahInfoV4(double? XPosition, double? XVelocity, double? RewardRun, double? RewardCtrl) : MujocoInfo;

    /// <summary>HalfCheetah-v5's info (<see cref="HalfCheetahEnvV5"/>), typed (DR-91): a reset sends <c>x_position</c>, a step all four keys.</summary>
    /// <param name="XPosition">The root's x position (<c>x_position</c>, <c>qpos[0]</c>).</param>
    /// <param name="XVelocity">The x velocity over the step (<c>x_velocity</c>); <see langword="null"/> on a reset's info.</param>
    /// <param name="RewardForward">The forward term of the reward, the weighted x velocity (<c>reward_forward</c>).</param>
    /// <param name="RewardCtrl">The control term, the negative weighted squared action, sent in the action's dtype (<c>reward_ctrl</c>).</param>
    [GymDict]
    public sealed partial record HalfCheetahInfoV5(double XPosition, double? XVelocity, double? RewardForward, double? RewardCtrl) : MujocoInfo;

    /// <summary>Hopper-v4's info (<see cref="HopperEnvV4"/>), typed (DR-91): a step's two keys; a v4 reset sends <c>{}</c>.</summary>
    /// <param name="XPosition">The torso's x position after the step (<c>x_position</c>).</param>
    /// <param name="XVelocity">The x velocity over the step (<c>x_velocity</c>).</param>
    [GymDict]
    public sealed partial record HopperInfoV4(double? XPosition, double? XVelocity) : MujocoInfo;

    /// <summary>Hopper-v5's info (<see cref="HopperEnvV5"/>), typed (DR-91): a reset sends the first two keys, a step all six.</summary>
    /// <param name="XPosition">The torso's x position (<c>x_position</c>, <c>qpos[0]</c>).</param>
    /// <param name="ZDistanceFromOrigin">The torso's height above its initial height (<c>z_distance_from_origin</c>).</param>
    /// <param name="XVelocity">The x velocity over the step (<c>x_velocity</c>); <see langword="null"/> on a reset's info.</param>
    /// <param name="RewardForward">The forward term of the reward, the weighted x velocity (<c>reward_forward</c>).</param>
    /// <param name="RewardCtrl">The control term, the negative weighted squared action, sent in the action's dtype (<c>reward_ctrl</c>).</param>
    /// <param name="RewardSurvive">The healthy term (<c>reward_survive</c>): the healthy reward while healthy, 0 otherwise.</param>
    [GymDict]
    public sealed partial record HopperInfoV5(double XPosition, double ZDistanceFromOrigin, double? XVelocity, double? RewardForward, double? RewardCtrl,
                                              double? RewardSurvive) : MujocoInfo;

    /// <summary>Humanoid-v4's info (<see cref="HumanoidEnvV4"/>), typed (DR-91): a step's nine keys; a v4 reset sends <c>{}</c>.</summary>
    /// <param name="RewardLinvel">The forward term of the reward (<c>reward_linvel</c>, v5's <c>reward_forward</c>).</param>
    /// <param name="RewardQuadctrl">The control term, the negative weighted squared action (<c>reward_quadctrl</c>, v5's <c>reward_ctrl</c>).</param>
    /// <param name="RewardAlive">The healthy term (<c>reward_alive</c>, v5's <c>reward_survive</c>), given every step in v4.</param>
    /// <param name="XPosition">The center of mass's x position after the step (<c>x_position</c>).</param>
    /// <param name="YPosition">The center of mass's y position after the step (<c>y_position</c>).</param>
    /// <param name="DistanceFromOrigin">The center of mass's distance from the origin in the x-y plane (<c>distance_from_origin</c>).</param>
    /// <param name="XVelocity">The center of mass's x velocity over the step (<c>x_velocity</c>).</param>
    /// <param name="YVelocity">The center of mass's y velocity over the step (<c>y_velocity</c>).</param>
    /// <param name="ForwardReward">The forward term again (<c>forward_reward</c>), the value of <paramref name="RewardLinvel"/>; v5 drops it.</param>
    [GymDict]
    public sealed partial record HumanoidInfoV4(double? RewardLinvel, double? RewardQuadctrl, double? RewardAlive, double? XPosition, double? YPosition,
                                                double? DistanceFromOrigin, double? XVelocity, double? YVelocity, double? ForwardReward) : MujocoInfo;

    /// <summary>Humanoid-v5's info (<see cref="HumanoidEnvV5"/>), typed (DR-91): a reset sends the first five keys, a step all eleven.</summary>
    /// <param name="XPosition">The torso's x position (<c>x_position</c>, <c>qpos[0]</c>).</param>
    /// <param name="YPosition">The torso's y position (<c>y_position</c>, <c>qpos[1]</c>).</param>
    /// <param name="TendonLength">The lengths of the two tendons from the hips to the knees (<c>tendon_length</c>), a float64 array the step result owns.</param>
    /// <param name="TendonVelocity">The velocities of the two tendons (<c>tendon_velocity</c>), a float64 array the step result owns.</param>
    /// <param name="DistanceFromOrigin">The torso's distance from the origin in the x-y plane (<c>distance_from_origin</c>).</param>
    /// <param name="XVelocity">The center of mass's x velocity over the step (<c>x_velocity</c>); <see langword="null"/> on a reset's info.</param>
    /// <param name="YVelocity">The center of mass's y velocity over the step (<c>y_velocity</c>); <see langword="null"/> on a reset's info.</param>
    /// <param name="RewardSurvive">The healthy term (<c>reward_survive</c>): the healthy reward while healthy, 0 otherwise.</param>
    /// <param name="RewardForward">The forward term of the reward, the weighted x velocity (<c>reward_forward</c>).</param>
    /// <param name="RewardCtrl">The control term, the negative weighted squared action (<c>reward_ctrl</c>).</param>
    /// <param name="RewardContact">The contact term, the negative weighted squared clipped external contact forces (<c>reward_contact</c>).</param>
    [GymDict]
    public sealed partial record HumanoidInfoV5(double XPosition, double YPosition, NDArray TendonLength, NDArray TendonVelocity, double DistanceFromOrigin,
                                                double? XVelocity, double? YVelocity, double? RewardSurvive, double? RewardForward, double? RewardCtrl,
                                                double? RewardContact) : MujocoInfo;

    /// <summary>HumanoidStandup-v4's info (<see cref="HumanoidStandupEnvV4"/>), typed (DR-91): a step's three keys; a v4 reset sends <c>{}</c>.</summary>
    /// <param name="RewardLinup">The standing-up term, the weighted upward velocity of the torso (<c>reward_linup</c>).</param>
    /// <param name="RewardQuadctrl">The control term, the negative weighted squared action (<c>reward_quadctrl</c>).</param>
    /// <param name="RewardImpact">The impact term, the negative weighted squared clipped external contact forces (<c>reward_impact</c>).</param>
    [GymDict]
    public sealed partial record HumanoidStandupInfoV4(double? RewardLinup, double? RewardQuadctrl, double? RewardImpact) : MujocoInfo;

    /// <summary>HumanoidStandup-v5's info (<see cref="HumanoidStandupEnvV5"/>), typed (DR-91): a reset sends the first five keys, a step all eight.</summary>
    /// <param name="XPosition">The torso's x position (<c>x_position</c>, <c>qpos[0]</c>).</param>
    /// <param name="YPosition">The torso's y position (<c>y_position</c>, <c>qpos[1]</c>).</param>
    /// <param name="ZDistanceFromOrigin">The torso's height above its initial height (<c>z_distance_from_origin</c>).</param>
    /// <param name="TendonLength">The lengths of the two tendons from the hips to the knees (<c>tendon_length</c>), a float64 array the step result owns.</param>
    /// <param name="TendonVelocity">The velocities of the two tendons (<c>tendon_velocity</c>), a float64 array the step result owns.</param>
    /// <param name="RewardLinup">The standing-up term, the weighted upward velocity of the torso (<c>reward_linup</c>); <see langword="null"/> on a reset's info.</param>
    /// <param name="RewardQuadctrl">The control term, the negative weighted squared action (<c>reward_quadctrl</c>).</param>
    /// <param name="RewardImpact">The impact term, the negative weighted squared clipped external contact forces (<c>reward_impact</c>).</param>
    [GymDict]
    public sealed partial record HumanoidStandupInfoV5(double XPosition, double YPosition, double ZDistanceFromOrigin, NDArray TendonLength, NDArray TendonVelocity,
                                                       double? RewardLinup, double? RewardQuadctrl, double? RewardImpact) : MujocoInfo;

    /// <summary>InvertedDoublePendulum-v5's info (<see cref="InvertedDoublePendulumEnvV5"/>), typed (DR-91): a step's three keys; a reset sends <c>{}</c>.</summary>
    /// <param name="RewardSurvive">The healthy term (<c>reward_survive</c>), a Python float.</param>
    /// <param name="DistancePenalty">The negative penalty for the pole tip's distance from upright (<c>distance_penalty</c>).</param>
    /// <param name="VelocityPenalty">The negative penalty for the joints' velocities (<c>velocity_penalty</c>).</param>
    [GymDict]
    public sealed partial record InvertedDoublePendulumInfoV5(double? RewardSurvive, double? DistancePenalty, double? VelocityPenalty) : MujocoInfo;

    /// <summary>InvertedPendulum-v5's info (<see cref="InvertedPendulumEnvV5"/>), typed (DR-91): a step's one key; a reset sends <c>{}</c>.</summary>
    /// <param name="RewardSurvive">The reward itself (<c>reward_survive</c>), a Python int (1 while the pole stands), read as a <see langword="double"/>.</param>
    [GymDict]
    public sealed partial record InvertedPendulumInfoV5(double? RewardSurvive) : MujocoInfo;

    /// <summary>Pusher-v5's info (<see cref="PusherEnvV5"/>), typed (DR-91): a step's three keys; a reset sends <c>{}</c>.</summary>
    /// <param name="RewardDist">The distance term, the negative weighted distance of the object from the goal (<c>reward_dist</c>).</param>
    /// <param name="RewardCtrl">The control term, the negative weighted squared action, sent in the action's dtype (<c>reward_ctrl</c>).</param>
    /// <param name="RewardNear">The near term, the negative weighted distance of the fingertip from the object (<c>reward_near</c>).</param>
    [GymDict]
    public sealed partial record PusherInfoV5(double? RewardDist, double? RewardCtrl, double? RewardNear) : MujocoInfo;

    /// <summary>Reacher-v4's info (<see cref="ReacherEnvV4"/>), typed (DR-91): a step's two keys; a v4 reset sends <c>{}</c>.</summary>
    /// <param name="RewardDist">The distance term, the negative distance of the fingertip from the target (<c>reward_dist</c>).</param>
    /// <param name="RewardCtrl">The control term, the negative squared action, sent in the action's dtype (<c>reward_ctrl</c>).</param>
    [GymDict]
    public sealed partial record ReacherInfoV4(double? RewardDist, double? RewardCtrl) : MujocoInfo;

    /// <summary>Reacher-v5's info (<see cref="ReacherEnvV5"/>), typed (DR-91): a step's two keys; a reset sends <c>{}</c>.</summary>
    /// <param name="RewardDist">The distance term, the negative weighted distance of the fingertip from the target (<c>reward_dist</c>).</param>
    /// <param name="RewardCtrl">The control term, the negative weighted squared action, sent in the action's dtype (<c>reward_ctrl</c>).</param>
    [GymDict]
    public sealed partial record ReacherInfoV5(double? RewardDist, double? RewardCtrl) : MujocoInfo;

    /// <summary>Swimmer-v4's info (<see cref="SwimmerEnvV4"/>), typed (DR-91): a step's eight keys; a v4 reset sends <c>{}</c>.</summary>
    /// <param name="RewardFwd">The forward term of the reward (<c>reward_fwd</c>, v5's <c>reward_forward</c>).</param>
    /// <param name="RewardCtrl">The control term, the negative weighted squared action, sent in the action's dtype (<c>reward_ctrl</c>).</param>
    /// <param name="XPosition">The front tip's x position after the step (<c>x_position</c>).</param>
    /// <param name="YPosition">The front tip's y position after the step (<c>y_position</c>).</param>
    /// <param name="DistanceFromOrigin">The front tip's distance from the origin in the x-y plane (<c>distance_from_origin</c>).</param>
    /// <param name="XVelocity">The x velocity over the step (<c>x_velocity</c>).</param>
    /// <param name="YVelocity">The y velocity over the step (<c>y_velocity</c>).</param>
    /// <param name="ForwardReward">The forward term again (<c>forward_reward</c>), the value of <paramref name="RewardFwd"/>; v5 drops it.</param>
    [GymDict]
    public sealed partial record SwimmerInfoV4(double? RewardFwd, double? RewardCtrl, double? XPosition, double? YPosition, double? DistanceFromOrigin,
                                               double? XVelocity, double? YVelocity, double? ForwardReward) : MujocoInfo;

    /// <summary>Swimmer-v5's info (<see cref="SwimmerEnvV5"/>), typed (DR-91): a reset sends the first three keys, a step all seven.</summary>
    /// <param name="XPosition">The front tip's x position (<c>x_position</c>, <c>qpos[0]</c>).</param>
    /// <param name="YPosition">The front tip's y position (<c>y_position</c>, <c>qpos[1]</c>).</param>
    /// <param name="DistanceFromOrigin">The front tip's distance from the origin in the x-y plane (<c>distance_from_origin</c>).</param>
    /// <param name="XVelocity">The x velocity over the step (<c>x_velocity</c>); <see langword="null"/> on a reset's info.</param>
    /// <param name="YVelocity">The y velocity over the step (<c>y_velocity</c>); <see langword="null"/> on a reset's info.</param>
    /// <param name="RewardForward">The forward term of the reward, the weighted x velocity (<c>reward_forward</c>).</param>
    /// <param name="RewardCtrl">The control term, the negative weighted squared action, sent in the action's dtype (<c>reward_ctrl</c>).</param>
    [GymDict]
    public sealed partial record SwimmerInfoV5(double XPosition, double YPosition, double DistanceFromOrigin, double? XVelocity, double? YVelocity,
                                               double? RewardForward, double? RewardCtrl) : MujocoInfo;

    /// <summary>Walker2d-v4's info (<see cref="Walker2dEnvV4"/>), typed (DR-91): a step's two keys; a v4 reset sends <c>{}</c>.</summary>
    /// <param name="XPosition">The torso's x position after the step (<c>x_position</c>).</param>
    /// <param name="XVelocity">The x velocity over the step (<c>x_velocity</c>).</param>
    [GymDict]
    public sealed partial record Walker2dInfoV4(double? XPosition, double? XVelocity) : MujocoInfo;

    /// <summary>Walker2d-v5's info (<see cref="Walker2dEnvV5"/>), typed (DR-91): a reset sends the first two keys, a step all six.</summary>
    /// <param name="XPosition">The torso's x position (<c>x_position</c>, <c>qpos[0]</c>).</param>
    /// <param name="ZDistanceFromOrigin">The torso's height above its initial height (<c>z_distance_from_origin</c>).</param>
    /// <param name="XVelocity">The x velocity over the step (<c>x_velocity</c>); <see langword="null"/> on a reset's info.</param>
    /// <param name="RewardForward">The forward term of the reward, the weighted x velocity (<c>reward_forward</c>).</param>
    /// <param name="RewardCtrl">The control term, the negative weighted squared action, sent in the action's dtype (<c>reward_ctrl</c>).</param>
    /// <param name="RewardSurvive">The healthy term (<c>reward_survive</c>): the healthy reward while healthy, 0 otherwise.</param>
    [GymDict]
    public sealed partial record Walker2dInfoV5(double XPosition, double ZDistanceFromOrigin, double? XVelocity, double? RewardForward, double? RewardCtrl,
                                                double? RewardSurvive) : MujocoInfo;
}
