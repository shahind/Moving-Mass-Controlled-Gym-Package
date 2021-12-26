"""
Classic mass-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import mmcgym
from mmcgym import spaces, logger
from mmcgym.utils import seeding
import numpy as np


class MMCBiRotor1DOFTestBedEnv(mmcgym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a stand, there is a servo
        motor which moves along a frictionless track on the pole. The pole starts
        horizontal, and the goal is to prevent it from falling over by increasing
        and reducing the mass's velocity.
    Source:
        This environment corresponds to the version of the MMC Bi-Rotor problem
        described by Shahin Darvishpoor
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Mass Position             -0.25                   0.25
        1       mass Velocity             -0.10                   0.10
        2       Pole Angle                -10  deg                10  deg
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push mass to the left
        1     Push mass to the right
        Note: The amount the velocity that is reduced or increased is fixed
        as it is a servo mechanism in fact
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.005..0.005]
    Episode Termination:
        Pole Angle is more than 20 degrees.
        mass Position is more than 0.25 (center of the mass reaches the edge of
        the pole).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.gravity = 9.8
        self.m = 0.05
        self.I = 0.000838605
        self.ycg = 0.00945
        self.l = 0.25  # actually half the pole's length
        self.v_mag = 0.1
        self.tau = 0.01  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 10 * math.pi / 180
        self.x_threshold = 0.25

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        control_command = self.v_mag if action == 1 else -self.v_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        thetaacc = (self.ycg * self.m * self.gravity * sintheta / self.I) - (x * self.m * self.gravity * costheta / self.I) 
        
        #previous states
        theta_dot_old = theta_dot
        
        #dynamics
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = control_command
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = control_command
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            if(theta*(theta_dot-theta_dot_old)<0):
                reward = 1
            else:
                reward = 0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.005, high=0.005, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        massy0 = screen_height/2  # TOP OF mass
        poleheight = 10.0
        polelen = scale * self.l
        masswidth = 20.0
        massheight = 30.0

        if self.viewer is None:
            from mmcgym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #stand
            tx, ty, bl, br = (
                screen_width / 2,
                screen_height/2 + 25,
                screen_width/2 - 75,
                screen_width / 2 + 75,
            )
            stand = rendering.FilledPolygon([(tx, ty), (bl, 0), (br, 0)])
            stand.set_color(0.6, 0.6, 0.6)
            self.viewer.add_geom(stand)
            #moving mass
            l, r, t, b = -masswidth / 2, masswidth / 2, massheight / 2, -massheight / 2
            axleoffset = massheight / 4.0
            mass = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.masstrans = rendering.Transform()
            mass.add_attr(self.masstrans)
            self.viewer.add_geom(mass)
            #pole
            l, r, t, b = (
                -polelen / 2,
                polelen / 2,
                poleheight / 2,
                -poleheight / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(poleheight / 2)
            self.axle.add_attr(self.masstrans)
            self.axle.set_color(0.8, 0.0, 0.0)
            self.axle2 = rendering.make_circle(poleheight / 4)
            self.axle2.add_attr(self.poletrans)
            self.axle2.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.viewer.add_geom(self.axle2)
            self.track = rendering.Line((0, massy0), (screen_width, massy0))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        x = self.state
        massx = x[0] * scale * math.cos(x[2]) + screen_width / 2.0  # MIDDLE OF mass
        massy = x[0] * scale * math.sin(x[2]) + screen_height / 2.0
        self.masstrans.set_translation(massx, massy)
        self.poletrans.set_translation(screen_width / 2.0, screen_height / 2.0)
        self.masstrans.set_rotation(x[2])
        self.poletrans.set_rotation(x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None