import gymnasium as gym
from gymnasium.envs import classic_control
import pygame
from pygame import gfxdraw
from gymnasium.envs.classic_control import utils
import numpy as np
from gymnasium.vector.utils import batch_space
from gymnasium import logger, spaces

class ModifiedCartPoleVectorEnv(classic_control.cartpole.CartPoleVectorEnv):
    def __init__(self, *args, **kwargs):
        self.clock = pygame.time.Clock()
        self.last_reward = None
        self.screen = None
        super().__init__(*args, **kwargs)
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                1,
                np.inf,
                1
            ],
            dtype=np.float32,
        )
        self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)


    def step(
            self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.state
        force = np.sign(action - 0.5) * self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
                       force + self.polemass_length * np.square(theta_dot) * sintheta
               ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length
                * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.stack((x, x_dot, theta, theta_dot))

        terminated: np.ndarray = (
                (x < -self.x_threshold)
                | (x > self.x_threshold)
        )

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        if self._sutton_barto_reward:
            reward = -np.array(terminated, dtype=np.float32)
        else:
            reward = np.cos(theta) > np.cos(12 / 180 * np.pi)
            reward = reward.astype(np.float32)
            reward[terminated] = -100
        self.last_reward = reward

        # Reset all environments which terminated or were truncated in the last step
        self.state[:, self.prev_done] = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.prev_done.sum())
        )
        self.state[2, self.prev_done] += np.pi
        self.steps[self.prev_done] = 0
        reward[self.prev_done] = 0.0
        terminated[self.prev_done] = False
        truncated[self.prev_done] = False

        self.prev_done = np.logical_or(terminated, truncated)
        cos_vals = np.cos(self.state[None, 2, :])
        state2 = np.concatenate((self.state, cos_vals), axis=0)
        state2[2, :] = np.sin(self.state[2, :])

        return state2.T.astype(np.float32), reward, terminated, truncated, {}
    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # -0.05 and 0.05 is the default low and high bounds
        self.low, self.high = utils.maybe_parse_reset_bounds(options, -0.2, 0.2)
        self.state = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.num_envs)
        )
        self.state[2, :] += np.pi
        self.steps_beyond_terminated = None
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_done = np.zeros(self.num_envs, dtype=np.bool_)

        cos_vals = np.cos(self.state[None, 2, :])
        state2 = np.concatenate((self.state, cos_vals), axis=0)
        state2[2, :] = np.sin(self.state[2, :])

        # if self.render_mode == "human":
        #    self.render()
        return state2.T.astype(np.float32), {}
    def render(self, epoch, order):
        if self.screen is None:
            pygame.font.init()
            pygame.display.init()



            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
        my_font = pygame.font.SysFont('arial', 30)
        self.text_surface = my_font.render('Generation: ' + str(epoch), True, (0, 0, 0))
        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            raise ValueError(
                "Cartpole's state is None, it probably hasn't be reset yet."
            )
        bgColor = np.array([200, 220, 230])
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill(bgColor)

        for idx, o in enumerate(order[-20:]):

            x = self.state.T[o]
            if (idx == len(order[-20:]) - 1):
                percentage = 1
            else:
                percentage = (((idx + 1) / len(order[-20:]))) ** 4
                percentage *= 0.6


            poleColor = percentage * np.array([250, 200, 90]) + (1 - percentage) * bgColor
            polePointColor = percentage * np.array([245, 190, 185]) + (1 - percentage) * bgColor
            railColor = np.array([195, 180, 110])
            circleColor = percentage * np.array([140, 160, 130]) + (1 - percentage) * bgColor
            cartColor = percentage * np.array([75, 75, 75]) + (1 - percentage) * bgColor
            wheelColor = percentage * np.array([0, 0, 0]) + (1 - percentage) * bgColor

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
            carty = 100+50  # TOP OF CART
            cart_coords = [(l, b), (l, t), (r, t), (r, b)]
            cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
            l, r, t, b = -cartwidth / 2-2, cartwidth / 2+2, cartheight / 2+2, -cartheight / 2-2
            cart_coords2 = [(l, b), (l, t), (r, t), (r, b)]
            cart_coords2 = [(c[0] + cartx, c[1] + carty) for c in cart_coords2]

            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            l2, r2, t2, b2 = (
                -polewidth / 2 - 2,
                polewidth / 2 + 2,
                polelen - polewidth / 2 + 2,
                -polewidth / 2 - 2,
            )

            pole_coords = []
            for coord in [(l, b), (l, t), (r, t), (r, b)]:
                coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
                coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
                pole_coords.append(coord)
            pole_coords2 = []
            for coord in [(l2, b2), (l2, t2), (r2, t2), (r2, b2)]:
                coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
                coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
                pole_coords2.append(coord)
            color = poleColor if np.cos(x[2]) < np.cos(12 / 180 * np.pi) else polePointColor
            gfxdraw.aapolygon(self.surf, pole_coords2, wheelColor)
            gfxdraw.filled_polygon(self.surf, pole_coords2, wheelColor)
            gfxdraw.aapolygon(self.surf, pole_coords, color)
            gfxdraw.filled_polygon(self.surf, pole_coords, color)

            gfxdraw.aacircle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(17),
                wheelColor,
            )
            gfxdraw.filled_circle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(17),
                wheelColor,
            )
            gfxdraw.aacircle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(15),
                circleColor,
            )
            gfxdraw.filled_circle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(15),
                circleColor,
            )

            gfxdraw.aapolygon(self.surf, cart_coords2, wheelColor)
            gfxdraw.filled_polygon(self.surf, cart_coords2, wheelColor)
            gfxdraw.aapolygon(self.surf, cart_coords, cartColor)
            gfxdraw.filled_polygon(self.surf, cart_coords, cartColor)

            gfxdraw.aacircle(
                self.surf,
                int(cartx+ cartwidth/4),
                int(carty - cartheight/2),
                int(7),
                wheelColor,
            )
            gfxdraw.filled_circle(
                self.surf,
                int(cartx+ cartwidth/4),
                int(carty - cartheight/2),
                int(7),
                wheelColor,
            )

            gfxdraw.aacircle(
                self.surf,
                int(cartx - cartwidth / 4),
                int(carty - cartheight / 2),
                int(7),
                wheelColor,
            )
            gfxdraw.filled_circle(
                self.surf,
                int(cartx - cartwidth / 4),
                int(carty - cartheight / 2),
                int(7),
                wheelColor,
            )



            gfxdraw.box(self.surf, pygame.Rect(0,int(carty-cartheight/2-7-9), self.screen_width, 9), (0, 0, 0))
            gfxdraw.box(self.surf, pygame.Rect(0, int(carty - cartheight / 2 - 7 - 7), self.screen_width, 5), railColor)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        self.screen.blit(self.text_surface, (20, 10))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(60)
            pygame.display.flip()
        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )