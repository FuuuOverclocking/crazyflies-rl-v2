import gym
from gym import spaces
import math
import numpy as np

TIME_STEP = 1. / 50.


class BaseWorldObject(object):
    def __init__(self, x=0., y=0., vx=0., vy=0., a=1.):
        self.updateXY(x, y)
        self.updateV(vx, vy)
        self.updateTargetV(vx, vy)
        self.a = a

    def updateXY(self, x, y):
        self.x = x
        self.y = y

    def updateV(self, vx, vy):
        self.vx = vx
        self.vy = vy

    def updateTargetV(self, tvx, tvy):
        self.tvx = tvx
        self.tvy = tvy


def calc_shift(v0, target_v, a_abs):
    is_always_accelerating_during_timestep = True
    is_a_positive = target_v > v0

    a = a_abs if is_a_positive else -a_abs

    v1 = v0 + TIME_STEP * a

    if (is_a_positive and target_v < v1) or (not is_a_positive and target_v > v1):
        v1 = target_v
        is_always_accelerating_during_timestep = False

    delta = (v1 * v1 - v0 * v0) / (2 * a)

    if not is_always_accelerating_during_timestep:
        delta += (TIME_STEP - (v1 - v0) / a) * v1

    return (delta, v1)


class BaseWorld(object):
    def __init__(self):
        self.objects = []
        self.time = 0.

    def add_object(self, obj: BaseWorldObject):
        self.objects.append(obj)

    def step(self):
        self.time += TIME_STEP
        for obj in self.objects:
            if obj.a < 0.01:
                obj.updateXY(obj.x + obj.vx * TIME_STEP,
                             obj.y + obj.vy * TIME_STEP)
                continue

            if abs(obj.vx - obj.tvx) < 0.001:
                vx1 = obj.tvx
                delta_x = vx1 * TIME_STEP
            else:
                delta_x, vx1 = calc_shift(obj.vx, obj.tvx, obj.a)

            if abs(obj.vy - obj.tvy) < 0.001:
                vy1 = obj.tvy
                delta_y = vy1 * TIME_STEP
            else:
                delta_y, vy1 = calc_shift(obj.vy, obj.tvy, obj.a)

            obj.updateXY(obj.x + delta_x, obj.y + delta_y)
            obj.updateV(vx1, vy1)


class CFWorldEnv(gym.Env):
    '''
    二维伪连续世界
    '''
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        world = self.world = BaseWorld()
        uav = self.uav = BaseWorldObject(
            x=0.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            a=6.0
        )
        source = self.source = BaseWorldObject(
            x=16.0,
            y=16.0,
            vx=0.0,
            vy=0.0,
            a=0.0
        )
        world.add_object(uav)
        world.add_object(source)

        # specify (target v_x, target v_y) of UAV
        self.action_space = spaces.Box(
            low=-50.0,
            high=50.0,
            shape=(2,)
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=100.0,
            shape=(1,)
        )

    def reset(self):
        self.uav.updateXY(0, 0)
        self.uav.updateV(0, 0)
        self.uav.updateTargetV(0, 0)
        self.source.updateXY(16.0, 16.0)
        self.source.updateV(0, 0)
        return [math.sqrt(16 * 16 * 2)]

    # As TIME_STEP = `0.02s` and DISTANCE_BUFFER_MAXLEN = `4`,
    # the real distance doesn't takes effect on the observed
    # distance until `80ms` later.
    distance_buffer = []
    DISTANCE_BUFFER_MAXLEN = 4

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        self.uav.updateTargetV(action[0], action[1])
        self.world.step()

        distance = (self.source.x - self.uav.x) ** 2 + \
            (self.source.y - self.uav.y) ** 2
        distance = math.sqrt(distance)

        if len(self.distance_buffer) == self.DISTANCE_BUFFER_MAXLEN:
            self.distance_buffer.pop()

        self.distance_buffer.insert(0, distance)

        for i in range(1, len(self.distance_buffer)):
            distance += self.distance_buffer[i]

        distance /= len(self.distance_buffer)

        info = {
            'time': self.world.time,
            'uav': self.uav,
            'source': self.source,
        }

        return [distance], 1.0, (distance < 1.0), info

    viewer = None

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        def cord_world_to_screen(x: float, y: float):
            return x * 25 + 100, y * 25 + 100

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # TODO: 绘图

        return None

        # TODO: 绘图

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
