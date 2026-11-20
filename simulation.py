import numpy as np
from ai import Dqn


class Simulation:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.sand = np.zeros((width, height), dtype=np.float32)

        # Goal starts near top-left 
        self.goal_x = 20.0
        self.goal_y = float(height - 20)

        # Car state
        self.car_x = width / 2.0
        self.car_y = height / 2.0
        self.angle = 0.0
        self.velocity_x = 6.0
        self.velocity_y = 0.0

        # Sensor coordinates & signals
        self.sensor1_x = self.car_x
        self.sensor1_y = self.car_y
        self.sensor2_x = self.car_x
        self.sensor2_y = self.car_y
        self.sensor3_x = self.car_x
        self.sensor3_y = self.car_y
        self.signal1 = 0.0
        self.signal2 = 0.0
        self.signal3 = 0.0

        # DQN Agent
        self.brain = Dqn(5, 3, 0.9)
        self.action2rotation = [0, 20, -20]
        self.last_reward = 0.0
        self.last_distance = np.hypot(self.car_x - self.goal_x, self.car_y - self.goal_y)

    def _rotate_vector(self, vx, vy, angle_deg):
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        return vx * c - vy * s, vx * s + vy * c

    def _move_car(self, rotation):
        self.car_x += self.velocity_x
        self.car_y += self.velocity_y
        self.angle = (self.angle + rotation) % 360

        s1_dx, s1_dy = self._rotate_vector(30.0, 0.0, self.angle)
        s2_dx, s2_dy = self._rotate_vector(30.0, 0.0, (self.angle + 30) % 360)
        s3_dx, s3_dy = self._rotate_vector(30.0, 0.0, (self.angle - 30) % 360)

        self.sensor1_x, self.sensor1_y = self.car_x + s1_dx, self.car_y + s1_dy
        self.sensor2_x, self.sensor2_y = self.car_x + s2_dx, self.car_y + s2_dy
        self.sensor3_x, self.sensor3_y = self.car_x + s3_dx, self.car_y + s3_dy

        self.signal1 = self._sense_sand(self.sensor1_x, self.sensor1_y)
        self.signal2 = self._sense_sand(self.sensor2_x, self.sensor2_y)
        self.signal3 = self._sense_sand(self.sensor3_x, self.sensor3_y)

    def _sense_sand(self, sx, sy):
        if sx > self.width - 10 or sx < 10 or sy > self.height - 10 or sy < 10:
            return 1.0
        ix, iy = int(sx), int(sy)
        patch = self.sand[max(0, ix - 10):min(self.width, ix + 10),
                          max(0, iy - 10):min(self.height, iy + 10)]
        return float(np.sum(patch)) / 400.0

    def add_sand(self, x, y, radius=10):
        ix, iy = int(x), int(y)
        x0, x1 = max(0, ix - radius), min(self.width, ix + radius)
        y0, y1 = max(0, iy - radius), min(self.height, iy + radius)
        self.sand[x0:x1, y0:y1] = 1.0

    def clear_sand(self):
        self.sand.fill(0.0)

    def reset_car(self):
        self.car_x = self.width / 2.0
        self.car_y = self.height / 2.0
        self.angle = 0.0
        self.velocity_x = 6.0
        self.velocity_y = 0.0

    def step(self):
        xx = self.goal_x - self.car_x
        yy = self.goal_y - self.car_y

        # Angle between velocity vector and goal vector normalized to [-1, 1]
        vel_angle = np.arctan2(self.velocity_y, self.velocity_x)
        goal_angle = np.arctan2(yy, xx)
        diff = (goal_angle - vel_angle + np.pi) % (2 * np.pi) - np.pi
        orientation = float(diff / np.pi)

        last_signal = [
            self.signal1,
            self.signal2,
            self.signal3,
            orientation,
            -orientation,
        ]
        action = self.brain.update(self.last_reward, last_signal)
        rotation = self.action2rotation[action]
        self._move_car(rotation)

        distance = float(np.hypot(self.car_x - self.goal_x, self.car_y - self.goal_y))

        cx = int(np.clip(self.car_x, 0, self.width - 1))
        cy = int(np.clip(self.car_y, 0, self.height - 1))

        if self.sand[cx, cy] > 0:
            self.velocity_x, self.velocity_y = self._rotate_vector(1.0, 0.0, self.angle)
            self.last_reward = -1.0
        else:
            self.velocity_x, self.velocity_y = self._rotate_vector(6.0, 0.0, self.angle)
            self.last_reward = -0.2
            if distance < self.last_distance:
                self.last_reward = 0.1

        # Wall boundary checks
        if self.car_x < 10:
            self.car_x = 10.0
            self.last_reward = -1.0
        if self.car_x > self.width - 10:
            self.car_x = float(self.width - 10)
            self.last_reward = -1.0
        if self.car_y < 10:
            self.car_y = 10.0
            self.last_reward = -1.0
        if self.car_y > self.height - 10:
            self.car_y = float(self.height - 10)
            self.last_reward = -1.0

        # Goal switching
        if distance < 100:
            self.goal_x = float(self.width - self.goal_x)
            self.goal_y = float(self.height - self.goal_y)

        self.last_distance = distance

        return {
            "car": {"x": self.car_x, "y": self.car_y, "angle": self.angle},
            "sensors": [
                {"x": self.sensor1_x, "y": self.sensor1_y},
                {"x": self.sensor2_x, "y": self.sensor2_y},
                {"x": self.sensor3_x, "y": self.sensor3_y},
            ],
            "goal": {"x": self.goal_x, "y": self.goal_y},
            "score": round(self.brain.score(), 4),
            "speed": round(float(np.hypot(self.velocity_x, self.velocity_y)), 1),
        }
