import numpy as np
import cv2
import time

class PIDController:
    def __init__(self, kp, ki, kd, setpoint =0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update_steering(self, current_point):
        current_time = time.time()
        dt= current_time - self.last_time
        if dt == 0:
            return 1e-9
        self.last_time = current_time
        error = self.setpoint - current_point
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        steering = self.kp * error + self.ki * self.integral + self.kd * derivative
        return steering