from random import Random
import pyvista as pv # type: ignore
import numpy as np # type: ignore
import time

class Rod:
    def __init__(self, center=(0, 0, 0), length=2) -> None:
        self.center = center
        self.length = length

        self.start = np.array(center)
        self.end = np.array(center)

        self.start = self.start - np.pow(length, 1/3)
        self.end = self.end + np.pow(length, 1/3)

        line = pv.Line(self.start, self.end)

        self.vista = line.tube(radius=0.05)

    def rotate_point_around_z(self, point, angle, center=(0, 0, 0)):
        dist_from_center = np.pow(point[0] - center[0], 2)
        dist_from_center += np.pow(point[1] - center[1], 2)

        dist_from_center = np.pow(dist_from_center, 0.5)

        current_angle = np.arctan2(point[1], point[0])
        current_angle+=angle
        new_x = dist_from_center*np.cos(current_angle)
        new_y = dist_from_center*np.sin(current_angle)
        new_point = np.array([new_x, new_y, point[2]])
        return new_point
    
    def rotate_point_around_y(self, point, angle, center=(0, 0, 0)):
        dist_from_center = np.pow(point[0] - center[0], 2)
        dist_from_center += np.pow(point[2] - center[2], 2)

        dist_from_center = np.pow(dist_from_center, 0.5)

        current_angle = np.arctan2(point[2], point[0])
        current_angle+=angle
        new_x = dist_from_center*np.cos(current_angle)
        new_z = dist_from_center*np.sin(current_angle)
        new_point = np.array([new_x, point[1], new_z])
        return new_point

    def rotate_rod_z(self, angle):
        self.start = self.rotate_point_around_z(self.start, angle)
        self.end = self.rotate_point_around_z(self.end, angle)

        line = pv.Line(self.start, self.end)
        self.vista.points = line.tube(radius=0.05).points

    def rotate_rod_y(self, angle):
        self.start = self.rotate_point_around_y(self.start, angle)
        self.end = self.rotate_point_around_y(self.end, angle)

        line = pv.Line(self.start, self.end)
        self.vista.points = line.tube(radius=0.05).points

    def move_center(self, move):
        self.vista.points+=np.array(move)
        self.start+=np.array(move)
        self.end+=np.array(move)
        self.center+=np.array(move)

    def initiate_random_movement(self):
        self.next_center = np.array([0, 0, 0])
        self.next_center[0] = +3 #np.randint(-5, 5)
        self.next_center[1] = +3 #np.randint(-5, 5)
        self.next_center[2] = +3 #np.randint(-5, 5)
        self.speed = 0.1

    def move_random(self):
        diff = self.next_center - self.center
        dist = np.pow(np.pow(diff[0], 2) + np.pow(diff[1], 2) + np.pow(diff[2], 2),0.5)

        print(dist)

        to_travel = diff*0.1

        if dist <= 1:
            self.initiate_random_movement()
            return
        
        self.move_center(to_travel)


if __name__ == "__main__":
    rod = Rod(length=8, center=(1, 1, 1))

    print(rod.start)
    print(rod.end)
