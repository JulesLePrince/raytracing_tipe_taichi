import numpy as np
import math
import taichi as ti
from vec3 import vec3
from taichi.ui.utils import Vector
from ray import Ray
from camera import init_camera
ti.init(arch=ti.gpu)


# Image
aspect_ratio = 16.0 / 9.0;
image_width = 1080
image_height = int(image_width/aspect_ratio)
res = image_width, image_height

# Taichi
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)
Color = vec3
Point = vec3


# Camera
cam = init_camera(camera_center=Point([0.,0.,0.]),  aspect_ratio=aspect_ratio, image_width=image_width, focal_length=1.0, viewport_height=2.0)

@ti.func
def ray_color(ray):
    unit_direction = ray.direction
    a = 0.5*(unit_direction[1] + 1.0)
    return (1.0-a)*Color([1.,1.,1.]) + a*Color([0.5,0.7,1.])

@ti.kernel
def paint():
    for u, v in color_buffer:  # Parallelized over all pixels
        pixel_center = cam.pixel00_loc + (u * cam.pixel_delta_u) + (v * cam.pixel_delta_v)
        ray_direction = pixel_center - cam.center
        r = Ray(origin=cam.center, direction=ray_direction)
        color_buffer[u, v] = ray_color(r)

gui = ti.GUI("RayTracer", res=res)
while True:
    paint()
    gui.set_image(color_buffer)
    gui.show()
