import numpy as np
import math
import taichi as ti
from taichi.ui.utils import Vector
from ray import Ray
ti.init(arch=ti.gpu)


# Image
aspect_ratio = 16.0 / 9.0;
image_width = 1080
image_height = int(image_width/aspect_ratio)
res = image_width, image_height

# Taichi
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)
Color = ti.math.vec3
Point = ti.math.vec3

# Camera
focal_length = 1.0
viewport_height = 2.0
viewport_width = viewport_height * (image_width)/image_height
camera_center = Point([0,0,0])


# Calculate the vectors across the horizontal and down the vertical viewport edges
viewport_u = ti.Vector([viewport_width, 0, 0])
viewport_v = ti.Vector([0, viewport_height, 0])

# Calculate the horizontal and vertical delta vectors from pixel to pixel.
pixel_delta_u = viewport_u / image_width
pixel_delta_v = viewport_v / image_height

# Calculate the location of the upper left pixel
viewport_upper_left = camera_center - ti.Vector([0,0,focal_length]) - viewport_u/2 - viewport_v/2
pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

@ti.func
def ray_color(ray):
    unit_direction = ray.direction
    a = 0.5*(unit_direction[1] + 1.0)
    return (1.0-a)*Color([1.,1.,1.]) + a*Color([0.5,0.7,1.])


@ti.kernel
def paint():
    for u, v in color_buffer:  # Parallelized over all pixels
        pixel_center = pixel00_loc + (u * pixel_delta_u) + (v * pixel_delta_v)
        ray_direction = pixel_center - camera_center
        r = Ray(origin=camera_center, direction=ray_direction)
        color_buffer[u, v] = ray_color(r)

gui = ti.GUI("RayTracer", res=res)
while True:
    paint()
    gui.set_image(color_buffer)
    gui.show()
