import numpy as np
import math
import taichi as ti
from taichi.lang.ast.ast_transformer import Vector
from taichi.ui.gui import taichi

@ti.dataclass
class Ray:
    origin: ti.math.vec3
    direction: ti.math.vec3

    @ti.func
    def at(self, t:float) -> ti.math.vec3:
        return self.origin + t*self.direction


from taichi.ui.utils import Vector

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
viewport_v = ti.Vector([0, -viewport_height, 0])

# Calculate the horizontal and vertical delta vectors from pixel to pixel.
pixel_delta_u = viewport_u / image_width
pixel_delta_v = viewport_v / image_height

# Calculate the location of the upper left pixel
viewport_upper_left = camera_center - ti.Vector([0,0,focal_length]) - viewport_u/2 - viewport_v/2
pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)


@ti.func
def hitSphere(cent,r,ray):
    oc = cent - ray.origin
    a= ti.math.dot(ray.direction,ray.direction)
    b= -2.0 * ti.math.dot(ray.direction,oc)
    c= ti.math.dot(oc,oc) - r*r
    discrminant = b*b -4*a*c
    return ti.select(discrminant<0,-1.0,(-b - ti.sqrt(discrminant))/2*a)


@ti.func
def ray_color(ray):
    unit_direction = ray.direction
    a = 0.5*(unit_direction[1] + 1.0)
    t = hitSphere(Point([0,0,-1]),0.5,ray)
    Nv = ray.at(t)-Point([0,0,-1]) 
    N = Nv.normalized() 
    return ti.select(t>0, 0.5*Color([N.x+1,N.y+1,N.z+1]),(1.0-a)*Color([1.,1.,1.]) + a*Color([0.5,0.7,1.]))


@ti.kernel
def paint():
    for u, v in color_buffer:  # Parallelized over all pixels
        pixel_center = pixel00_loc + (u * pixel_delta_u) + (v * pixel_delta_v)
        ray_direction = pixel_center - camera_center
        r = Ray(origin=camera_center, direction=ray_direction)
        color_buffer[u,v] = ray_color(r)
        
        



gui = ti.GUI("RayTracer", res=res)
while gui.running:
    paint()
    gui.set_image(color_buffer)
    gui.show()
