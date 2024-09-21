import numpy as np
import taichi as ti
from taichi.ui.gui import taichi
from vec3 import vec3


@ti.dataclass
class Camera:
    center: vec3
    aspect_ratio: ti.f32
    image_width: ti.f32
    image_height: ti.i32
    focal_length: ti.f32
    viewport_height: ti.f32
    viewport_width: ti.f32
    viewport_u: vec3
    viewport_v: vec3
    pixel_delta_u: vec3
    pixel_delta_v: vec3
    viewport_bottom_left: vec3
    pixel00_loc: vec3
    res: ti.types.vector(2, ti.i32)


@ti.kernel
def init_camera(camera_center:vec3,  aspect_ratio: float, image_width: int, focal_length: float, viewport_height:  float) -> Camera:
    camera = Camera()
    camera.center = camera_center
    camera.aspect_ratio = aspect_ratio
    camera.image_width = image_width
    camera.image_height = ti.i32(image_width / aspect_ratio)
    camera.res = image_width, camera.image_height
    camera.focal_length = focal_length
    camera.viewport_height = viewport_height
    camera.viewport_width = viewport_height * (image_width)/camera.image_height
    camera.viewport_u = vec3([camera.viewport_width, 0, 0])
    camera.viewport_v = vec3([0, viewport_height, 0])
    camera.pixel_delta_u = camera.viewport_u / image_width
    camera.pixel_delta_v = camera.viewport_v / camera.image_height
    camera.viewport_bottom_left = camera_center + ti.Vector([0,0,focal_length]) - camera.viewport_u/2 - camera.viewport_v/2
    camera.pixel00_loc = camera.viewport_bottom_left + 0.5 * (camera.pixel_delta_u + camera.pixel_delta_v)
    return camera
