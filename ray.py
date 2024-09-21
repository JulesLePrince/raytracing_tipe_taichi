import numpy as np
import taichi as ti
from taichi.ui.gui import taichi
from vec3 import vec3


@ti.dataclass
class Ray:
    origin: vec3
    direction: vec3

    @ti.func
    def at(self, t:float) -> vec3:
        return self.origin + t*self.direction
