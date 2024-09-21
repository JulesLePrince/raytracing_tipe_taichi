import numpy as np
import math
import taichi as ti
from taichi.lang.ast.ast_transformer import Vector
from taichi.ui.gui import taichi

@ti.dataclass
class Ray:
    origin: ti.types.vector(3, ti.f32)
    direction: ti.types.vector(3, ti.f32)

    @ti.func
    def at(self, t:float) -> ti.types.vector(3, ti.f32):
        return self.origin + t*self.direction
