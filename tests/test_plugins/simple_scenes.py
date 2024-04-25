from __future__ import annotations

from manim import *


class SquareToCircle(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Transform(square, circle))
