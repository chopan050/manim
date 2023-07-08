"""Mobjects representing function graphs."""

from __future__ import annotations

__all__ = ["ParametricFunction", "FunctionGraph", "ImplicitFunction"]


from typing import Callable, Iterable, Sequence

import numpy as np
from isosurfaces import plot_isoline

from manim import config
from manim.mobject.graphing.scale import LinearBase, _ScaleBase
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VMobject
from manim.utils.bezier import get_smooth_handle_points, interpolate
from manim.utils.color import YELLOW


class ParametricFunction(VMobject, metaclass=ConvertToOpenGL):
    """A parametric curve.

    Parameters
    ----------
    function
        The function to be plotted in the form of ``(lambda x: x**2)``. It should return
        an ndarray of coordinates which must be a valid input for :attr:`coords_to_point`,
        which should convert them into a valid ``(n, 3)``-shaped ndarray of 3D points to be
        positioned on the scene. If :attr:`coords_to_point` is not assigned a value and
        thus defaults to the identity function, then :attr:`function` must already return an
        ndarray of 3D points.
    coords_to_point
        A function (intended to be :meth:`CoordinateSystem.coords_to_point`) which converts
        coordinates into points to be positioned on scene. It should return an
        ``(n, 3)``-shaped ndarray containing 3D points. By default it is the identity function:
        ``lambda coords: coords``.
    t_range
        Determines the length that the function spans. By default: ``[0, 1]``.
    scaling
        Scaling class applied to the points of the function. Default of :class:`~.LinearBase`.
    use_smoothing
        Whether to interpolate between the points of the function after they have been created.
        (Will have odd behaviour with a low number of points)
    use_vectorized
        Whether to pass in the generated t value array to the function as ``[t_0, t_1, ...]``.
        Only use this if your function supports it. Output should be a NumPy array
        of shape ``[[x_0, x_1, ...], [y_0, y_1, ...], [z_0, z_1, ...]]`` but ``z`` can
        also be 0 if the Axes is 2D.
    discontinuities
        Values of t at which the function experiences discontinuity.
    dt
        The left and right tolerance for the discontinuities.


    Examples
    --------
    .. manim:: PlotParametricFunction
        :save_last_frame:

        class PlotParametricFunction(Scene):
            def func(self, t):
                return np.array((np.sin(2 * t), np.sin(3 * t), 0))

            def construct(self):
                func = ParametricFunction(self.func, t_range = np.array([0, TAU]), fill_opacity=0).set_color(RED)
                self.add(func.scale(3))

    .. manim:: ThreeDParametricSpring
        :save_last_frame:

        class ThreeDParametricSpring(ThreeDScene):
            def construct(self):
                curve1 = ParametricFunction(
                    lambda u: np.array([
                        1.2 * np.cos(u),
                        1.2 * np.sin(u),
                        u * 0.05
                    ]), color=RED, t_range = np.array([-3*TAU, 5*TAU, 0.01])
                ).set_shade_in_3d(True)
                axes = ThreeDAxes()
                self.add(axes, curve1)
                self.set_camera_orientation(phi=80 * DEGREES, theta=-60 * DEGREES)
                self.wait()

    .. attention::
        If your function has discontinuities, you'll have to specify the location
        of the discontinuities manually. See the following example for guidance.

    .. manim:: DiscontinuousExample
        :save_last_frame:

        class DiscontinuousExample(Scene):
            def construct(self):
                ax1 = NumberPlane((-3, 3), (-4, 4))
                ax2 = NumberPlane((-3, 3), (-4, 4))
                VGroup(ax1, ax2).arrange()
                discontinuous_function = lambda x: (x ** 2 - 2) / (x ** 2 - 4)
                incorrect = ax1.plot(discontinuous_function, color=RED)
                correct = ax2.plot(
                    discontinuous_function,
                    discontinuities=[-2, 2],  # discontinuous points
                    dt=0.1,  # left and right tolerance of discontinuity
                    color=GREEN,
                )
                self.add(ax1, ax2, incorrect, correct)
    """

    def __init__(
        self,
        function: Callable[[float, float], float],
        t_range: Sequence[float] | None = None,
        scaling: _ScaleBase = LinearBase(),
        dt: float = 1e-8,
        discontinuities: Iterable[float] | None = None,
        use_smoothing: bool = True,
        use_vectorized: bool = False,
        coords_to_point: Callable[Sequence[float], [float, float, float]] = None,
        **kwargs,
    ):
        self.function = function
        t_range = [0, 1, 0.01] if t_range is None else t_range
        if len(t_range) == 2:
            t_range = np.array([*t_range, 0.01])

        self.scaling = scaling
        self.dt = dt
        self.discontinuities = discontinuities
        self.use_smoothing = use_smoothing
        self.use_vectorized = use_vectorized
        # TODO: For some reason, directly using
        # self.coords_to_point = coords_to_point
        # increases Mobject.copy's runtime a lot: it messes up with deepcopy.
        # But why?
        if coords_to_point is None:
            self.coords_to_point = lambda *coords: coords
        else:
            self.coords_to_point = lambda *coords: coords_to_point(*coords)
        self.t_min, self.t_max, self.t_step = t_range
        self.set_t_values()

        super().__init__(**kwargs)

    def get_function(self):
        return self.function

    def get_point_from_function(self, t):
        """Evaluates :attr:`function` f(t) on a given value of t,
        and positions the obtained coordinates in the scene with
        :attr:`coords_to_point`."""
        scaled_t = self.scaling.function(np.asarray(t))
        ft = np.asarray(self.function(scaled_t))
        if ft.ndim == 1:
            point = self.coords_to_point(*ft)
        else:
            point = self.coords_to_point(ft.T)
        return point

    def set_t_values(self):
        """Calculates an array of t values to evaluate on ParametricFunction.function f(t)."""
        if self.t_min >= self.t_max:
            for attr in (
                "t_lower_bounds",
                "t_upper_bounds",
                "n_beziers_per_path",
                "acc_n_beziers",
                "scaled_t_range",
                "scaled_t_upper_bounds",
            ):
                setattr(self, attr, np.empty(0))
            self.total_n_beziers = 0
            return

        # Get t boundaries for subpaths
        if self.discontinuities is not None:
            jumps = np.asarray(self.discontinuities)
            jumps = jumps[(jumps >= self.t_min) & (jumps <= self.t_max)]
            n_jumps = jumps.size

            if n_jumps == 0:
                self.t_lower_bounds = np.array([self.t_min])
                self.t_upper_bounds = np.array([self.t_max])

            else:
                jumps = np.sort(jumps)

                self.t_lower_bounds = np.empty(n_jumps + 1)
                self.t_lower_bounds[0] = self.t_min
                self.t_lower_bounds[1:] = jumps + self.dt

                self.t_upper_bounds = np.empty(n_jumps + 1)
                self.t_upper_bounds[:-1] = jumps - self.dt
                self.t_upper_bounds[-1] = self.t_max

                # If the corresponding upper bound is less than the lower bound,
                # there cannot be a subpath between them: discard these values
                subpaths_exist = self.t_upper_bounds - self.t_lower_bounds > 1e-6
                self.t_lower_bounds = self.t_lower_bounds[subpaths_exist]
                self.t_upper_bounds = self.t_upper_bounds[subpaths_exist]
                if self.t_lower_bounds.size == 0:
                    for attr in (
                        "n_beziers_per_path",
                        "acc_n_beziers",
                        "scaled_t_range",
                        "scaled_t_upper_bounds",
                    ):
                        setattr(self, attr, np.empty(0))
                    self.total_n_beziers = 0
                    return

        else:
            self.t_lower_bounds = np.array([self.t_min])
            self.t_upper_bounds = np.array([self.t_max])

        # Convenience attributes for later use
        self.n_beziers_per_path = np.ceil(
            (self.t_upper_bounds - self.t_lower_bounds) / self.t_step
        ).astype(int)
        self.acc_n_beziers = np.add.accumulate(self.n_beziers_per_path)
        # This avoids using np.sum which is more expensive, because we need
        # the accumulated number of Bezier curves anyways
        self.total_n_beziers = self.acc_n_beziers[-1]

        # Calculate ts to be passed to self.function later
        i = 0
        self.scaled_t_range = np.empty(self.total_n_beziers)
        for lower_t, upper_t, n_beziers in zip(
            self.t_lower_bounds, self.t_upper_bounds, self.n_beziers_per_path
        ):
            self.scaled_t_range[i : i + n_beziers] = self.scaling.function(
                np.arange(lower_t, upper_t, self.t_step)
            )
            i += n_beziers

        self.scaled_t_upper_bounds = self.scaling.function(self.t_upper_bounds)

    def generate_points(self):
        if self.t_lower_bounds.size == 0:
            return

        # Calculate start and end anchors for every Bezier curve
        if self.use_vectorized:
            # ndarray.T is more efficient than np.transpose for these purposes
            start_anchors = self.function(self.scaled_t_range).T
            subpath_end_points = self.function(self.scaled_t_upper_bounds).T
        else:
            start_anchors = np.array([self.function(t) for t in self.scaled_t_range])
            subpath_end_points = [self.function(t) for t in self.scaled_t_upper_bounds]

        start_anchors = self.coords_to_point(start_anchors)
        subpath_end_points = self.coords_to_point(subpath_end_points)

        # Just in case there's a single start anchor, this
        # transforms it into an array containing the anchor
        start_anchors = start_anchors.reshape(-1, 3)
        end_anchors = np.empty((self.total_n_beziers, 3))
        end_anchors[:-1] = start_anchors[1:]
        end_anchors[self.acc_n_beziers - 1] = subpath_end_points

        # Set anchors as points in ParametricFunction
        nppcc = self.n_points_per_cubic_curve
        self.points = np.empty((nppcc * self.total_n_beziers, 3))
        self.points[::nppcc] = start_anchors
        self.points[nppcc - 1 :: nppcc] = end_anchors

        # Calculate handles and set them as points in
        # ParametricFunction
        if self.use_smoothing:
            # Smooth curve: Set handles such that the resulting curve is smooth
            # TODO: the following "to-do" might not apply anymore since
            # VMobject.make_smooth was skipped here
            # TODO: not in line with upstream, approx_smooth does not exist
            start_i = 0
            for n_beziers in self.n_beziers_per_path:
                end_i = start_i + n_beziers
                subpath_anchors = np.empty((n_beziers + 1, 3))
                subpath_anchors[:-1] = start_anchors[start_i:end_i]
                subpath_anchors[-1] = end_anchors[end_i - 1]
                subpath_handles = get_smooth_handle_points(subpath_anchors)
                for i in range(1, nppcc - 1):
                    self.points[
                        nppcc * start_i + i : nppcc * end_i + i : nppcc
                    ] = subpath_handles[i - 1]
                start_i = end_i
        else:
            # Jagged curve: Create handles which lay on the segment
            for i in range(1, nppcc - 1):
                self.points[i::nppcc] = interpolate(
                    start_anchors, end_anchors, i / (nppcc - 1)
                )

        return self

    # Alias for ParametricFunction.generate_points
    init_points = generate_points


class FunctionGraph(ParametricFunction):
    """A :class:`ParametricFunction` that spans the length of the scene by default.

    Examples
    --------
    .. manim:: ExampleFunctionGraph
        :save_last_frame:

        class ExampleFunctionGraph(Scene):
            def construct(self):
                cos_func = FunctionGraph(
                    lambda t: np.cos(t) + 0.5 * np.cos(7 * t) + (1 / 7) * np.cos(14 * t),
                    color=RED,
                )

                sin_func_1 = FunctionGraph(
                    lambda t: np.sin(t) + 0.5 * np.sin(7 * t) + (1 / 7) * np.sin(14 * t),
                    color=BLUE,
                )

                sin_func_2 = FunctionGraph(
                    lambda t: np.sin(t) + 0.5 * np.sin(7 * t) + (1 / 7) * np.sin(14 * t),
                    x_range=[-4, 4],
                    color=GREEN,
                ).move_to([0, 1, 0])

                self.add(cos_func, sin_func_1, sin_func_2)
    """

    def __init__(self, function, x_range=None, color=YELLOW, **kwargs):
        if x_range is None:
            x_range = np.array([-config["frame_x_radius"], config["frame_x_radius"]])

        self.x_range = x_range
        self.parametric_function = lambda t: np.array([t, function(t), 0])
        self.function = function
        super().__init__(self.parametric_function, self.x_range, color=color, **kwargs)

    def get_function(self):
        return self.function

    def get_point_from_function(self, x):
        return self.parametric_function(x)


class ImplicitFunction(VMobject, metaclass=ConvertToOpenGL):
    def __init__(
        self,
        func: Callable[[float, float], float],
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        min_depth: int = 5,
        max_quads: int = 1500,
        use_smoothing: bool = True,
        **kwargs,
    ):
        """An implicit function.

        Parameters
        ----------
        func
            The implicit function in the form ``f(x, y) = 0``.
        x_range
            The x min and max of the function.
        y_range
            The y min and max of the function.
        min_depth
            The minimum depth of the function to calculate.
        max_quads
            The maximum number of quads to use.
        use_smoothing
            Whether or not to smoothen the curves.
        kwargs
            Additional parameters to pass into :class:`VMobject`


        .. note::
            A small ``min_depth`` :math:`d` means that some small details might
            be ignored if they don't cross an edge of one of the
            :math:`4^d` uniform quads.

            The value of ``max_quads`` strongly corresponds to the
            quality of the curve, but a higher number of quads
            may take longer to render.

        Examples
        --------
        .. manim:: ImplicitFunctionExample
            :save_last_frame:

            class ImplicitFunctionExample(Scene):
                def construct(self):
                    graph = ImplicitFunction(
                        lambda x, y: x * y ** 2 - x ** 2 * y - 2,
                        color=YELLOW
                    )
                    self.add(NumberPlane(), graph)
        """
        self.function = func
        self.min_depth = min_depth
        self.max_quads = max_quads
        self.use_smoothing = use_smoothing
        self.x_range = x_range or [
            -config.frame_width / 2,
            config.frame_width / 2,
        ]
        self.y_range = y_range or [
            -config.frame_height / 2,
            config.frame_height / 2,
        ]

        super().__init__(**kwargs)

    def generate_points(self):
        p_min, p_max = (
            np.array([self.x_range[0], self.y_range[0]]),
            np.array([self.x_range[1], self.y_range[1]]),
        )
        curves = plot_isoline(
            fn=lambda u: self.function(u[0], u[1]),
            pmin=p_min,
            pmax=p_max,
            min_depth=self.min_depth,
            max_quads=self.max_quads,
        )  # returns a list of lists of 2D points
        curves = [
            np.pad(curve, [(0, 0), (0, 1)]) for curve in curves if curve != []
        ]  # add z coord as 0
        for curve in curves:
            self.start_new_path(curve[0])
            self.add_points_as_corners(curve[1:])
        if self.use_smoothing:
            self.make_smooth()
        return self

    init_points = generate_points
