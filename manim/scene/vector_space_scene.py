"""A scene suitable for vector spaces."""

from __future__ import annotations

__all__ = ["VectorScene", "LinearTransformationScene"]

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np

from manim.animation.creation import DrawBorderThenFill, Group
from manim.camera.camera import Camera
from manim.mobject.geometry.arc import Dot
from manim.mobject.geometry.line import Arrow, Line, Vector
from manim.mobject.geometry.polygram import Rectangle
from manim.mobject.graphing.coordinate_systems import Axes, NumberPlane
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.text.tex_mobject import MathTex, Tex
from manim.utils.config_ops import update_dict_recursively

from .. import config
from ..animation.animation import Animation
from ..animation.creation import Create, Write
from ..animation.fading import FadeOut
from ..animation.growing import GrowArrow
from ..animation.transform import ApplyFunction, ApplyPointwiseFunction, Transform
from ..constants import *
from ..mobject.matrix import Matrix
from ..mobject.mobject import Mobject
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..scene.scene import Scene
from ..utils.color import (
    BLACK,
    BLUE_D,
    GREEN_C,
    GREY,
    RED_C,
    WHITE,
    YELLOW,
    ManimColor,
    ParsableManimColor,
)
from ..utils.rate_functions import rush_from, rush_into
from ..utils.space_ops import angle_of_vector

if TYPE_CHECKING:
    from typing import Any

    from typing_extensions import Self

    from manim.typing import MappingFunction, Point2DLike, Point3D, Point3DLike


X_COLOR = GREEN_C
Y_COLOR = RED_C
Z_COLOR = BLUE_D


# TODO: Much of this scene type seems dependent on the coordinate system chosen.
# That is, being centered at the origin with grid units corresponding to the
# arbitrary space units.  Change it!
#
# Also, methods I would have thought of as getters, like coords_to_vector, are
# actually doing a lot of animating.
class VectorScene(Scene):
    def __init__(self, basis_vector_stroke_width: float = 6.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.basis_vector_stroke_width = basis_vector_stroke_width

    def add_plane(self, animate: bool = False, **kwargs: Any) -> NumberPlane:
        """
        Adds a NumberPlane object to the background.

        Parameters
        ----------
        animate
            Whether or not to animate the addition of the plane via Create.
        **kwargs
            Any valid keyword arguments accepted by NumberPlane.

        Returns
        -------
        NumberPlane
            The NumberPlane object.
        """
        plane = NumberPlane(**kwargs)
        if animate:
            self.play(Create(plane, lag_ratio=0.5))
        self.add(plane)
        return plane

    def add_axes(
        self,
        animate: bool = False,
        color: ParsableManimColor | Iterable[ParsableManimColor] = WHITE,
    ) -> Axes:
        """
        Adds a pair of Axes to the Scene.

        Parameters
        ----------
        animate
            Whether or not to animate the addition of the axes through Create.
        color
            The color of the axes. Defaults to WHITE.
        """
        axes = Axes(color=color, axis_config={"unit_size": 1})
        if animate:
            self.play(Create(axes))
        self.add(axes)
        return axes

    def lock_in_faded_grid(
        self, dimness: float = 0.7, axes_dimness: float = 0.5
    ) -> None:
        """
        This method freezes the NumberPlane and Axes that were already
        in the background, and adds new, manipulatable ones to the foreground.

        Parameters
        ----------
        dimness
            The required dimness of the NumberPlane

        axes_dimness
            The required dimness of the Axes.
        """
        plane = self.add_plane()
        axes = plane.get_axes()
        plane.fade(dimness)
        axes.set_color(WHITE)
        axes.fade(axes_dimness)
        self.add(axes)

        # TODO
        # error: Missing positional argument "scene" in call to "update_frame" of "CairoRenderer"  [call-arg]
        self.renderer.update_frame()  # type: ignore[call-arg]
        self.renderer.camera = Camera(self.renderer.get_frame())
        self.clear()

    def get_vector(self, numerical_vector: Point3DLike, **kwargs: Any) -> Arrow:
        """
        Returns an arrow on the Plane given an input numerical vector.

        Parameters
        ----------
        numerical_vector
            The Vector to plot.
        **kwargs
            Any valid keyword argument of Arrow.

        Returns
        -------
        Arrow
            The Arrow representing the Vector.
        """
        return Arrow(
            # TODO
            # error: "VectorScene" has no attribute "plane"  [attr-defined]
            self.plane.coords_to_point(0, 0),  # type: ignore[attr-defined]
            self.plane.coords_to_point(*numerical_vector[:2]),  # type: ignore[attr-defined]
            buff=0,
            **kwargs,
        )

    def add_vector(
        self,
        vector: Arrow | Point3DLike,
        color: ParsableManimColor | Iterable[ParsableManimColor] = YELLOW,
        animate: bool = True,
        **kwargs: Any,
    ) -> Arrow:
        """
        Returns the Vector after adding it to the Plane.

        Parameters
        ----------
        vector
            It can be a pre-made graphical vector, or the
            coordinates of one.

        color
            The string of the hex color of the vector.
            This is only taken into consideration if
            'vector' is not an Arrow. Defaults to YELLOW.

        animate
            Whether or not to animate the addition of the vector
            by using GrowArrow

        **kwargs
            Any valid keyword argument of Arrow.
            These are only considered if vector is not
            an Arrow.

        Returns
        -------
        Arrow
            The arrow representing the vector.
        """
        if not isinstance(vector, Arrow):
            vector = Vector(np.asarray(vector), color=color, **kwargs)
        if animate:
            self.play(GrowArrow(vector))
        self.add(vector)
        return vector

    def write_vector_coordinates(self, vector: Vector, **kwargs: Any) -> Matrix:
        """
        Returns a column matrix indicating the vector coordinates,
        after writing them to the screen.

        Parameters
        ----------
        vector
            The arrow representing the vector.

        **kwargs
            Any valid keyword arguments of :meth:`~.Vector.coordinate_label`:

        Returns
        -------
        :class:`.Matrix`
            The column matrix representing the vector.
        """
        coords: Matrix = vector.coordinate_label(**kwargs)
        self.play(Write(coords))
        return coords

    def get_basis_vectors(
        self,
        i_hat_color: ParsableManimColor | Iterable[ParsableManimColor] = X_COLOR,
        j_hat_color: ParsableManimColor | Iterable[ParsableManimColor] = Y_COLOR,
    ) -> VGroup:
        """
        Returns a VGroup of the Basis Vectors (1,0) and (0,1)

        Parameters
        ----------
        i_hat_color
            The hex colour to use for the basis vector in the x direction

        j_hat_color
            The hex colour to use for the basis vector in the y direction

        Returns
        -------
        VGroup
            VGroup of the Vector Mobjects representing the basis vectors.
        """
        return VGroup(
            *(
                Vector(
                    np.asarray(vect),
                    color=color,
                    stroke_width=self.basis_vector_stroke_width,
                )
                for vect, color in [([1, 0], i_hat_color), ([0, 1], j_hat_color)]
            )
        )

    def get_basis_vector_labels(self, **kwargs: Any) -> VGroup:
        """
        Returns naming labels for the basis vectors.

        Parameters
        ----------
        **kwargs
            Any valid keyword arguments of get_vector_label:
                vector,
                label (str,MathTex)
                at_tip (bool=False),
                direction (str="left"),
                rotate (bool),
                color (str),
                label_scale_factor=VECTOR_LABEL_SCALE_FACTOR (int, float),
        """
        i_hat, j_hat = self.get_basis_vectors()
        return VGroup(
            *(
                self.get_vector_label(
                    vect, label, color=color, label_scale_factor=1, **kwargs
                )
                for vect, label, color in [
                    (i_hat, "\\hat{\\imath}", X_COLOR),
                    (j_hat, "\\hat{\\jmath}", Y_COLOR),
                ]
            )
        )

    def get_vector_label(
        self,
        vector: Vector,
        label: MathTex | str,
        at_tip: bool = False,
        direction: str = "left",
        rotate: bool = False,
        color: ParsableManimColor | None = None,
        label_scale_factor: float = LARGE_BUFF - 0.2,
    ) -> MathTex:
        """
        Returns naming labels for the passed vector.

        Parameters
        ----------
        vector
            Vector Object for which to get the label.

        at_tip
            Whether or not to place the label at the tip of the vector.

        direction
            If the label should be on the "left" or right of the vector.
        rotate
            Whether or not to rotate it to align it with the vector.
        color
            The color to give the label.
        label_scale_factor
            How much to scale the label by.

        Returns
        -------
        MathTex
            The MathTex of the label.
        """
        if not isinstance(label, MathTex):
            if len(label) == 1:
                label = "\\vec{\\textbf{%s}}" % label  # noqa: UP031
            label = MathTex(label)
            if color is None:
                prepared_color: ParsableManimColor = vector.get_color()
            else:
                prepared_color = color
            label.set_color(prepared_color)
        assert isinstance(label, MathTex)
        label.scale(label_scale_factor)
        label.add_background_rectangle()

        if at_tip:
            vect = vector.get_vector()
            vect /= np.linalg.norm(vect)
            label.next_to(vector.get_end(), vect, buff=SMALL_BUFF)
        else:
            angle = vector.get_angle()
            if not rotate:
                label.rotate(-angle, about_point=ORIGIN)
            if direction == "left":
                temp_shift_1: Point3D = np.asarray(label.get_bottom())
                label.shift(-temp_shift_1 + 0.1 * UP)
            else:
                temp_shift_2: Point3D = np.asarray(label.get_top())
                label.shift(-temp_shift_2 + 0.1 * DOWN)
            label.rotate(angle, about_point=ORIGIN)
            label.shift((vector.get_end() - vector.get_start()) / 2)
        return label

    def label_vector(
        self, vector: Vector, label: MathTex | str, animate: bool = True, **kwargs: Any
    ) -> MathTex:
        """
        Shortcut method for creating, and animating the addition of
        a label for the vector.

        Parameters
        ----------
        vector
            The vector for which the label must be added.

        label
            The MathTex/string of the label.

        animate
            Whether or not to animate the labelling w/ Write

        **kwargs
            Any valid keyword argument of get_vector_label

        Returns
        -------
        :class:`~.MathTex`
            The MathTex of the label.
        """
        mathtex_label = self.get_vector_label(vector, label, **kwargs)
        if animate:
            self.play(Write(mathtex_label, run_time=1))
        self.add(mathtex_label)
        return mathtex_label

    def position_x_coordinate(
        self,
        x_coord: MathTex,
        x_line: Line,
        vector: Point3DLike,
    ) -> MathTex:  # TODO Write DocStrings for this.
        x_coord.next_to(x_line, -np.sign(vector[1]) * UP)
        x_coord.set_color(X_COLOR)
        return x_coord

    def position_y_coordinate(
        self,
        y_coord: MathTex,
        y_line: Line,
        vector: Point3DLike,
    ) -> MathTex:  # TODO Write DocStrings for this.
        y_coord.next_to(y_line, np.sign(vector[0]) * RIGHT)
        y_coord.set_color(Y_COLOR)
        return y_coord

    def coords_to_vector(
        self,
        vector: Point2DLike,
        coords_start: Point3DLike = 2 * RIGHT + 2 * UP,
        clean_up: bool = True,
    ) -> None:
        """
        This method writes the vector as a column matrix (henceforth called the label),
        takes the values in it one by one, and form the corresponding
        lines that make up the x and y components of the vector. Then, an
        Vector() based vector is created between the lines on the Screen.

        Parameters
        ----------
        vector
            The vector to show.

        coords_start
            The starting point of the location of
            the label of the vector that shows it
            numerically.
            Defaults to 2 * RIGHT + 2 * UP or (2,2)

        clean_up
            Whether or not to remove whatever
            this method did after it's done.

        """
        starting_mobjects = list(self.mobjects)
        array = Matrix(vector)
        array.shift(coords_start)
        arrow = Vector(vector)
        x_line = Line(ORIGIN, vector[0] * RIGHT)
        y_line = Line(x_line.get_end(), arrow.get_end())
        x_line.set_color(X_COLOR)
        y_line.set_color(Y_COLOR)
        mob_matrix = array.get_mob_matrix()
        x_coord = mob_matrix[0][0]
        y_coord = mob_matrix[1][0]

        self.play(Write(array, run_time=1))
        self.wait()
        self.play(
            ApplyFunction(
                lambda x: self.position_x_coordinate(x, x_line, vector),  # type: ignore[arg-type]
                x_coord,
            ),
        )
        self.play(Create(x_line))
        animations = [
            ApplyFunction(
                lambda y: self.position_y_coordinate(y, y_line, vector),  # type: ignore[arg-type]
                y_coord,
            ),
            FadeOut(array.get_brackets()),
        ]
        self.play(*animations)
        # TODO: Can we delete the line below? I don't think it have any purpose.
        # y_coord, _ = (anim.mobject for anim in animations)
        self.play(Create(y_line))
        self.play(Create(arrow))
        self.wait()
        if clean_up:
            self.clear()
            self.add(*starting_mobjects)

    def vector_to_coords(
        self,
        vector: Point3DLike,
        integer_labels: bool = True,
        clean_up: bool = True,
    ) -> tuple[Matrix, Line, Line]:
        """
        This method displays vector as a Vector() based vector, and then shows
        the corresponding lines that make up the x and y components of the vector.
        Then, a column matrix (henceforth called the label) is created near the
        head of the Vector.

        Parameters
        ----------
        vector
            The vector to show.

        integer_labels
            Whether or not to round the value displayed.
            in the vector's label to the nearest integer

        clean_up
            Whether or not to remove whatever
            this method did after it's done.

        """
        starting_mobjects = list(self.mobjects)
        show_creation = False
        if isinstance(vector, Arrow):
            arrow = vector
            vector = arrow.get_end()[:2]
        else:
            arrow = Vector(vector)
            show_creation = True
        array = arrow.coordinate_label(integer_labels=integer_labels)
        x_line = Line(ORIGIN, vector[0] * RIGHT)
        y_line = Line(x_line.get_end(), arrow.get_end())
        x_line.set_color(X_COLOR)
        y_line.set_color(Y_COLOR)
        x_coord, y_coord = cast(VGroup, array.get_entries())
        x_coord_start = self.position_x_coordinate(x_coord.copy(), x_line, vector)
        y_coord_start = self.position_y_coordinate(y_coord.copy(), y_line, vector)
        brackets = array.get_brackets()

        if show_creation:
            self.play(Create(arrow))
        self.play(Create(x_line), Write(x_coord_start), run_time=1)
        self.play(Create(y_line), Write(y_coord_start), run_time=1)
        self.wait()
        self.play(
            Transform(x_coord_start, x_coord, lag_ratio=0),
            Transform(y_coord_start, y_coord, lag_ratio=0),
            Write(brackets, run_time=1),
        )
        self.wait()

        self.remove(x_coord_start, y_coord_start, brackets)
        self.add(array)
        if clean_up:
            self.clear()
            self.add(*starting_mobjects)
        return array, x_line, y_line

    def show_ghost_movement(self, vector: Arrow | Point2DLike | Point3DLike) -> None:
        """
        This method plays an animation that partially shows the entire plane moving
        in the direction of a particular vector. This is useful when you wish to
        convey the idea of mentally moving the entire plane in a direction, without
        actually moving the plane.

        Parameters
        ----------
        vector
            The vector which indicates the direction of movement.
        """
        if isinstance(vector, Arrow):
            vector = vector.get_end() - vector.get_start()
        else:
            vector = np.asarray(vector)
            if len(vector) == 2:
                vector = np.append(np.array(vector), 0.0)
        vector_cleaned: Point3D = vector

        x_max = int(config["frame_x_radius"] + abs(vector_cleaned[0]))
        y_max = int(config["frame_y_radius"] + abs(vector_cleaned[1]))
        # TODO:
        # I think that this should be a VGroup instead of a VMobject.
        dots = VMobject(
            *(  # type: ignore[arg-type]
                Dot(x * RIGHT + y * UP)
                for x in range(-x_max, x_max)
                for y in range(-y_max, y_max)
            )
        )
        dots.set_fill(BLACK, opacity=0)
        dots_halfway = dots.copy().shift(vector_cleaned / 2).set_fill(WHITE, 1)
        dots_end = dots.copy().shift(vector_cleaned)

        self.play(Transform(dots, dots_halfway, rate_func=rush_into))
        self.play(Transform(dots, dots_end, rate_func=rush_from))
        self.remove(dots)


class LinearTransformationScene(VectorScene):
    """
    This scene contains special methods that make it
    especially suitable for showing linear transformations.

    Parameters
    ----------
    include_background_plane
        Whether or not to include the background plane in the scene.
    include_foreground_plane
        Whether or not to include the foreground plane in the scene.
    background_plane_kwargs
        Parameters to be passed to :class:`NumberPlane` to adjust the background plane.
    foreground_plane_kwargs
        Parameters to be passed to :class:`NumberPlane` to adjust the foreground plane.
    show_coordinates
        Whether or not to include the coordinates for the background plane.
    show_basis_vectors
        Whether to show the basis x_axis -> ``i_hat`` and y_axis -> ``j_hat`` vectors.
    basis_vector_stroke_width
        The ``stroke_width`` of the basis vectors.
    i_hat_color
        The color of the ``i_hat`` vector.
    j_hat_color
        The color of the ``j_hat`` vector.
    leave_ghost_vectors
        Indicates the previous position of the basis vectors following a transformation.

    Examples
    -------

    .. manim:: LinearTransformationSceneExample

        class LinearTransformationSceneExample(LinearTransformationScene):
            def __init__(self, **kwargs):
                LinearTransformationScene.__init__(
                    self,
                    show_coordinates=True,
                    leave_ghost_vectors=True,
                    **kwargs
                )

            def construct(self):
                matrix = [[1, 1], [0, 1]]
                self.apply_matrix(matrix)
                self.wait()
    """

    def __init__(
        self,
        include_background_plane: bool = True,
        include_foreground_plane: bool = True,
        background_plane_kwargs: dict[str, Any] | None = None,
        foreground_plane_kwargs: dict[str, Any] | None = None,
        show_coordinates: bool = False,
        show_basis_vectors: bool = True,
        basis_vector_stroke_width: float = 6,
        i_hat_color: ParsableManimColor = X_COLOR,
        j_hat_color: ParsableManimColor = Y_COLOR,
        leave_ghost_vectors: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.include_background_plane = include_background_plane
        self.include_foreground_plane = include_foreground_plane
        self.show_coordinates = show_coordinates
        self.show_basis_vectors = show_basis_vectors
        self.basis_vector_stroke_width = basis_vector_stroke_width
        self.i_hat_color = ManimColor(i_hat_color)
        self.j_hat_color = ManimColor(j_hat_color)
        self.leave_ghost_vectors = leave_ghost_vectors
        self.background_plane_kwargs: dict[str, Any] = {
            "color": GREY,
            "axis_config": {
                "color": GREY,
            },
            "background_line_style": {
                "stroke_color": GREY,
                "stroke_width": 1,
            },
        }

        self.ghost_vectors = VGroup()

        self.foreground_plane_kwargs: dict[str, Any] = {
            "x_range": np.array([-config["frame_width"], config["frame_width"], 1.0]),
            "y_range": np.array([-config["frame_width"], config["frame_width"], 1.0]),
            "faded_line_ratio": 1,
        }

        self.update_default_configs(
            (self.foreground_plane_kwargs, self.background_plane_kwargs),
            (foreground_plane_kwargs, background_plane_kwargs),
        )

    @staticmethod
    def update_default_configs(
        default_configs: Iterable[dict[str, Any]],
        passed_configs: Iterable[dict[str, Any] | None],
    ) -> None:
        for default_config, passed_config in zip(default_configs, passed_configs):
            if passed_config is not None:
                update_dict_recursively(default_config, passed_config)

    def setup(self) -> None:
        # The has_already_setup attr is to not break all the old Scenes
        if hasattr(self, "has_already_setup"):
            return
        self.has_already_setup = True
        self.background_mobjects: list[Mobject] = []
        self.foreground_mobjects: list[Mobject] = []
        self.transformable_mobjects: list[Mobject] = []
        self.moving_vectors: list[Mobject] = []
        self.transformable_labels: list[MathTex] = []
        self.moving_mobjects: list[Mobject] = []

        self.background_plane = NumberPlane(**self.background_plane_kwargs)

        if self.show_coordinates:
            self.background_plane.add_coordinates()
        if self.include_background_plane:
            self.add_background_mobject(self.background_plane)
        if self.include_foreground_plane:
            self.plane = NumberPlane(**self.foreground_plane_kwargs)
            self.add_transformable_mobject(self.plane)
        if self.show_basis_vectors:
            self.basis_vectors = self.get_basis_vectors(
                i_hat_color=self.i_hat_color,
                j_hat_color=self.j_hat_color,
            )
            self.moving_vectors += list(self.basis_vectors)
            self.i_hat, self.j_hat = self.basis_vectors
            self.add(self.basis_vectors)

    def add_special_mobjects(
        self, mob_list: list[Mobject], *mobs_to_add: Mobject
    ) -> None:
        """
        Adds mobjects to a separate list that can be tracked,
        if these mobjects have some extra importance.

        Parameters
        ----------
        mob_list
            The special list to which you want to add
            these mobjects.

        *mobs_to_add
            The mobjects to add.

        """
        for mobject in mobs_to_add:
            if mobject not in mob_list:
                mob_list.append(mobject)
                self.add(mobject)

    def add_background_mobject(self, *mobjects: Mobject) -> None:
        """
        Adds the mobjects to the special list
        self.background_mobjects.

        Parameters
        ----------
        *mobjects
            The mobjects to add to the list.
        """
        self.add_special_mobjects(self.background_mobjects, *mobjects)

    # TODO, this conflicts with Scene.add_foreground_mobject
    # Please be aware that there is also the method Scene.add_foreground_mobjects.
    def add_foreground_mobject(self, *mobjects: Mobject) -> None:  # type: ignore[override]
        """
        Adds the mobjects to the special list
        self.foreground_mobjects.

        Parameters
        ----------
        *mobjects
            The mobjects to add to the list
        """
        self.add_special_mobjects(self.foreground_mobjects, *mobjects)

    def add_transformable_mobject(self, *mobjects: Mobject) -> None:
        """
        Adds the mobjects to the special list
        self.transformable_mobjects.

        Parameters
        ----------
        *mobjects
            The mobjects to add to the list.
        """
        self.add_special_mobjects(self.transformable_mobjects, *mobjects)

    def add_moving_mobject(
        self, mobject: Mobject, target_mobject: Mobject | None = None
    ) -> None:
        """
        Adds the mobject to the special list
        self.moving_mobject, and adds a property
        to the mobject called mobject.target, which
        keeps track of what the mobject will move to
        or become etc.

        Parameters
        ----------
        mobject
            The mobjects to add to the list

        target_mobject
            What the moving_mobject goes to, etc.
        """
        mobject.target = target_mobject
        self.add_special_mobjects(self.moving_mobjects, mobject)

    def get_ghost_vectors(self) -> VGroup:
        """
        Returns all ghost vectors ever added to ``self``. Each element is a ``VGroup`` of
        two ghost vectors.
        """
        return self.ghost_vectors

    def get_unit_square(
        self,
        color: ParsableManimColor | Iterable[ParsableManimColor] = YELLOW,
        opacity: float = 0.3,
        stroke_width: float = 3,
    ) -> Rectangle:
        """
        Returns a unit square for the current NumberPlane.

        Parameters
        ----------
        color
            The string of the hex color code of the color wanted.

        opacity
            The opacity of the square

        stroke_width
            The stroke_width in pixels of the border of the square

        Returns
        -------
        Square
        """
        square = self.square = Rectangle(
            color=color,
            width=self.plane.get_x_unit_size(),
            height=self.plane.get_y_unit_size(),
            stroke_color=color,
            stroke_width=stroke_width,
            fill_color=color,
            fill_opacity=opacity,
        )
        square.move_to(self.plane.coords_to_point(0, 0), DL)
        return square

    def add_unit_square(self, animate: bool = False, **kwargs: Any) -> Self:
        """
        Adds a unit square to the scene via
        self.get_unit_square.

        Parameters
        ----------
        animate
            Whether or not to animate the addition
            with DrawBorderThenFill.
        **kwargs
            Any valid keyword arguments of
            self.get_unit_square()

        Returns
        -------
        Square
            The unit square.
        """
        square = self.get_unit_square(**kwargs)
        if animate:
            self.play(
                DrawBorderThenFill(square),
                Animation(Group(*self.moving_vectors)),
            )
        self.add_transformable_mobject(square)
        self.bring_to_front(*self.moving_vectors)
        self.square = square
        return self

    def add_vector(
        self,
        vector: Arrow | list | tuple | np.ndarray,
        color: ParsableManimColor = YELLOW,
        animate: bool = False,
        **kwargs: Any,
    ) -> Arrow:
        """
        Adds a vector to the scene, and puts it in the special
        list self.moving_vectors.

        Parameters
        ----------
        vector
            It can be a pre-made graphical vector, or the
            coordinates of one.

        color
            The string of the hex color of the vector.
            This is only taken into consideration if
            'vector' is not an Arrow. Defaults to YELLOW.

        **kwargs
            Any valid keyword argument of VectorScene.add_vector.

        Returns
        -------
        Arrow
            The arrow representing the vector.
        """
        vector = super().add_vector(vector, color=color, animate=animate, **kwargs)
        self.moving_vectors.append(vector)
        return vector

    def write_vector_coordinates(self, vector: Vector, **kwargs: Any) -> Matrix:
        """
        Returns a column matrix indicating the vector coordinates,
        after writing them to the screen, and adding them to the
        special list self.foreground_mobjects

        Parameters
        ----------
        vector
            The arrow representing the vector.

        **kwargs
            Any valid keyword arguments of VectorScene.write_vector_coordinates

        Returns
        -------
        Matrix
            The column matrix representing the vector.
        """
        coords = super().write_vector_coordinates(vector, **kwargs)
        self.add_foreground_mobject(coords)
        return coords

    def add_transformable_label(
        self,
        vector: Vector,
        label: MathTex | str,
        transformation_name: str | MathTex = "L",
        new_label: str | MathTex | None = None,
        **kwargs: Any,
    ) -> MathTex:
        """
        Method for creating, and animating the addition of
        a transformable label for the vector.

        Parameters
        ----------
        vector
            The vector for which the label must be added.

        label
            The MathTex/string of the label.

        transformation_name
            The name to give the transformation as a label.

        new_label
            What the label should display after a Linear Transformation

        **kwargs
            Any valid keyword argument of get_vector_label

        Returns
        -------
        :class:`~.MathTex`
            The MathTex of the label.
        """
        # TODO: Clear up types in this function. This is currently a mess.
        label_mob = self.label_vector(vector, label, **kwargs)
        if new_label:
            label_mob.target_text = new_label  # type: ignore[attr-defined]
        else:
            label_mob.target_text = (  # type: ignore[attr-defined]
                f"{transformation_name}({label_mob.get_tex_string()})"
            )
        label_mob.vector = vector  # type: ignore[attr-defined]
        label_mob.kwargs = kwargs  # type: ignore[attr-defined]
        if "animate" in label_mob.kwargs:  # type: ignore[attr-defined]
            label_mob.kwargs.pop("animate")  # type: ignore[attr-defined]
        self.transformable_labels.append(label_mob)
        return cast(MathTex, label_mob)

    def add_title(
        self,
        title: str | MathTex | Tex,
        scale_factor: float = 1.5,
        animate: bool = False,
    ) -> Self:
        """
        Adds a title, after scaling it, adding a background rectangle,
        moving it to the top and adding it to foreground_mobjects adding
        it as a local variable of self. Returns the Scene.

        Parameters
        ----------
        title
            What the title should be.

        scale_factor
            How much the title should be scaled by.

        animate
            Whether or not to animate the addition.

        Returns
        -------
        LinearTransformationScene
            The scene with the title added to it.
        """
        if not isinstance(title, (Mobject, OpenGLMobject)):
            title = Tex(title).scale(scale_factor)
        title.to_edge(UP)
        title.add_background_rectangle()
        if animate:
            self.play(Write(title))
        self.add_foreground_mobject(title)
        self.title = title
        return self

    def get_matrix_transformation(
        self, matrix: np.ndarray | list | tuple
    ) -> Callable[[Point3D], Point3D]:
        """
        Returns a function corresponding to the linear
        transformation represented by the matrix passed.

        Parameters
        ----------
        matrix
            The matrix.
        """
        return self.get_transposed_matrix_transformation(np.array(matrix).T)

    def get_transposed_matrix_transformation(
        self, transposed_matrix: np.ndarray | list | tuple
    ) -> Callable[[Point3D], Point3D]:
        """
        Returns a function corresponding to the linear
        transformation represented by the transposed
        matrix passed.

        Parameters
        ----------
        transposed_matrix
            The matrix.
        """
        transposed_matrix = np.array(transposed_matrix)
        if transposed_matrix.shape == (2, 2):
            new_matrix = np.identity(3)
            new_matrix[:2, :2] = transposed_matrix
            transposed_matrix = new_matrix
        elif transposed_matrix.shape != (3, 3):
            raise ValueError("Matrix has bad dimensions")
        return lambda point: np.dot(point, transposed_matrix)

    def get_piece_movement(self, pieces: Iterable[Mobject]) -> Transform:
        """
        This method returns an animation that moves an arbitrary
        mobject in "pieces" to its corresponding .target value.
        If self.leave_ghost_vectors is True, ghosts of the original
        positions/mobjects are left on screen

        Parameters
        ----------
        pieces
            The pieces for which the movement must be shown.

        Returns
        -------
        Animation
            The animation of the movement.
        """
        v_pieces = [piece for piece in pieces if isinstance(piece, VMobject)]
        start = VGroup(*v_pieces)
        target = VGroup(*(mob.target for mob in v_pieces))

        # don't add empty VGroups
        if self.leave_ghost_vectors and start.submobjects:
            # start.copy() gives a VGroup of Vectors
            self.ghost_vectors.add(start.copy().fade(0.7))
            self.add(self.ghost_vectors[-1])
        return Transform(start, target, lag_ratio=0)

    def get_moving_mobject_movement(self, func: MappingFunction) -> Transform:
        """
        This method returns an animation that moves a mobject
        in "self.moving_mobjects"  to its corresponding .target value.
        func is a function that determines where the .target goes.

        Parameters
        ----------

        func
            The function that determines where the .target of
            the moving mobject goes.

        Returns
        -------
        Animation
            The animation of the movement.
        """
        for m in self.moving_mobjects:
            if m.target is None:
                m.target = m.copy()
            temp: Point3D = m.get_center()
            target_point = func(temp)
            m.target.move_to(target_point)
        return self.get_piece_movement(self.moving_mobjects)

    def get_vector_movement(self, func: MappingFunction) -> Transform:
        """
        This method returns an animation that moves a mobject
        in "self.moving_vectors"  to its corresponding .target value.
        func is a function that determines where the .target goes.

        Parameters
        ----------

        func
            The function that determines where the .target of
            the moving mobject goes.

        Returns
        -------
        Animation
            The animation of the movement.
        """
        for v in self.moving_vectors:
            v.target = Vector(func(v.get_end()), color=v.get_color())
            norm = float(np.linalg.norm(v.target.get_end()))
            if norm < 0.1:
                v.target.get_tip().scale(norm)
        return self.get_piece_movement(self.moving_vectors)

    def get_transformable_label_movement(self) -> Transform:
        """
        This method returns an animation that moves all labels
        in "self.transformable_labels" to its corresponding .target .

        Returns
        -------
        Animation
            The animation of the movement.
        """
        for label in self.transformable_labels:
            # TODO: This location and lines 933 and 335 are the only locations in
            # the code where the target_text property is referenced.
            target_text: MathTex | str = label.target_text  # type: ignore[assignment]
            label.target = self.get_vector_label(
                label.vector.target,  # type: ignore[attr-defined]
                target_text,
                **label.kwargs,  # type: ignore[arg-type]
            )
        return self.get_piece_movement(self.transformable_labels)

    def apply_matrix(self, matrix: np.ndarray | list | tuple, **kwargs: Any) -> None:
        """
        Applies the transformation represented by the
        given matrix to the number plane, and each vector/similar
        mobject on it.

        Parameters
        ----------
        matrix
            The matrix.
        **kwargs
            Any valid keyword argument of self.apply_transposed_matrix()
        """
        self.apply_transposed_matrix(np.array(matrix).T, **kwargs)

    def apply_inverse(self, matrix: np.ndarray | list | tuple, **kwargs: Any) -> None:
        """
        This method applies the linear transformation
        represented by the inverse of the passed matrix
        to the number plane, and each vector/similar mobject on it.

        Parameters
        ----------
        matrix
            The matrix whose inverse is to be applied.
        **kwargs
            Any valid keyword argument of self.apply_matrix()
        """
        self.apply_matrix(np.linalg.inv(matrix), **kwargs)

    def apply_transposed_matrix(
        self, transposed_matrix: np.ndarray | list | tuple, **kwargs: Any
    ) -> None:
        """
        Applies the transformation represented by the
        given transposed matrix to the number plane,
        and each vector/similar mobject on it.

        Parameters
        ----------
        transposed_matrix
            The matrix.
        **kwargs
            Any valid keyword argument of self.apply_function()
        """
        func = self.get_transposed_matrix_transformation(transposed_matrix)
        if "path_arc" not in kwargs:
            net_rotation = np.mean(
                [angle_of_vector(func(RIGHT)), angle_of_vector(func(UP)) - np.pi / 2],
            )
            kwargs["path_arc"] = net_rotation
        self.apply_function(func, **kwargs)

    def apply_inverse_transpose(
        self, t_matrix: np.ndarray | list | tuple, **kwargs: Any
    ) -> None:
        """
        Applies the inverse of the transformation represented
        by the given transposed matrix to the number plane and each
        vector/similar mobject on it.

        Parameters
        ----------
        t_matrix
            The matrix.
        **kwargs
            Any valid keyword argument of self.apply_transposed_matrix()
        """
        t_inv = np.linalg.inv(np.array(t_matrix).T).T
        self.apply_transposed_matrix(t_inv, **kwargs)

    def apply_nonlinear_transformation(
        self, function: Callable[[np.ndarray], np.ndarray], **kwargs: Any
    ) -> None:
        """
        Applies the non-linear transformation represented
        by the given function to the number plane and each
        vector/similar mobject on it.

        Parameters
        ----------
        function
            The function.
        **kwargs
            Any valid keyword argument of self.apply_function()
        """
        self.plane.prepare_for_nonlinear_transform()
        self.apply_function(function, **kwargs)

    def apply_function(
        self,
        function: MappingFunction,
        added_anims: list[Animation] = [],
        **kwargs: Any,
    ) -> None:
        """
        Applies the given function to each of the mobjects in
        self.transformable_mobjects, and plays the animation showing
        this.

        Parameters
        ----------
        function
            The function that affects each point
            of each mobject in self.transformable_mobjects.

        added_anims
            Any other animations that need to be played
            simultaneously with this.

        **kwargs
            Any valid keyword argument of a self.play() call.
        """
        if "run_time" not in kwargs:
            kwargs["run_time"] = 3
        anims = (
            [
                ApplyPointwiseFunction(function, t_mob)  # type: ignore[arg-type]
                for t_mob in self.transformable_mobjects
            ]
            + [
                self.get_vector_movement(function),
                self.get_transformable_label_movement(),
                self.get_moving_mobject_movement(function),
            ]
            + [Animation(f_mob) for f_mob in self.foreground_mobjects]
            + added_anims
        )
        self.play(*anims, **kwargs)
