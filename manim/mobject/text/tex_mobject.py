r"""Mobjects representing text rendered using LaTeX.

.. important::

   See the corresponding tutorial :ref:`rendering-with-latex`

.. note::

   Just as you can use :class:`~.Text` (from the module :mod:`~.text_mobject`) to add text to your videos, you can use :class:`~.Tex` and :class:`~.MathTex` to insert LaTeX.

"""

from __future__ import annotations

from manim.utils.color import ManimColor

__all__ = [
    "SingleStringMathTex",
    "MathTex",
    "Tex",
    "BulletedList",
    "Title",
]


import itertools as it
import operator as op
import re
from functools import reduce
from textwrap import dedent
from typing import Iterable

from manim import config, logger
from manim.constants import *
from manim.mobject.geometry.line import Line
from manim.mobject.svg.svg_mobject import SVGMobject
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.utils.tex import TexTemplate
from manim.utils.tex_file_writing import tex_to_svg_file

tex_string_to_mob_map = {}


class SingleStringMathTex(SVGMobject):
    """Elementary building block for rendering text with LaTeX. It is the base
    used to build the much more frequently used :class:`MathTex`.

    Attributes
    ----------
    tex_string
        The string to render as LaTeX.
    stroke_width
        The width of the text border. Defaults to 0 (no border, only fill).
    should_center
        Whether this Mobject should be centered to the origin of the scene
        once rendered. Defaults to True.
    height
        The height (in Munits) to which the Mobject will rescale after the TeX
        file is compiled and imported into Manim. If None, the Mobject will not
        be rescaled, and its size will be determined by the `font_size`
        property used for compiling the TeX file. Defaults to None.
    organize_left_to_right
        If True, sort the generated submobjects by their X coordinate, from
        left to right. Defaults to False: no inner sorting would be applied.
    tex_environment
        A string indicating a TeX environment, such as the default ``align*``.
        If None, use the default tex_template indicated in the configuration.
    tex_template
        A :class:`TexTemplate` object, or None.
    font_size
        The font size used for compiling the TeX file. If height is None (no
        rescaling), it is also the font size of the final Mobject. Otherwise,
        it becomes

    Tests
    -----
    Check that creating a :class:`~.SingleStringMathTex` object works::

        >>> SingleStringMathTex('Test') # doctest: +SKIP
        SingleStringMathTex('Test')
    """

    def __init__(
        self,
        tex_string: str,
        stroke_width: float = 0,
        should_center: bool = True,
        height: float | None = None,
        organize_left_to_right: bool = False,
        tex_environment: str | None = "align*",
        tex_template: TexTemplate | None = None,
        font_size: float = DEFAULT_FONT_SIZE,
        **kwargs,
    ):
        if kwargs.get("color") is None:
            # makes it so that color isn't explicitly passed for these mobs,
            # and can instead inherit from the parent
            kwargs["color"] = VMobject().color

        self._font_size = font_size
        self.organize_left_to_right = organize_left_to_right
        self.tex_environment = tex_environment
        if tex_template is None:
            tex_template = config["tex_template"]
        self.tex_template = tex_template

        assert isinstance(tex_string, str)
        self.tex_string = tex_string
        file_name = tex_to_svg_file(
            self._get_modified_expression(tex_string),
            environment=self.tex_environment,
            tex_template=self.tex_template,
        )
        super().__init__(
            file_name=file_name,
            should_center=should_center,
            stroke_width=stroke_width,
            height=height,
            path_string_config={
                "should_subdivide_sharp_curves": True,
                "should_remove_null_curves": True,
            },
            **kwargs,
        )
        self.init_colors()

        # used for scaling via font_size.setter
        self.initial_height = self.height

        if height is None:
            self.font_size = self._font_size

        if self.organize_left_to_right:
            self._organize_submobjects_left_to_right()

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.tex_string)})"

    @property
    def font_size(self) -> float:
        """The font size of the tex mobject."""
        return self.height / self.initial_height / SCALE_FACTOR_PER_FONT_POINT

    @font_size.setter
    def font_size(self, font_val: float) -> None:
        if font_val <= 0:
            raise ValueError("font_size must be greater than 0.")
        elif self.height > 0:
            # sometimes manim generates a SingleStringMathex mobject with 0 height.
            # can't be scaled regardless and will error without the elif.

            # scale to a factor of the initial height so that setting
            # font_size does not depend on current size.
            self.scale(font_val / self.font_size)

    def _get_modified_expression(self, tex_string: str) -> str:
        result = tex_string
        result = result.strip()
        result = self._modify_special_strings(result)
        return result

    def _modify_special_strings(self, tex: str) -> str:
        tex = tex.strip()
        should_add_filler = reduce(
            op.or_,
            [
                # Fraction line needs something to be over
                tex == "\\over",
                tex == "\\overline",
                # Make sure sqrt has overbar
                tex == "\\sqrt",
                tex == "\\sqrt{",
                # Need to add blank subscript or superscript
                tex.endswith("_"),
                tex.endswith("^"),
                tex.endswith("dot"),
            ],
        )

        if should_add_filler:
            filler = "{\\quad}"
            tex += filler

        if tex == "\\substack":
            tex = "\\quad"

        if tex == "":
            tex = "\\quad"

        # To keep files from starting with a line break
        if tex.startswith("\\\\"):
            tex = tex.replace("\\\\", "\\quad\\\\")

        # Handle imbalanced \left and \right
        num_lefts, num_rights = (
            len([s for s in tex.split(substr)[1:] if s and s[0] in "(){}[]|.\\"])
            for substr in ("\\left", "\\right")
        )
        if num_lefts != num_rights:
            tex = tex.replace("\\left", "\\big")
            tex = tex.replace("\\right", "\\big")

        tex = self._remove_stray_braces(tex)

        for context in ["array"]:
            begin_in = ("\\begin{%s}" % context) in tex
            end_in = ("\\end{%s}" % context) in tex
            if begin_in ^ end_in:
                # Just turn this into a blank string,
                # which means caller should leave a
                # stray \\begin{...} with other symbols
                tex = ""
        return tex

    def _remove_stray_braces(self, tex: str) -> str:
        r"""
        Makes :class:`~.MathTex` resilient to unmatched braces.

        This is important when the braces in the TeX code are spread over
        multiple arguments as in, e.g., ``MathTex(r"e^{i", r"\tau} = 1")``.
        """

        # "\{" does not count (it's a brace literal), but "\\{" counts (it's a new line and then brace)
        num_lefts = tex.count("{") - tex.count("\\{") + tex.count("\\\\{")
        num_rights = tex.count("}") - tex.count("\\}") + tex.count("\\\\}")
        while num_rights > num_lefts:
            tex = "{" + tex
            num_lefts += 1
        while num_lefts > num_rights:
            tex = tex + "}"
            num_rights += 1
        return tex

    def _organize_submobjects_left_to_right(self) -> Self:
        self.sort(lambda p: p[0])
        return self

    def get_tex_string(self) -> str:
        return self.tex_string

    def init_colors(self, propagate_colors: bool = True) -> Self:
        if config.renderer == RendererType.OPENGL:
            super().init_colors()
        elif config.renderer == RendererType.CAIRO:
            super().init_colors(propagate_colors=propagate_colors)
        return self


class MathTex(SingleStringMathTex):
    r"""A string compiled with LaTeX in math mode.

    Examples
    --------
    .. manim:: Formula
        :save_last_frame:

        class Formula(Scene):
            def construct(self):
                t = MathTex(r"\int_a^b f'(x) dx = f(b)- f(a)")
                self.add(t)

    Tests
    -----
    Check that creating a :class:`~.MathTex` works::

        >>> MathTex('a^2 + b^2 = c^2') # doctest: +SKIP
        MathTex('a^2 + b^2 = c^2')

    Check that double brace group splitting works correctly::

        >>> t1 = MathTex('{{ a }} + {{ b }} = {{ c }}') # doctest: +SKIP
        >>> len(t1.submobjects) # doctest: +SKIP
        5
        >>> t2 = MathTex(r"\frac{1}{a+b\sqrt{2}}") # doctest: +SKIP
        >>> len(t2.submobjects) # doctest: +SKIP
        1

    """

    def __init__(
        self,
        *tex_strings: Sequence[str],
        arg_separator: str = " ",
        substrings_to_isolate: Iterable[str] | None = None,
        tex_to_color_map: dict[str | Iterable[str], ParsableManimColor] = None,
        tex_environment: str | None = "align*",
        **kwargs,
    ):
        self.tex_template: str = kwargs.pop("tex_template", config["tex_template"])
        self.arg_separator: str = arg_separator
        self.substrings_to_isolate: Iterable[str] = (
            [] if substrings_to_isolate is None else substrings_to_isolate
        )
        self.tex_to_color_map: dict[str | Iterable[str], ParsableManimColor] = (
            {} if tex_to_color_map is None else tex_to_color_map
        )
        self.tex_environment: str = tex_environment
        self.brace_notation_split_occurred: bool = False
        self.tex_strings: list[str] = self._break_up_tex_strings(tex_strings)
        try:
            super().__init__(
                self.arg_separator.join(self.tex_strings),
                tex_environment=self.tex_environment,
                tex_template=self.tex_template,
                **kwargs,
            )
            self._break_up_by_substrings()
        except ValueError as compilation_error:
            if self.brace_notation_split_occurred:
                logger.error(
                    dedent(
                        """\
                        A group of double braces, {{ ... }}, was detected in
                        your string. Manim splits TeX strings at the double
                        braces, which might have caused the current
                        compilation error. If you didn't use the double brace
                        split intentionally, add spaces between the braces to
                        avoid the automatic splitting: {{ ... }} --> { { ... } }.
                        """,
                    ),
                )
            raise compilation_error
        self.set_color_by_tex_to_color_map(self.tex_to_color_map)

        if self.organize_left_to_right:
            self._organize_submobjects_left_to_right()

    def _break_up_tex_strings(self, tex_strings: Sequence[str]) -> list[str]:
        # Separate out anything surrounded in double braces
        pre_split_length = len(tex_strings)
        tex_strings = [re.split("{{(.*?)}}", str(t)) for t in tex_strings]
        tex_strings = sum(tex_strings, [])
        if len(tex_strings) > pre_split_length:
            self.brace_notation_split_occurred = True

        # Separate out any strings specified in the isolate
        # or tex_to_color_map lists.
        patterns = [
            f"({re.escape(ss)})"
            for ss in it.chain(
                self.substrings_to_isolate,
                self.tex_to_color_map.keys(),
            )
        ]
        pattern = "|".join(patterns)
        if pattern:
            pieces = []
            for s in tex_strings:
                pieces.extend(re.split(pattern, s))
        else:
            pieces = tex_strings
        return [p for p in pieces if p]

    def _break_up_by_substrings(self) -> Self:
        """
        Reorganize existing submobjects one layer
        deeper based on the structure of tex_strings (as a list
        of tex_strings)
        """
        new_submobjects = []
        curr_index = 0
        for tex_string in self.tex_strings:
            sub_tex_mob = SingleStringMathTex(
                tex_string,
                tex_environment=self.tex_environment,
                tex_template=self.tex_template,
            )
            num_submobs = len(sub_tex_mob.submobjects)
            new_index = (
                curr_index + num_submobs + len("".join(self.arg_separator.split()))
            )
            if num_submobs == 0:
                last_submob_index = min(curr_index, len(self.submobjects) - 1)
                sub_tex_mob.move_to(self.submobjects[last_submob_index], RIGHT)
            else:
                sub_tex_mob.submobjects = self.submobjects[curr_index:new_index]
            new_submobjects.append(sub_tex_mob)
            curr_index = new_index
        self.submobjects = new_submobjects
        return self

    def get_parts_by_tex(
        self, tex: str, match_by_substring: bool = True, case_sensitive: bool = True
    ) -> VGroup:
        """Find all the submobjects matching the TeX string passed as an
        argument, and return a new VGroup containing them.

        Parameters
        ----------

        tex
            TeX string used for querying all the matching submobjects.
        match_by_substring
            If True, get all submobjects whose TeX string contains tex as a
            substring. Otherwise, only get the submobjects whose TeX is
            strictly equal to tex. Default is True.
        case_sensitive
            True if the query is case sensitive, False if not. Default is True.

        Returns
        -------
            A VGroup containing all the matches.
        """
        def has_match(query_tex: str, submob_tex: str) -> bool:
            if not case_sensitive:
                query_tex = query_tex.lower()
                submob_tex = submob_tex.lower()
            if match_by_substring:
                return query_tex in submob_tex
            else:
                return query_tex == submob_tex

        return VGroup(*(m for m in self.submobjects if has_match(tex, m.get_tex_string())))

    def get_part_by_tex(self, tex: str, **kwargs) -> VMobject | None:
        """Find and return the first submobject matching the TeX string passed
        as an argument, if it exists. If there are no matches, return None.

        Parameters
        ----------

        tex
            TeX string used for querying all the matching submobjects.
        **kwargs
            Additional parameters for the underlying call to
            :meth:`MathTex.get_parts_by_tex()`.
        
        Returns
        -------
            The first submobject matching the TeX string, or None if there are
            no matches.
        """
        all_parts = self.get_parts_by_tex(tex, **kwargs)
        return all_parts[0] if all_parts else None

    def set_color_by_tex(self, tex: str, color: ParsableManimColor, **kwargs) -> Self:
        parts_to_color = self.get_parts_by_tex(tex, **kwargs)
        for part in parts_to_color:
            part.set_color(color)
        return self

    def set_opacity_by_tex(
        self,
        tex: str,
        opacity: float = 0.5,
        remaining_opacity: float | None = None,
        **kwargs,
    ) -> Self:
        """
        Sets the opacity of the tex specified. If 'remaining_opacity' is specified,
        then the remaining tex will be set to that opacity.

        Parameters
        ----------
        tex
            The tex to set the opacity of.
        opacity
            Default 0.5. The opacity to set the tex to
        remaining_opacity
            Default None. The opacity to set the remaining tex to.
            If None, then the remaining tex will not be changed
        """
        if remaining_opacity is not None:
            self.set_opacity(opacity=remaining_opacity)
        for part in self.get_parts_by_tex(tex):
            part.set_opacity(opacity)
        return self

    def set_color_by_tex_to_color_map(
        self, texs_to_color_map: dict[str | Iterable[str], ParsableManimColor], **kwargs
    ) -> Self:
        for texs, color in list(texs_to_color_map.items()):
            try:
                # If the given key behaves like tex_strings
                texs + ""
                self.set_color_by_tex(texs, color, **kwargs)
            except TypeError:
                # If the given key is a tuple
                for tex in texs:
                    self.set_color_by_tex(tex, color, **kwargs)
        return self

    def index_of_part(self, part: VMobject) -> int:
        split_self = self.split()
        if part not in split_self:
            raise ValueError("Trying to get index of part not in MathTex")
        return split_self.index(part)

    def index_of_part_by_tex(self, tex: str, **kwargs):
        part = self.get_part_by_tex(tex, **kwargs)
        return self.index_of_part(part)

    def sort_alphabetically(self) -> Self:
        self.submobjects.sort(key=lambda m: m.get_tex_string())
        return self


class Tex(MathTex):
    r"""A string compiled with LaTeX in normal mode.

    Tests
    -----

    Check whether writing a LaTeX string works::

        >>> Tex('The horse does not eat cucumber salad.') # doctest: +SKIP
        Tex('The horse does not eat cucumber salad.')

    """

    def __init__(
        self,
        *tex_strings: Sequence[str],
        arg_separator: str = "",
        tex_environment: str | None = "center",
        **kwargs,
    ):
        super().__init__(
            *tex_strings,
            arg_separator=arg_separator,
            tex_environment=tex_environment,
            **kwargs,
        )


class BulletedList(Tex):
    """A bulleted list.

    Examples
    --------

    .. manim:: BulletedListExample
        :save_last_frame:

        class BulletedListExample(Scene):
            def construct(self):
                blist = BulletedList("Item 1", "Item 2", "Item 3", height=2, width=2)
                blist.set_color_by_tex("Item 1", RED)
                blist.set_color_by_tex("Item 2", GREEN)
                blist.set_color_by_tex("Item 3", BLUE)
                self.add(blist)
    """

    def __init__(
        self,
        *items: Sequence[str],
        buff: float = MED_LARGE_BUFF,
        dot_scale_factor: float = 2,
        tex_environment: str | None = None,
        **kwargs,
    ):
        self.buff = buff
        self.dot_scale_factor = dot_scale_factor
        self.tex_environment = tex_environment
        line_separated_items = [s + "\\\\" for s in items]
        super().__init__(
            *line_separated_items,
            tex_environment=tex_environment,
            **kwargs,
        )
        for part in self:
            dot = MathTex("\\cdot").scale(self.dot_scale_factor)
            dot.next_to(part[0], LEFT, SMALL_BUFF)
            part.add_to_back(dot)
        self.arrange(DOWN, aligned_edge=LEFT, buff=self.buff)

    def fade_all_but(self, index_or_string: int | str, opacity: float = 0.5) -> Self:
        arg = index_or_string
        if isinstance(arg, str):
            part = self.get_part_by_tex(arg)
        elif isinstance(arg, int):
            part = self.submobjects[arg]
        else:
            raise TypeError(f"Expected int or string, got {arg}")
        for other_part in self.submobjects:
            if other_part is part:
                other_part.set_fill(opacity=1)
            else:
                other_part.set_fill(opacity=opacity)
        
        return self


class Title(Tex):
    """A mobject representing an underlined title.

    Examples
    --------
    .. manim:: TitleExample
        :save_last_frame:

        import manim

        class TitleExample(Scene):
            def construct(self):
                banner = ManimBanner()
                title = Title(f"Manim version {manim.__version__}")
                self.add(banner, title)

    """

    def __init__(
        self,
        *text_parts: Sequence[str],
        include_underline: bool = True,
        match_underline_width_to_text: bool = False,
        underline_buff: float = MED_SMALL_BUFF,
        **kwargs,
    ):
        self.include_underline = include_underline
        self.match_underline_width_to_text = match_underline_width_to_text
        self.underline_buff = underline_buff
        super().__init__(*text_parts, **kwargs)
        self.to_edge(UP)
        if self.include_underline:
            underline_width = config["frame_width"] - 2
            underline = Line(LEFT, RIGHT)
            underline.next_to(self, DOWN, buff=self.underline_buff)
            if self.match_underline_width_to_text:
                underline.match_width(self)
            else:
                underline.width = underline_width
            self.add(underline)
            self.underline = underline
