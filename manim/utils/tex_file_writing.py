"""Interface for writing, compiling, and converting ``.tex`` files.

.. SEEALSO::

    :mod:`.mobject.svg.tex_mobject`

"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable

from manim.utils.tex import TexTemplate

from .. import config, logger

__all__ = ["tex_to_svg_file"]


def tex_hash(expression):
    id_str = str(expression)
    hasher = hashlib.sha256()
    hasher.update(id_str.encode())
    # Truncating at 16 bytes for cleanliness
    return hasher.hexdigest()[:16]


def tex_to_svg_file(
    expression: str,
    environment: str | None = None,
    tex_template: TexTemplate | None = None,
):
    """Takes a tex expression and returns the SVG version of the compiled TeX.

    Parameters
    ----------
    expression
        String containing the TeX expression to be rendered, e.g. ``\\sqrt{2}``
        or ``foo``.
    environment
        The string containing the environment in which the expression should be
        typeset, e.g. ``align*``.
    tex_template
        Template class used to typesetting. If not set, use the default
        template set via `config["tex_template"]`.

    Returns
    -------
    :class:`Path`
        Path to the generated SVG file.
    """
    if tex_template is None:
        tex_template = config["tex_template"]
    tex_file = generate_tex_file(expression, environment, tex_template)

    # check if svg already exists
    svg_file = tex_file.with_suffix(".svg")
    if svg_file.exists():
        return svg_file

    dvi_file = compile_tex(
        tex_file,
        tex_template.tex_compiler,
        tex_template.output_format,
    )
    svg_file = convert_to_svg(dvi_file, tex_template.output_format)
    if not config["no_latex_cleanup"]:
        delete_nonsvg_files()
    return svg_file


def generate_tex_file(
    expression: str,
    environment: str | None = None,
    tex_template: TexTemplate | None = None,
) -> Path:
    """Takes a tex expression (and an optional tex environment),
    and returns a fully formed tex file ready for compilation.

    Parameters
    ----------
    expression
        String containing the TeX expression to be rendered, e.g. ``\\sqrt{2}`` or ``foo``
    environment
        The string containing the environment in which the expression should be typeset, e.g. ``align*``
    tex_template
        Template class used to typesetting. If not set, use default template set via `config["tex_template"]`

    Returns
    -------
    :class:`Path`
        Path to generated TeX file
    """
    if tex_template is None:
        tex_template = config["tex_template"]
    if environment is not None:
        output = tex_template.get_texcode_for_expression_in_env(expression, environment)
    else:
        output = tex_template.get_texcode_for_expression(expression)

    tex_dir = config.get_dir("tex_dir")
    if not tex_dir.exists():
        tex_dir.mkdir()

    result = tex_dir / (tex_hash(output) + ".tex")
    if not result.exists():
        logger.info(
            "Writing %(expression)s to %(path)s",
            {"expression": expression, "path": f"{result}"},
        )
        result.write_text(output, encoding="utf-8")
    return result


def tex_compilation_command(
    tex_compiler: str, output_format: str, tex_file: Path, tex_dir: Path
) -> str:
    """Prepares the tex compilation command with all necessary cli flags

    Parameters
    ----------
    tex_compiler
        String containing the compiler to be used, e.g. ``pdflatex`` or ``lualatex``
    output_format
        String containing the output format generated by the compiler, e.g. ``.dvi`` or ``.pdf``
    tex_file
        File name of TeX file to be typeset.
    tex_dir
        Path to the directory where compiler output will be stored.

    Returns
    -------
    :class:`str`
        Compilation command according to given parameters
    """
    if tex_compiler in {"latex", "pdflatex", "luatex", "lualatex"}:
        commands = [
            tex_compiler,
            "-interaction=batchmode",
            f'-output-format="{output_format[1:]}"',
            "-halt-on-error",
            f'-output-directory="{tex_dir.as_posix()}"',
            f'"{tex_file.as_posix()}"',
            ">",
            os.devnull,
        ]
    elif tex_compiler == "xelatex":
        if output_format == ".xdv":
            outflag = "-no-pdf"
        elif output_format == ".pdf":
            outflag = ""
        else:
            raise ValueError("xelatex output is either pdf or xdv")
        commands = [
            "xelatex",
            outflag,
            "-interaction=batchmode",
            "-halt-on-error",
            f'-output-directory="{tex_dir.as_posix()}"',
            f'"{tex_file.as_posix()}"',
            ">",
            os.devnull,
        ]
    else:
        raise ValueError(f"Tex compiler {tex_compiler} unknown.")
    return " ".join(commands)


def insight_inputenc_error(matching):
    code_point = chr(int(matching[1], 16))
    name = unicodedata.name(code_point)
    yield f"TexTemplate does not support character '{name}' (U+{matching[1]})."
    yield "See the documentation for manim.mobject.svg.tex_mobject for details on using a custom TexTemplate."


def insight_package_not_found_error(matching):
    yield f"You do not have package {matching[1]} installed."
    yield f"Install {matching[1]} it using your LaTeX package manager, or check for typos."


def compile_tex(tex_file: Path, tex_compiler: str, output_format: str) -> Path:
    """Compiles a tex_file into a .dvi or a .xdv or a .pdf

    Parameters
    ----------
    tex_file
        File name of TeX file to be typeset.
    tex_compiler
        String containing the compiler to be used, e.g. ``pdflatex`` or ``lualatex``
    output_format
        String containing the output format generated by the compiler, e.g. ``.dvi`` or ``.pdf``

    Returns
    -------
    :class:`Path`
        Path to generated output file in desired format (DVI, XDV or PDF).
    """
    result = tex_file.with_suffix(output_format)
    tex_dir = config.get_dir("tex_dir")
    if not result.exists():
        command = tex_compilation_command(
            tex_compiler,
            output_format,
            tex_file,
            tex_dir,
        )
        exit_code = os.system(command)
        if exit_code != 0:
            log_file = tex_file.with_suffix(".log")
            print_all_tex_errors(log_file, tex_compiler, tex_file)
            raise ValueError(
                f"{tex_compiler} error converting to"
                f" {output_format[1:]}. See log output above or"
                f" the log file: {log_file}",
            )
    return result


def convert_to_svg(dvi_file: Path, extension: str, page: int = 1):
    """Converts a .dvi, .xdv, or .pdf file into an svg using dvisvgm.

    Parameters
    ----------
    dvi_file
        File name of the input file to be converted.
    extension
        String containing the file extension and thus indicating the file type, e.g. ``.dvi`` or ``.pdf``
    page
        Page to be converted if input file is multi-page.

    Returns
    -------
    :class:`Path`
        Path to generated SVG file.
    """
    result = dvi_file.with_suffix(".svg")
    if not result.exists():
        commands = [
            "dvisvgm",
            "--pdf" if extension == ".pdf" else "",
            "-p " + str(page),
            f'"{dvi_file.as_posix()}"',
            "-n",
            "-v 0",
            "-o " + f'"{result.as_posix()}"',
            ">",
            os.devnull,
        ]
        os.system(" ".join(commands))

    # if the file does not exist now, this means conversion failed
    if not result.exists():
        raise ValueError(
            f"Your installation does not support converting {dvi_file.suffix} files to SVG."
            f" Consider updating dvisvgm to at least version 2.4."
            f" If this does not solve the problem, please refer to our troubleshooting guide at:"
            f" https://docs.manim.community/en/stable/faq/general.html#my-installation-"
            f"does-not-support-converting-pdf-to-svg-help",
        )

    return result


def delete_nonsvg_files(additional_endings: Iterable[str] = ()) -> None:
    """Deletes every file that does not have a suffix in ``(".svg", ".tex", *additional_endings)``

    Parameters
    ----------
    additional_endings
        Additional endings to whitelist
    """

    tex_dir = config.get_dir("tex_dir")
    file_suffix_whitelist = {".svg", ".tex", *additional_endings}

    for f in tex_dir.iterdir():
        if f.suffix not in file_suffix_whitelist:
            f.unlink()


def print_all_tex_errors(log_file: Path, tex_compiler: str, tex_file: Path) -> None:
    if not log_file.exists():
        raise RuntimeError(
            f"{tex_compiler} failed but did not produce a log file. "
            "Check your LaTeX installation.",
        )
    with log_file.open(encoding="utf-8") as f:
        tex_compilation_log = f.readlines()
    error_indices = [
        index for index, line in enumerate(tex_compilation_log) if line.startswith("!")
    ]
    if error_indices:
        with tex_file.open() as f:
            tex = f.readlines()
        for error_index in error_indices:
            print_tex_error(tex_compilation_log, error_index, tex)


LATEX_ERROR_INSIGHTS = [
    (
        r"inputenc Error: Unicode character (?:.*) \(U\+([0-9a-fA-F]+)\)",
        insight_inputenc_error,
    ),
    (
        r"LaTeX Error: File `(.*?[clsty])' not found",
        insight_package_not_found_error,
    ),
]


def print_tex_error(tex_compilation_log, error_start_index, tex_source):
    logger.error(
        f"LaTeX compilation error: {tex_compilation_log[error_start_index][2:]}",
    )

    # TeX errors eventually contain a line beginning 'l.xxx` where xxx is the line number that caused the compilation
    # failure. This code finds the next such line after the error current error message
    line_of_tex_error = (
        int(
            [
                log_line
                for log_line in tex_compilation_log[error_start_index:]
                if log_line.startswith("l.")
            ][0]
            .split(" ")[0]
            .split(".")[1],
        )
        - 1
    )
    # our tex error may be on a line outside our user input because of post-processing
    if line_of_tex_error >= len(tex_source):
        return None

    context = ["Context of error: \n"]
    if line_of_tex_error < 3:
        context += tex_source[: line_of_tex_error + 3]
        context[-4] = "-> " + context[-4]
    elif line_of_tex_error > len(tex_source) - 3:
        context += tex_source[line_of_tex_error - 1 :]
        context[1] = "-> " + context[1]
    else:
        context += tex_source[line_of_tex_error - 3 : line_of_tex_error + 3]
        context[-4] = "-> " + context[-4]

    context = "".join(context)
    logger.error(context)

    for insights in LATEX_ERROR_INSIGHTS:
        prob, get_insight = insights
        matching = re.search(
            prob,
            "".join(tex_compilation_log[error_start_index])[2:],
        )
        if matching is not None:
            for insight in get_insight(matching):
                logger.info(insight)
