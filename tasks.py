# type: ignore
import pathlib
import time

from invoke import Exit
from invoke import task

if pathlib.Path(__file__).parent != pathlib.Path().resolve():
    print("Can only run from the root directory, aborting...")
    raise SystemExit

COLOR_GRAY = 37
COLOR_GREEN = 32
COLOR_RED = 91
DEFAULT_COLOR = COLOR_GRAY

PROJECT_DIR = "birder_clip"


def echo(msg: str, color: int = DEFAULT_COLOR) -> None:
    print(f"\033[1;{color}m{msg}\033[0m")


#####################
# Linting and testing
#####################


@task
def ci(ctx, coverage=False, failfast=False):
    """
    Run all linters and tests, set return code 0 only if everything succeeded.
    """

    tic = time.time()

    return_code = 0

    if pylint(ctx) != 0:
        return_code = 1
        if failfast is True:
            echo("CI Failed", color=COLOR_RED)
            raise Exit(code=return_code)

    if sec(ctx) != 0:
        return_code = 1
        if failfast is True:
            echo("CI Failed", color=COLOR_RED)
            raise Exit(code=return_code)

    if pytest(ctx, coverage, failfast) != 0:
        return_code = 1
        if failfast is True:
            echo("CI Failed", color=COLOR_RED)
            raise Exit(code=return_code)

    echo("")
    toc = time.time()
    echo(f"CI took {(toc - tic):.1f}s")
    if return_code == 0:
        echo("CI Passed", color=COLOR_GREEN)

    else:
        echo("CI Failed", color=COLOR_RED)

    raise Exit(code=return_code)


@task
def pylint(ctx):
    """
    Run pylint & flake8 on all Python files, type check and formatting check
    """

    return_code = 0

    # pylint
    result = ctx.run(
        f"python -m pylint *.py tests {PROJECT_DIR}",
        echo=True,
        pty=True,
        warn=True,
    )
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)
    else:
        echo("Passed", color=COLOR_GREEN)

    # flake8
    result = ctx.run("python -m flake8 .", echo=True, pty=True, warn=True)
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)
    else:
        echo("Passed", color=COLOR_GREEN)

    # mypy type checking
    result = ctx.run(
        "python -m mypy --pretty --show-error-codes .",
        echo=True,
        pty=True,
        warn=True,
    )
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)
    else:
        echo("Passed", color=COLOR_GREEN)

    # Format check, black
    result = ctx.run("python -m black --check .", echo=True, pty=True, warn=True)
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)
    else:
        echo("Passed", color=COLOR_GREEN)

    # Import check, isort
    result = ctx.run("python -m isort --check-only .", echo=True, pty=True, warn=True)
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)
    else:
        echo("Passed", color=COLOR_GREEN)

    return return_code


@task
def sec(ctx):
    """
    Run security related analysis
    """

    return_code = 0

    result = ctx.run("python -m bandit -r .", echo=True, pty=True, warn=True)

    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)

    else:
        echo("Passed", color=COLOR_GREEN)

    return return_code


@task
def pytest(ctx, coverage=False, failfast=False):
    """
    Run Python tests
    """

    return_code = 0

    if coverage is True:
        result = ctx.run(
            f"python -m coverage run --source={PROJECT_DIR} -m unittest discover -s tests -v",
            echo=True,
            pty=True,
            warn=True,
        )
        ctx.run("python -m coverage report", echo=True, pty=True, warn=True)
    else:
        failfast_str = " --failfast" if failfast is True else ""
        result = ctx.run(f"python -m unittest discover -s tests -v{failfast_str}", echo=True, pty=True, warn=True)

    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)
    else:
        echo("Passed", color=COLOR_GREEN)

    return return_code
