# Standard Library
import os
import subprocess


def run_shell_cmd(
    cmd: list[str],
    env: dict[str, str] = os.environ.copy(),
    check: bool = True,
    capture_output: bool = False,
    stdout=None,
    stderr=None,
    stdin=None,
    cwd=None,
) -> tuple[str, str]:
    """Runs a shell command and returns stdout and stderr.

    Args:
        cmd (list[str]): List of strings representing the command to run
        env (dict[str, str]): Dictionary of environment variables to pass
        stdout (PIPE): Buffer for command stdout
        stderr (PIPE): Buffer for command stderr
        stdin (PIPE): Buffer for command stdin
        cwd (str): String representation of the command cwd

    Returns:
        tuple[str, str]: Tuple of stdout and stderr
    """

    ps = subprocess.run(
        cmd,
        check=check,
        capture_output=capture_output,
        env=env,
        stdout=stdout,
        stderr=stderr,
        stdin=stdin,
        cwd=cwd,
    )
    try:
        ps.wait()
        print(ps)
        ps_std_out, ps_std_err = ps.communicate()

    except Exception as e:
        ps_std_err = str(e)
        ps_std_out = None

    return ps_std_out, ps_std_err
