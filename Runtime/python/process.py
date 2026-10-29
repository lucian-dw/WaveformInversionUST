"""Bounded process-group lifetime; no persistent engine or scheduler."""

import os
import signal
import subprocess
import sys
import threading
import time


class ProcessTerminated(RuntimeError):
    reason = "process_termination"


def interrupted(signum, _frame):
    raise ProcessTerminated(
        f"Launcher received signal {signum}; terminating MATLAB process group"
    )


def terminate_process_group(child, grace_s=0.2):
    """Reap our child and stop descendants even if the group leader has exited."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def execute(command, timeout_s):
    if os.name != "posix":
        raise RuntimeError("Supervised execution currently requires POSIX")
    child = None
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous[signum] = signal.signal(signum, interrupted)
    try:
        child = subprocess.Popen(
            command, start_new_session=True, stdout=sys.stderr, stderr=sys.stderr
        )
        code = child.wait(timeout=timeout_s)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if child is not None:
                terminate_process_group(child)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
