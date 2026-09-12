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
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                else:
                    time.sleep(0.2)
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                child.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
