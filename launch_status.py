"""Atomic startup acknowledgement shared by the bootstrap and GUI process."""
import json
import os
from pathlib import Path


STATUS_ENV = "RETENTION_STARTUP_STATUS"


def report_startup(state, **details):
    destination = os.environ.get(STATUS_ENV)
    if not destination:
        return
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(dict(state=state, pid=os.getpid(), **details)),
                         encoding="utf-8")
    temporary.replace(target)
