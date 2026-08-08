"""Compatibility shim.

The application moved to :mod:`cloudsealed_jit.api` when the project became an
installable package. This module is kept so existing deployments that run
``uvicorn api:app`` continue to work.

Prefer ``python -m cloudsealed_jit.api`` or ``uvicorn cloudsealed_jit.api:app``.
"""

from cloudsealed_jit.api import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":  # pragma: no cover
    main()
