"""Application-wide infrastructure shared by API and service packages.

Only stable infrastructure objects are exported here. Business logic belongs
to domain packages such as :mod:`forecast` and :mod:`reports`.
"""

from core.config import settings

# Infrastructure modules are intentionally not imported eagerly here. Callers
# that only need configuration should not create a SQLAlchemy engine as a side
# effect of importing ``core``.
__all__ = ["settings"]
