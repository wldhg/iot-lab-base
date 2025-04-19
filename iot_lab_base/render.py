import logging
import re
from datetime import datetime

from .db import AppendOnlyDB
from .engine.abc import InferenceEngine

log = logging.getLogger(__name__)


def render_html(db: AppendOnlyDB, engines: dict[str, InferenceEngine], orig_html: str) -> str:
    """
    Render the HTML template with the given database and engines.
    Should replace all "{{...}}" placeholders in the HTML with the corresponding values.
    """

    new_html = orig_html

    find_all_placeholders: list[str] = re.findall(r"\{\{(.*?)\}\}", orig_html, re.DOTALL)
    for placeholder in find_all_placeholders:
        try:
            # TODO: Implement the logic to replace the placeholder with actual values
            raise NotImplementedError(f"Placeholder {{ {placeholder} }} is not implemented yet.")
        except Exception as e:
            log.error(f"Error rendering placeholder {{ {placeholder} }}: {e}")
            continue

    return new_html
