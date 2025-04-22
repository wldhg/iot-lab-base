import importlib
import logging
import re
from datetime import datetime

from . import render_button, render_checkbox
from .db import AppendOnlyDB
from .engine.abc import InferenceEngine

log = logging.getLogger(__name__)


def render_html(db: AppendOnlyDB, engines: dict[str, InferenceEngine], orig_html: str) -> str:
    """
    Render the HTML template with the given database and engines.
    Should replace all "{{...}}" placeholders in the HTML with the corresponding values.
    """

    importlib.reload(render_button)
    importlib.reload(render_checkbox)

    new_html = orig_html

    find_all_placeholders: list[str] = re.findall(r"\{\{(.*?)\}\}", orig_html, re.DOTALL)
    for placeholder in find_all_placeholders:
        try:
            replaced_value: str | None = None

            if placeholder.startswith("value:") or placeholder.startswith("ts:"):
                selector, db_item_name = placeholder.split(":", 1)
                db_item = db.get(db_item_name)
                assert db_item is not None, f"{db_item_name} not found in database."
                if selector == "value":
                    replaced_value = str(db_item.value)
                else:
                    replaced_value = str(datetime.fromtimestamp(db_item.ts))
            elif placeholder.startswith("eval:"):
                eval_expr = placeholder.split(":", 1)[1].replace("\n", "")
                replaced_value = str(eval(eval_expr, {"db": db, "engines": engines}))
            elif placeholder.startswith("button:"):
                replaced_value = render_button.render_button(placeholder)
            elif placeholder.startswith("chkbox:"):
                replaced_value = render_checkbox.render_checkbox(db, placeholder)
            elif placeholder.startswith("infer:"):
                _, model_name = placeholder.split(":", 1)
                replaced_value = str(engines[model_name].infer())

            assert replaced_value is not None, f"Value for placeholder {{ {placeholder} }} is None."
            new_html = new_html.replace("{{" + placeholder + "}}", replaced_value)
        except Exception as e:
            log.error(f"Error rendering placeholder {{ {placeholder} }}: {e}")
            continue

    return new_html
