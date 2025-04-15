import logging
import os

import flask.cli
from flask import Flask, Response, abort

from .db import AppendOnlyDB
from .engine.abc import InferenceEngine
from .render import render_html

log = logging.getLogger(__name__)

flask.cli.show_server_banner = lambda *args, **kwargs: None  # Disable Flask's default banner


def proc_server(db: AppendOnlyDB, engines: dict[str, InferenceEngine], web_port: int):
    log.info("Server Thread 『Start』。 I ask of you. Are you my master?")

    app = Flask(__name__)

    @app.route("/<path:path>")
    def serve_html(path: str):  # pyright: ignore[reportUnusedFunction]
        file_path = os.path.join(os.path.dirname(__file__), "html", path)

        if not os.path.exists(file_path):
            log.warning(f"File not found: {file_path}")
            abort(404)

        try:
            with open(file_path, "r") as f:
                content = f.read()
            rendered_content = render_html(db, engines, content)
            return Response(rendered_content, mimetype="text/html")
        except Exception as e:
            log.error(f"Error reading file: {file_path} - {e}")
            abort(500)

    # TODO: Add api for PUT request

    app.run("0.0.0.0", web_port)
