import importlib
import logging
import os

import flask.cli
from flask import Flask, Response, abort, request

from . import render
from .db import AppendOnlyDB
from .engine.abc import InferenceEngine

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
            importlib.reload(render)
            with open(file_path, "r") as f:
                content = f.read()
            rendered_content = render.render_html(db, engines, content)
            return Response(rendered_content, mimetype="text/html")
        except Exception as e:
            log.error(f"Error reading file: {file_path} - {e}")
            abort(500)

    @app.route("/_put")
    def handle_put():  # pyright: ignore[reportUnusedFunction]
        try:
            query_dict = request.args.to_dict()
            for key, value in query_dict.items():
                db.save(key, float(value))
            return "OK"
        except Exception as e:
            log.error(f"Error handling PUT request: {e}")
            abort(500)

    app.run("0.0.0.0", web_port)
