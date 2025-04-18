import glob
import logging
import os
import threading
from typing import Annotated

import typer
from rich.logging import RichHandler


def main(
    tcp_port: Annotated[int, typer.Option(help="TCP port for the broker")] = 3545,
    web_port: Annotated[int, typer.Option(help="Web port for the server")] = 3600,
    serial_dev: Annotated[
        str,
        typer.Option(
            help="Serial device for the mocked broker (give a device path to enable mocked broker)"
        ),
    ] = "",
    debug: Annotated[bool, typer.Option(help="Enable debug mode for the logging")] = False,
):
    if serial_dev:
        if not os.path.exists(serial_dev):
            raise ValueError(f"Serial device {serial_dev} does not exist.")

    if tcp_port < 1024 or tcp_port > 65535:
        raise ValueError("TCP port must be between 1024 and 65535.")

    if web_port < 1024 or web_port > 65535:
        raise ValueError("Web port must be between 1024 and 65535.")

    if tcp_port == web_port:
        raise ValueError("TCP port and web port must be different.")

    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG if debug else logging.INFO,
        handlers=[RichHandler()],
    )

    from .db import AppendOnlyDB
    from .engine.abc import InferenceEngine
    from .proc_broker import proc_broker
    from .proc_server import proc_server

    log = logging.getLogger(__name__)

    log.info("Launching the IoT Lab Base... Cheesed to meet you!")

    # Load the database
    db = AppendOnlyDB()
    db_stat = db.stat()
    for key, length in db_stat.items():
        log.info(f"Loaded {key} data: {length} entries")
    log.info("Successfully loaded all data.")

    # Load the inference engines
    engines: dict[str, InferenceEngine] = {}
    for engine_file in glob.glob(os.path.join(os.path.dirname(__file__), "engine", "*.py")):
        engine_module_name = os.path.basename(engine_file).replace(".py", "")
        module = __import__(f"iot_lab_base.engine.{engine_module_name}")
        engine_class: type[InferenceEngine] | None = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, InferenceEngine):
                engine_class = obj
                break
        if engine_class is not None:
            engine = engine_class(db)
            engines[engine.model_name] = engine
            log.info(f"Loaded engine: {engine.model_name}")

    # Start the server and broker threads
    server_thr = threading.Thread(
        target=proc_server,
        args=(db, engines, web_port),
        name="server_thr",
        daemon=True,
    )
    server_thr.start()
    broker_thr = threading.Thread(
        target=proc_broker,
        args=(db, engines, tcp_port, serial_dev),
        name="broker_thr",
        daemon=True,
    )
    broker_thr.start()

    # Wait for the threads to finish
    server_thr.join()
    broker_thr.join()

    # Clean up
    log.warning(f"I-It's not like I want to be here or anything... B-Bye!")
    db.close()


app = typer.Typer()
app.command()(main)


if __name__ == "__main__":
    app()
