import logging
import socket
import socketserver
from typing import override

import serial

from .broker import VS_SEP, handle_line
from .db import AppendOnlyDB
from .engine.abc import InferenceEngine

SERIAL_BAUD_RATE = 9600


def proc_broker(
    db: AppendOnlyDB,
    engines: dict[str, InferenceEngine],
    tcp_port: int,
    serial_dev: str,
):
    if serial_dev != "":
        log = logging.getLogger(f"{__name__}:DMYSYS")
        log.info(f"DUMMY BROKER IS BOOTING UP... on {serial_dev}")

        with serial.Serial(serial_dev, SERIAL_BAUD_RATE, timeout=5) as s:
            buffer = ""
            while True:
                lines = s.read(2048).decode(errors="ignore")
                if not lines:
                    continue
                buffer += lines
                while VS_SEP in buffer:
                    line, buffer = buffer.split(VS_SEP, 1)
                    if not line.strip():
                        continue

                    retvals = handle_line(db, engines, serial_dev, line)
                    for retval in retvals:
                        s.write(retval)

    else:
        log = logging.getLogger(__name__)

        class VSTCPHandler(socketserver.BaseRequestHandler):
            @override
            def handle(self):
                req: socket.socket = self.request
                client_id = str(self.client_address)
                log.info(f"+ [{client_id}] connected")

                try:
                    while True:
                        data_bytes = req.recv(4096)
                        if not data_bytes:
                            break

                        data = data_bytes.decode()

                        for line in data.split(VS_SEP):
                            if not line.strip():
                                continue

                            retvals = handle_line(db, engines, client_id, line)
                            for retval in retvals:
                                req.sendall(retval)

                finally:
                    log.info(f"- [{client_id}] disconnected")

        with socketserver.TCPServer(("0.0.0.0", tcp_port), VSTCPHandler) as server:
            log.info(f"Broker Thread 『Start』。 TCP broker is running on port {tcp_port}.")
            server.serve_forever()
