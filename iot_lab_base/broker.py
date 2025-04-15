import json
import logging
from typing import Literal, TypedDict

from .db import AppendOnlyDB
from .engine.abc import InferenceEngine

log = logging.getLogger(__name__)

VS_SEP = "__VGMT__"
VS_MAGIC_CODE_S = "_VGMT_S_"
VS_MAGIC_CODE_E = "_VGMT_E_"


class VSPutRequest(TypedDict):
    key: str
    value: int
    action: Literal["PUT"]


class VSGetRequest(TypedDict):
    key: str
    action: Literal["GET", "GET+set-zero", "GET+inference"]


VSRequest = VSPutRequest | VSGetRequest


def parse_vs_request(line: str, log: logging.Logger) -> list[VSRequest]:
    parsed_line: dict[str, int] = {}
    for l in line.strip().split(VS_SEP):
        try:
            parsed_line.update(json.loads(l.strip()))
        except Exception as e:
            log.error(f"Incoming request is incorrect. I am not at fault. {e}\nLook at this. → {l}")

    result: list[VSRequest] = []
    for key, value in parsed_line.items():
        try:
            real_key = key.split("!!", 1)[1]
            if key.startswith("PUT!!"):
                result.append({"key": real_key, "value": value, "action": "PUT"})
            elif key.startswith("GET!!"):
                result.append({"key": real_key, "action": "GET"})
            elif key.startswith("GET+set-zero!!"):
                result.append({"key": real_key, "action": "GET+set-zero"})
            elif key.startswith("GET+inference!!"):
                result.append({"key": real_key, "action": "GET+inference"})
            else:
                log.warning(f"How do you intend to control me? {key}")
        except Exception as e:
            log.error(
                f"Incoming request is incorrect. I am not at fault. {e}\nLook at this. → {key}"
            )

    return result


def make_vs_response(key: str, data: int | float) -> str:
    return f"{VS_MAGIC_CODE_S}{key}={data}{VS_MAGIC_CODE_E}"


def handle_line(
    db: AppendOnlyDB, engines: dict[str, InferenceEngine], client_id: str, line: str
) -> list[bytes]:
    log.debug(f">I [{client_id}] {line}")
    retval: list[bytes] = []

    reqs = parse_vs_request(line, log)
    for req in reqs:
        if req["action"] == "PUT":
            log.info(f"I> [{client_id}] {req['key']}={req['value']}")
            db.save(req["key"], req["value"])

        elif req["action"] == "GET" or req["action"] == "GET+set-zero":
            try:
                value = db.get(req["key"])
                if value is not None:
                    log.info(f"<O [{client_id}] {req['key']}={value}")
                    retval.append(make_vs_response(req["key"], value["value"]).encode())
                    if req["action"] == "GET+set-zero":
                        db.save(req["key"], 0)
                else:
                    log.warning(f"<O [{client_id}] {req['key']} not found.")
            except Exception as e:
                log.error(f"{req["action"]} error: {e}")

        elif req["action"] == "GET+inference":
            try:
                engine = engines.get(req["key"])
                if engine is None:
                    log.warning(f"<O [{client_id}] {req['key']} not found.")
                    continue
                value = engine.infer()
                log.info(f"<O [{client_id}] {req['key']}={value}")
                retval.append(make_vs_response(req["key"], value).encode())
            except Exception as e:
                log.error(f"GET+inference error: {e}")

        else:
            log.warning(f"Unknown action: {req['action']}")

    return retval
