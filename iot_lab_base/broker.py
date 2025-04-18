import json
import logging
from dataclasses import dataclass
from typing import Literal

from .db import AppendOnlyDB
from .engine.abc import InferenceEngine

log = logging.getLogger(__name__)

VS_SEP = "__VGMT__"
VS_MAGIC_CODE_S = "_VGMT_S_"
VS_MAGIC_CODE_E = "_VGMT_E_"


@dataclass
class VSPutRequest:
    key: str
    value: int
    action: Literal["PUT"]


@dataclass
class VSGetRequest:
    key: str
    action: Literal["GET", "GET+set-zero", "GET+inference"]


VSRequest = VSPutRequest | VSGetRequest


def parse_vs_request(data: str, log: logging.Logger) -> list[VSRequest]:
    """
    Parse the incoming VegemiteSandwich request.
    Expects the data to accurately preserve commands separated by VS_SEP separator without any
    truncation.
    """
    parsed_line: dict[str, int] = {}
    for l in data.strip().split(VS_SEP):
        try:
            parsed_line.update(json.loads(l.strip()))
        except Exception as e:
            log.error(f"Incoming request is incorrect. I am not at fault. {e}\nLook at this. → {l}")

    result: list[VSRequest] = []
    for key, value in parsed_line.items():
        try:
            real_key = key.split("!!", 1)[1]
            if key.startswith("PUT!!"):
                result.append(VSPutRequest(real_key, value, "PUT"))
            elif key.startswith("GET!!"):
                result.append(VSGetRequest(real_key, "GET"))
            elif key.startswith("GET+set-zero!!"):
                result.append(VSGetRequest(real_key, "GET+set-zero"))
            elif key.startswith("GET+inference!!"):
                result.append(VSGetRequest(real_key, "GET+inference"))
            else:
                log.warning(f"How do you intend to control me? {key}")
        except Exception as e:
            log.error(
                f"Incoming request is incorrect. I am not at fault. {e}\nLook at this. → {key}"
            )

    return result


def make_vs_response(key: str, data: int | float) -> str:
    """
    Make the VegemiteSandwich response.
    VegemiteSandwich response is basically a sequence of key-value pairs whose start and end
    are marked by magic codes.
    """
    return f"{VS_MAGIC_CODE_S}{key}={data}{VS_MAGIC_CODE_E}"


def handle_line(
    db: AppendOnlyDB, engines: dict[str, InferenceEngine], client_id: str, data: str
) -> list[bytes]:
    """
    Handle the incoming line from the VegemiteSandwich.
    """
    log.debug(f">I [{client_id}] {data}")
    retval: list[bytes] = []

    reqs = parse_vs_request(data, log)
    for req in reqs:
        if req.action == "PUT":
            log.info(f"I> [{client_id}] {req.key}={req.value} (PUT)")
            db.save(req.key, req.value)

        elif req.action == "GET" or req.action == "GET+set-zero":
            try:
                item = db.get(req.key)
                if item is not None:
                    log.info(f"<O [{client_id}] {req.key}={item.value} ({req.action})")
                    retval.append(make_vs_response(req.key, item.value).encode())
                    if req.action == "GET+set-zero":
                        db.save(req.key, 0)
                else:
                    log.warning(f"<O [{client_id}] {req.key} not found (GET+set-zero).")
            except Exception as e:
                log.error(f"{req.action} error: {e}")

        elif req.action == "GET+inference":
            try:
                engine = engines.get(req.key)
                if engine is None:
                    log.warning(f"<O [{client_id}] {req.key} not found (GET+inference).")
                    continue
                item = engine.infer()
                log.info(f"<O [{client_id}] {req.key}={item} (GET+inference)")
                retval.append(make_vs_response(req.key, item).encode())
            except Exception as e:
                log.error(f"GET+inference error: {e}")

        else:
            log.warning(f"Unknown action: {req.action}")

    return retval
