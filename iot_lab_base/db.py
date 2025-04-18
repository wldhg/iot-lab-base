import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from threading import Lock

log = logging.getLogger(__name__)


@dataclass
class AppendOnlyDBEntry:
    value: float
    ts: int  # timestamp (epoch time in seconds)


AppendOnlyDBData = dict[str, list[AppendOnlyDBEntry]]


class AppendOnlyDB:
    """
    Thread-safe append-only database.
    The database is stored in a binary file and can be used to store
    key-value pairs with timestamps. The database is loaded from the file
    on initialization and saved to the file on every write.
    """

    def __init__(self):
        self.lock = Lock()
        self.db: AppendOnlyDBData = {}

        self.db_dir = os.path.join(os.path.dirname(__file__), "..")
        self.db_path = os.path.join(self.db_dir, "db.bin")

        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                while True:
                    key_len = f.read(4)
                    if not key_len:
                        break
                    key_len = int(struct.unpack(">I", key_len)[0])
                    key_bytes = f.read(key_len)
                    value_bytes = f.read(8)
                    ts_bytes = f.read(4)
                    if not key_bytes or not value_bytes or not ts_bytes:
                        break
                    key = key_bytes.decode("utf-8")
                    value = float(struct.unpack(">d", value_bytes)[0])
                    ts = int(struct.unpack(">I", ts_bytes)[0])
                    if key not in self.db:
                        self.db[key] = []
                    self.db[key].append(AppendOnlyDBEntry(value, ts))

        self.db_file = open(self.db_path, "a+b")

    def get(self, key: str) -> AppendOnlyDBEntry | None:
        """
        Get the last entry for a given key.
        If the key does not exist, return None.
        """
        with self.lock:
            if key not in self.db or not self.db[key]:
                return None
            return self.db[key][-1]

    def get_last_n(self, key: str, n: int) -> list[AppendOnlyDBEntry]:
        """
        Get the last n entries for a given key.
        If the key does not exist, return an empty list.
        """
        with self.lock:
            if key not in self.db or not self.db[key]:
                return []
            return self.db[key][-n:]

    def stat(self) -> dict[str, int]:
        """
        Get the number of entries for each key in the database.
        """
        with self.lock:
            return {key: len(entries) for key, entries in self.db.items()}

    def save(self, key: str, value: float, ts: int | None = None):
        """
        Save a value for a given key.
        If ts (timestamp) is not provided, use the current time.
        """
        if ts is None:
            ts = int(time.time())
        with self.lock:
            if key not in self.db:
                self.db[key] = []
            self.db[key].append(AppendOnlyDBEntry(value, ts))
            key_encoded = key.encode("utf-8")
            value_encoded = struct.pack(">d", value)
            ts_encoded = struct.pack(">I", ts)
            len_encoded = struct.pack(">I", len(key_encoded))
            self.db_file.write(len_encoded + key_encoded + value_encoded + ts_encoded)
            self.db_file.flush()
        log.debug(f"Saved {key}={value} at {ts}")

    def dump(self) -> str:
        """
        Dump the database to a JSON file.
        This is useful for making training data.
        Returns the path to the dump file.
        """
        dump_path = os.path.join(self.db_dir, f"db-dump-{int(time.time())}.json")
        with open(dump_path, "w") as f:
            json.dump(self.db, f, indent=2)
        return dump_path

    def close(self):
        """
        Close the database file.
        """
        if not self.db_file.closed:
            self.db_file.close()
