import random

from .db import AppendOnlyDB


def render_checkbox(db: AppendOnlyDB, placeholder: str) -> str:
    """
    Render a checkbox that puts data into the database.
    Placeholder format: "chkbox:label:key=checked_value,unchecked_value"
    """
    try:
        _, label, key_value = placeholder.split(":", 2)
        key, values = key_value.split("=")
        checked_value, unchecked_value = values.split(",")
        checked_value = float(checked_value)
        unchecked_value = float(unchecked_value)
    except Exception as e:
        raise ValueError(f"Invalid placeholder format: {placeholder}") from e

    db_item = db.get(key)
    is_checked = True if (db_item is not None and db_item.value == checked_value) else False

    chkbox_id = "".join(random.choices("abcdef0123456789", k=5))
    chkbox_html = f"""
    <label for="chk_{chkbox_id}">{label}</label>
    <input type="checkbox" id="chk_{chkbox_id}" checked="{"true" if is_checked else "false"}">
    <script>
        document.getElementById("chk_{chkbox_id}").onchange = function(event) {{
            const value = event.target.checked ? {checked_value} : {unchecked_value};
            fetch("/_put?{key}=" + value)
                .then(response => alert("Success: (" + response.status + ") {key} = " + value))
                .catch(error => alert("Error: " + error));
        }}
    </script>
    """
    return chkbox_html
