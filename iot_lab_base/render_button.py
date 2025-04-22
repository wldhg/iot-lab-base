import random


def render_button(placeholder: str) -> str:
    """
    Render a button who puts data into the database.
    Placeholder format: "button:label:key=value"
    """
    try:
        _, label, key_value = placeholder.split(":", 2)
        key, value = key_value.split("=")
        value = float(value)
    except Exception as e:
        raise ValueError(f"Invalid placeholder format: {placeholder}") from e

    button_id = "".join(random.choices("abcdef0123456789", k=5))

    button_html = f"""
    <button id="btn_{button_id}">{label}</button>
    <script>
        document.getElementById("btn_{button_id}").onclick = function() {{
            fetch("/_put?{key}={value}").then(function(response) {{
                alert("Success: (" + response.status + ") {key} = " + {value});
            }}).catch(function(error) {{
                alert("Error: " + error);
            }});
        }};
    </script>
    """
    return button_html
