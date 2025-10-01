"""Common file operations.

Notes
-----
    *

References
----------
    [1]

Todo
----
    [ ]

"""

# Standard library
import datetime
from pathlib import Path

# Local libraries
from ariel import DATA, console


def generate_save_path(
    *,
    file_name: str | Path | None = None,
    file_path: str | Path | None = None,
    file_extension: str | None = None,
    append_date: bool = True,
) -> Path:
    # Check that at least one argument is valid
    if (file_name is None) and (file_path is None) and (file_extension is None):
        msg = "All arguments of `generate_save_path` are `None`. "
        msg += "Therefore no sensible name can be generated!"
        raise ValueError(msg)

    # Generate unique string using the date and time
    format_string = "%Y-%m-%d_%H-%M-%S-%f"
    time_zone = datetime.UTC
    date_time_str = str(
        datetime.datetime.now(tz=time_zone).strftime(format_string),
    )

    # Check if 'file path' is a string or None
    if file_path is None:
        file_path = DATA
    file_path = Path(file_path)

    # File extension on `file_path`
    maybe_extension_via_path = file_path.suffixes
    has_ext_via_path = len(maybe_extension_via_path) != 0

    # Check if 'file name' is a string or None
    if file_name is None:
        if has_ext_via_path:
            file_name = file_path.stem
            file_path = file_path.parent
        else:
            file_name = date_time_str
    file_name = Path(file_name)

    # File extension on `file_name`
    maybe_extension_via_name = file_name.suffixes
    has_ext_via_name = len(maybe_extension_via_name) != 0

    # Check that there is a valid `file extension`
    if file_extension is None:
        # Handle extensions
        if has_ext_via_path is True:
            file_extension = maybe_extension_via_path[-1]
        elif has_ext_via_name is True:
            file_extension = maybe_extension_via_name[-1]
        else:
            msg = "No valid file extension was found! "
            msg += "This may yield an invalid file path!"
            console.log(f"[bold yellow] --> {msg}")

    # Create output folder if it does not exist
    if file_path.is_dir():
        file_path.mkdir(exist_ok=True)

    # Generate file file name
    if append_date is True:
        final_file_path = (
            file_path / f"{file_name.stem}_{date_time_str}{file_extension}"
        )
    else:
        final_file_path = file_path / f"{file_name.stem}{file_extension}"
    msg = f"[bold cyan] Saving file --> {final_file_path}"
    console.log(msg)
    return final_file_path
