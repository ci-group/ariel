def exportdb_error_message(case_name: str) -> str: ...
def get_class_if_classified_error(e: Exception) -> str | None:
    """
    Returns a string case name if the export error e is classified.
    Returns None otherwise.
    """
