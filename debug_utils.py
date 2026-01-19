"""
Debug utilities for inspecting and visualizing various objects in a clean way.
"""


def inspect_object(obj, max_length: int = 100):
    """Inspect and print object properties with truncated output."""
    # ANSI escape codes for formatting
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    # Get the type/class name of the object
    obj_type = type(obj).__name__
    obj_module = type(obj).__module__
    if obj_module != 'builtins':
        full_type_name = f"{obj_module}.{obj_type}"
    else:
        full_type_name = obj_type
    
    header = f"=== {full_type_name.upper()} PROPERTIES ==="
    print(f"\n{BOLD}{header}{RESET}")
    
    # Show length if the object supports it
    try:
        obj_len = len(obj)
        print(f"{BOLD}Length:{RESET} {obj_len}")
    except (TypeError, AttributeError):
        pass
    
    methods = []
    
    # Get all attributes of the object
    for attr_name in dir(obj):
        if not attr_name.startswith('_'):  # Skip private attributes
            try:
                attr_value = getattr(obj, attr_name)
                if callable(attr_value):
                    methods.append(attr_name)
                else:
                    str_value = str(attr_value)
                    # Truncate long strings to first max_length characters
                    if len(str_value) > max_length:
                        truncated = str_value[:max_length] + "..."
                    else:
                        truncated = str_value
                    # Replace newlines with \\n for cleaner output
                    truncated = truncated.replace('\n', '\\n')
                    print(f"{BOLD}{attr_name}:{RESET} {truncated}")
            except Exception as e:
                print(f"{BOLD}{attr_name}:{RESET} <error accessing: {e}>")
    
    # Show methods at the bottom if any exist
    if methods:
        print(f"\n{BOLD}Methods:{RESET} {', '.join(methods)}")
    
    print(f"{BOLD}{'=' * len(header)}{RESET}\n")


def inspect_lm_prediction(lm_pred, max_length: int = 100):
    """Legacy alias for inspect_object - kept for backward compatibility."""
    inspect_object(lm_pred, max_length) 