import importlib.util
import os
import re


def import_vars_from_path(file_path):
    """
    Dynamically imports a Python file from a given path and returns it as a module.

    Args:
        file_path (str): The absolute or relative path to the .py file.

    Returns:
        module: The loaded module object, or None if the file doesn't exist.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    # 1. Create a unique module name from the file path to avoid conflicts.
    # e.g., 'path/to/my_config.py' becomes 'path.to.my_config'
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # 2. Create a "module spec" from the file location.
    # This spec contains the information Python needs to load the module.
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        print(f"Error: Could not create module spec for {file_path}")
        return None

    # 3. Create a new, empty module object based on the spec.
    module = importlib.util.module_from_spec(spec)

    # 4. Execute the code from the file in the new module's namespace.
    # This is the step that actually runs the .py file and populates
    # the 'module' object with its variables.
    try:
        spec.loader.exec_module(module)
        print(f"Successfully imported module: {module_name}")
        return module
    except Exception as e:
        print(f"Error executing module {module_name}: {e}")
        return None


def cleanup_models(dir_path: str) -> None:
    """
    Removes older or redundant model checkpoint files within a directory.
    It looks for numbered 'model_*.pth' files and removes all but the one with the highest number.

    Args:
        dir_path (str): The path to the directory containing checkpoint files.
    """
    model_pattern = re.compile(r'^model_(\d+)\.pth$')  # only exact pattern
    candidates = []
    for fname in os.listdir(dir_path):
        full = os.path.join(dir_path, fname)
        if os.path.isfile(full):
            m = model_pattern.match(fname)
            if m:
                num = int(m.group(1))
                candidates.append((num, full))

    if not candidates:
        return  # nothing to cleanup

    # find the highest-numbered file
    best_num, best_path = max(candidates, key=lambda x: x[0])

    # delete all model_*.pth except the best one
    for num, full in candidates:
        if full != best_path:
            os.remove(full)
    return
