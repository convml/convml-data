import importlib
from pathlib import Path


def get_user_function(**kwargs):
    user_function_context = kwargs.get("context", {})
    datasource_path = user_function_context.get("datasource_path", None)
    product_name = user_function_context.get("product_name", None)
    if datasource_path is None:
        raise NotImplementedError(
            "To use a user-function to generate an user you need to pass"
            " in the `datasource_path` as entry in a dict argument called"
            " `context`"
        )
    if product_name is None:
        raise NotImplementedError(
            "To use a user-function to generate an user you need to pass"
            " in the `product_name` as entry in a dict argument called"
            " `context`"
        )

    module_fpath = Path(datasource_path) / "user_functions.py"
    if not module_fpath.exists():
        raise NotImplementedError(
            f"To use user-function you should create a file called {module_fpath.name}"
            f" in {module_fpath.parent} containing a function called `{product_name}`"
            " and taking `da_scene` as an argument"
        )
    spec = importlib.util.spec_from_file_location("user_function", module_fpath)
    user_function_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_function_module)

    user_function = getattr(user_function_module, product_name, None)
    if user_function is None:
        raise NotImplementedError(
            f"Please define `{product_name}` in the user-function module source"
            f" file `{module_fpath}`"
        )

    return user_function
