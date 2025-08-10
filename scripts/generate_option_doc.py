import inspect
import textwrap

import twrapform


def is_strict_subclass(cls: type, base: type) -> bool:
    return issubclass(cls, base) and cls != base


def inject_docstring_into_file(filepath: str, placeholder: str, docstring: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    l_indent = "    "

    docstring = textwrap.indent(docstring, l_indent).lstrip() + f"\n{l_indent}"

    updated = content.replace(placeholder, docstring)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(updated)


if __name__ == "__main__":
    all_modules = getattr(twrapform.options, "__all__", [])
    for module_name in all_modules:
        module = getattr(twrapform.options, module_name)

        if all(
            [
                is_strict_subclass(module, twrapform.options.TFCommandOptions),
            ]
        ):
            print(
                f"Inject docstring into {module.__name__}",
                f"target filepath {inspect.getfile(module)}",
            )

            inject_docstring_into_file(
                filepath=inspect.getfile(module),
                placeholder="{{{{ {clazz_name}_DOCSTRING }}}}".format(
                    clazz_name=module_name
                ),
                docstring=module.__doc__,
            )
