import inspect
import textwrap
from dataclasses import fields, is_dataclass

import twrapform
from twrapform.options.options import (
    _TF_OPTION_DESCRIPTION_NAME,
    WorkspaceSelectTaskOptions,
)


def is_strict_subclass(cls: type, base: type) -> bool:
    return issubclass(cls, base) and cls != base


def generate_docstring_for_dataclass(
    clazz: type, description: str | None = None
) -> str:
    doc_lines = []
    attrs = dict()

    if description is not None:
        doc_lines.extend([description, ""])

    doc_lines.append("Attributes:")

    for base in reversed(clazz.__mro__):
        if is_dataclass(base):
            for f in fields(base):
                desc = f.metadata.get(_TF_OPTION_DESCRIPTION_NAME)
                name = f.name
                type_hint = f.type

                if isinstance(desc, tuple):
                    desc_str = " ".join(desc)
                elif isinstance(desc, str):
                    desc_str = desc
                else:
                    continue

                attrs[name] = (type_hint, desc_str)

    for key in sorted(attrs.keys()):
        type_hint, desc_str = attrs[key]
        doc_lines.append(f"    {key} ({type_hint}): {desc_str}")

    return "\n".join(doc_lines)


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

        if is_strict_subclass(module, twrapform.options.TFCommandOptions):

            print(
                f"Inject docstring into {module.__name__}",
                f"target filepath {inspect.getfile(module)}",
            )

            init_args = {}

            if module == WorkspaceSelectTaskOptions:
                init_args["workspace"] = "default"

            inject_docstring_into_file(
                filepath=inspect.getfile(module),
                placeholder="{{{{ {clazz_name}_DOCSTRING }}}}".format(
                    clazz_name=module_name
                ),
                docstring=generate_docstring_for_dataclass(
                    module,
                    f'Options for the "{' '.join(module(**init_args).command)}" command.',
                ),
            )
