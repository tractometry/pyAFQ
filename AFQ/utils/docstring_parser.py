import re
import inspect

__all__ = ["parse_numpy_docstring"]


def _white_space(text):
    if text is None:
        return None
    else:
        return re.sub(r'[\n\t\s]+', ' ', text).strip()


def _split_parameter_blocks(docstring):
    # First extract just the Parameters section
    params_section = re.search(
        r'Parameters\n-+\n(.*?)(?=\n\s*\n|\n-+\n|\Z)',
        docstring,
        re.DOTALL
    )
    if not params_section:
        return []

    params_text = params_section.group(1)
    params_text = re.sub(r'\n-+$', '', params_text)
    lines = params_text.split('\n')

    # Find base indentation (minimum indent of parameter lines)
    base_indent = float('inf')
    for line in lines:
        if re.match(r'\S', line):
            indent = len(re.match(r'^\s*', line).group())
            base_indent = min(base_indent, indent)

    if base_indent == float('inf'):
        return []

    # Split into blocks
    param_blocks = []
    current_block = []

    for line in lines:
        line_indent = len(re.match(r'^\s*', line).group())
        if line_indent == base_indent and re.match(r'\S', line):
            if current_block:
                param_blocks.append('\n'.join(current_block))
                current_block = []
        current_block.append(line)

    if current_block:
        param_blocks.append('\n'.join(current_block))

    return param_blocks


def parse_numpy_docstring(docstring):
    """
    Parse a NumPy-style docstring into parameter information.

    Returns:
        dict: {
            "description": str (first section of docstring),
            "arguments": {
                param_name: {
                    "help": str (description),
                    "metavar": str (type),
                    "default": any (default value)
                }
            }
        }
    """
    if callable(docstring):
        docstring = inspect.getdoc(docstring)

    if not docstring:
        return {"description": "", "arguments": {}}

    # Find the description section
    sections = re.split(r'\n\s*\n', docstring.strip())
    description = sections[0].strip()

    # Find the parameters section
    param_blocks = _split_parameter_blocks(docstring)
    params = {}

    for block in param_blocks:
        # Split into (param: value) and description
        parts = block.split('\n', maxsplit=1)
        if len(parts) == 0:
            continue

        # Parse the (param: value)
        header_match = re.match(
            r'(\w+)\s*:\s*([^,\n]+)(?:,\s*optional)?',
            parts[0].strip()
        )

        if not header_match:
            continue

        name, type_info = header_match.groups()
        desc = parts[1].strip() if len(parts) > 1 else ''

        # Parse the description
        default = None
        default_match = re.search(r'[Dd]efault:\s*([^\n]+)', desc)
        if default_match:
            default = default_match.group(1).strip(" .")
            try:
                default = eval(default)
            except:
                default = _white_space(default)

        params[name] = {
            "help": _white_space(desc),
            "metavar": _white_space(type_info),
            "default": default
        }

    return {
        "description": _white_space(description),
        "arguments": params
    }
