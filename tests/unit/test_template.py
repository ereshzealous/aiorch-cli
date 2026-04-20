import jinja2
import pytest

from aiorch.core.template import ShellTemplateError, resolve, resolve_for_shell


def test_resolve_substitutes_variable():
    assert resolve("hello {{name}}", {"name": "Eresh"}) == "hello Eresh"


def test_resolve_raises_on_missing_variable():
    with pytest.raises(jinja2.exceptions.UndefinedError):
        resolve("hello {{missing}}", {})


def test_resolve_for_shell_auto_quotes_whitespace_and_specials():
    out = resolve_for_shell("echo {{msg}}", {"msg": "hi there; rm -rf /"})
    # The dangerous payload must be quoted so the shell sees it as one arg.
    assert "'hi there; rm -rf /'" in out
    assert not out.endswith("/")


def test_resolve_for_shell_rejects_jinja_inside_quotes():
    # User wrapped {{x}} in double quotes — auto-quoting would double-quote it,
    # producing a broken command. The resolver must refuse.
    with pytest.raises(ShellTemplateError):
        resolve_for_shell('echo "{{x}}"', {"x": "value"})
