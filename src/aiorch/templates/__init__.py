# Copyright 2026 Eresh Gorantla
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Template registry for `aiorch init --template`."""

from __future__ import annotations

from importlib import resources
from typing import NamedTuple


class TemplateInfo(NamedTuple):
    name: str
    description: str


def list_templates() -> list[TemplateInfo]:
    """Discover all available pipeline templates."""
    templates = []
    template_files = resources.files("aiorch.templates")
    for item in sorted(template_files.iterdir()):
        if item.name.endswith(".yaml"):
            name = item.name.removesuffix(".yaml")
            description = ""
            content = item.read_text()
            for line in content.splitlines():
                if line.startswith("# description:"):
                    description = line.split(":", 1)[1].strip()
                    break
            templates.append(TemplateInfo(name=name, description=description))
    return templates


def get_template(name: str) -> str | None:
    """Get template content by name. Returns None if not found."""
    try:
        template_files = resources.files("aiorch.templates")
        resource = template_files.joinpath(f"{name}.yaml")
        return resource.read_text()
    except (FileNotFoundError, TypeError):
        return None
