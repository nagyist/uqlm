# Copyright 2025 CVS Health and/or one of its affiliates
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


from typing import Any, Dict, List
from rich.progress import SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.rule import Rule


HEADERS = ["🤖 Generation", "📈 Scoring", "⚙️ Optimization", "🤖🧮 Generation with Logprobs", "", "  - [black]Grading responses against provided ground truth answers with default grader...", "✂️ Decomposition", "✅️ Refinement", "\n🤖 Claim-QA Answer Generation"]
OPTIMIZATION_TASKS = ["  - [black]Optimizing weights...", "  - [black]Jointly optimizing weights and threshold using grid search...", "  - [black]Optimizing weights using grid search...", "  - [black]Optimizing threshold with grid search..."]


class ConditionalBarColumn(BarColumn):
    def render(self, task):
        if task.description in HEADERS:
            return ""
        return super().render(task)


class ConditionalTimeElapsedColumn(TimeElapsedColumn):
    def render(self, task):
        if task.description in HEADERS:
            return ""
        return super().render(task)


class ConditionalTextColumn(TextColumn):
    def render(self, task):
        if task.description in HEADERS:
            return ""
        elif task.description in OPTIMIZATION_TASKS:
            return f"[progress.percentage]{task.percentage:>3.0f}%"
        return super().render(task)


class ConditionalSpinnerColumn(SpinnerColumn):
    def render(self, task):
        if task.description in HEADERS:
            return ""
        return super().render(task)


def display_response_refinement(original_text: str, claims_data: List[Dict[str, Any]], refined_text: str) -> None:
    """
    Display a formatted comparison between original and refined text with highlighted removed claims.

    Parameters
    ----------
    original_text : str
        The original response text

    claims_to_remove : List[str]
        List of claims to be removed

    refined_text : str
        The refined response text after removing claims
    """

    console = Console()

    # Create centered title
    console.print(Rule(style="black bold"))
    centered_title = Align.center("[bold black]Response Refinement Example[/bold black]")
    console.print(centered_title)
    console.print(Rule(style="black bold"))
    console.print()  # Add a blank line for spacing

    # Convert strings to Text objects if they aren't already
    if isinstance(original_text, str):
        original_text = Text(original_text)

    if isinstance(refined_text, str):
        refined_text = Text(refined_text)

    # Display original response
    console.print(Panel(original_text, title="[bold]Original Response[/bold]", border_style="yellow"))

    # Format claims as a bulleted list in a single Text object
    claims_to_remove = [claims_data[i]["claim"] for i in range(len(claims_data)) if claims_data[i]["removed"]]
    claims_text = Text()
    for i, claim in enumerate(claims_to_remove):
        if i > 0:
            claims_text.append("\n")  # Add spacing between claims
        claims_text.append(f"• {claim}")

    # Display claims to be removed in a simple panel
    console.print(Panel(claims_text, title="[bold]Low-Confidence Claims to be Removed[/bold]", border_style="red"))

    # Display refined response
    console.print(Panel(refined_text, title="[bold]Refined Response[/bold]", border_style="green"))
