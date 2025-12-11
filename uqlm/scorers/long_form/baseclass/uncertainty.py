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


# import io
# import contextlib
# from typing import Any, List, Optional, Union
# from langchain_core.messages import BaseMessage
# from rich.progress import Progress, TextColumn
# from rich.errors import LiveError

# from uqlm.utils.response_generator import ResponseGenerator
# from uqlm.nli.nli import NLI
# from uqlm.judges.judge import LLMJudge
# from uqlm.utils.display import ConditionalBarColumn, ConditionalTimeElapsedColumn, ConditionalTextColumn, ConditionalSpinnerColumn

from uqlm.scorers.short_form.baseclass.uncertainty import UncertaintyQuantifier

DEFAULT_BLACK_BOX_SCORERS = ["semantic_negentropy", "noncontradiction", "exact_match", "cosine_sim"]

BLACK_BOX_SCORERS = DEFAULT_BLACK_BOX_SCORERS + ["bert_score", "entailment", "semantic_sets_confidence"]

DEFAULT_WHITE_BOX_SCORERS = ["normalized_probability", "min_probability"]


class LongFormUncertaintyQuantifier(UncertaintyQuantifier):
    def __init__(self, llm: Any = None, device: Any = None, system_prompt: Optional[str] = None, max_calls_per_min: Optional[int] = None, use_n_param: bool = False, postprocessor: Optional[Any] = None) -> None:
        """
        Parent class for uncertainty quantification of LLM responses

        Parameters
        ----------
        llm : BaseChatModel
            A langchain llm object to get passed to chain constructor. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `llm` object.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Only applies to 'semantic_negentropy', 'noncontradiction'
            scorers. Pass a torch.device to leverage GPU.

        system_prompt : str, default=None
            Optional argument for user to provide custom system prompt. If prompts are list of strings and system_prompt is None,
            defaults to "You are a helpful assistant."

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when num_responses > 1.

        postprocessor : callable, default=None
            A user-defined function that takes a string input and returns a string. Used for postprocessing
            outputs.
        """
        super().__init__(llm=llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param, postprocessor=postprocessor)
