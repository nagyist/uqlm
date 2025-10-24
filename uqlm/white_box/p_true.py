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


from typing import Any, Dict, List, Optional
import numpy as np
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.utils.response_generator import ResponseGenerator

PTRUE_SYSTEM_PROMPT = """
Your task is to determine whether a given answer to a question is correct.

Guidelines for your evaluation:
- Do NOT penalize phrasing differences
- Respond with EXACTLY one word: "True" or "False"
- Answer "True" if the response is correct
- Answer "False" if the response is incorrect
- Do not explain your reasoning or provide any additional commentary
"""


class PTrueScorer:
    def __init__(self, llm: BaseChatModel, max_calls_per_min: Optional[int] = None) -> None:
        llm.logprobs = True
        self.response_generator = ResponseGenerator(llm, max_calls_per_min=max_calls_per_min)
        self.response_generator.response_generator_type = "p_true"

    async def evaluate(self, prompts: List[str], responses: List[str], progress_bar: Optional[Progress] = None) -> Dict[str, float]:
        ptrue_prompts = [self._construct_ptrue_prompt(original_prompt, original_response) for original_prompt, original_response in zip(prompts, responses)]
        ptrue_responses = await self.response_generator.generate_responses(prompts=ptrue_prompts, system_prompt=PTRUE_SYSTEM_PROMPT, progress_bar=progress_bar)
        logprob_results = ptrue_responses["metadata"]["logprobs"]
        ptrue_scores = [self._extract_ptrue_from_logprobs_result(logprob_result) for logprob_result in logprob_results]
        return {"p_true": ptrue_scores}

    @staticmethod
    def _extract_ptrue_from_logprobs_result(logprobs_result: List[Dict[str, Any]]) -> float:
        first_token_data = logprobs_result[0]
        token = first_token_data.get("token", "").strip().lower()
        logprob = first_token_data.get("logprob", None)

        if logprob is not None:
            prob = np.exp(logprob)
            if token.startswith("true"):
                return prob  # High prob means high P_true
            elif token.startswith("false"):
                return 1.0 - prob  # High prob of False means low P_true
            else:
                return np.nan

    @staticmethod
    def _construct_ptrue_prompt(original_prompt: str, original_response: str) -> str:
        ptrue_prompt = f"""
        Question: {original_prompt}
        
        Proposed Answer: {original_response}
        
        Is the proposed answer to the question true or false? Answer with only one word true/false.
        
        True or False:
        """
        return ptrue_prompt
