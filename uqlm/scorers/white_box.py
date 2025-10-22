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

from typing import Any, Dict, List, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from uqlm.white_box.single_logprobs import SingleLogprobsScorer, SINGLE_LOGPROBS_SCORER_NAMES
from uqlm.white_box.top_logprobs import TopLogprobsScorer, TOP_LOGPROBS_SCORER_NAMES
from uqlm.white_box.sampled_logprobs import SampledLogprobsScorer, SAMPLED_LOGPROBS_SCORER_NAMES
from uqlm.white_box.reflexive import PTrueScorer
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
from uqlm.utils.results import UQResult

ALL_WHITE_BOX_SCORER_NAMES = SINGLE_LOGPROBS_SCORER_NAMES + TOP_LOGPROBS_SCORER_NAMES + SAMPLED_LOGPROBS_SCORER_NAMES + "p_true"


class WhiteBoxUQ(UncertaintyQuantifier):
    def __init__(self, llm: Optional[BaseChatModel] = None, system_prompt: Optional[str] = None, max_calls_per_min: Optional[int] = None, scorers: Optional[List[str]] = None, top_k_logprobs: int = 15) -> None:
        """
        Class for computing white-box UQ confidence scores. This class offers two confidence scores, normalized
        probability :footcite:`malinin2021uncertaintyestimationautoregressivestructured` and minimum probability :footcite:`manakul2023selfcheckgptzeroresourceblackboxhallucination`.

        Parameters
        ----------
        llm : BaseChatModel
            A langchain llm object to get passed to chain constructor. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `llm` object.

        max_calls_per_min : int, default=None
            Used to control rate limiting.

        system_prompt : str, default=None
            Optional argument for user to provide custom system prompt. If prompts are list of strings and system_prompt is None,
            defaults to "You are a helpful assistant."

        scorers : subset of {
            "normalized_probability", "min_probability", "sequence_probability", "max_token_negentropy", "mean_token_negentropy", "probability_margin"
        }, default=None
            Specifies which black box (consistency) scorers to include. If None, defaults to all.
        """
        super().__init__(llm=llm, max_calls_per_min=max_calls_per_min, system_prompt=system_prompt)
        self.top_k_logprobs = top_k_logprobs
        self._validate_scorers(scorers, top_k_logprobs)

    async def generate_and_score(self, prompts: List[Union[str, List[BaseMessage]]], num_responses: Optional[int] = 5, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Generate responses and compute white-box confidence scores based on extracted token probabilities.

        Parameters
        ----------
        prompts : List[Union[str, List[BaseMessage]]]
            List of prompts from which LLM responses will be generated. Prompts in list may be strings or lists of BaseMessage. If providing
            input type List[List[BaseMessage]], refer to https://python.langchain.com/docs/concepts/messages/#langchain-messages for support.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while generating and scoring responses

        Returns
        -------
        UQResult
            UQResult containing prompts, responses, logprobs, and white-box UQ scores
        """
        assert hasattr(self.llm, "logprobs"), """
        BaseChatModel must have logprobs attribute and have logprobs=True
        """
        self.llm.logprobs = True

        self._construct_progress_bar(show_progress_bars)
        self._display_generation_header(show_progress_bars, white_box=True)

        responses = await self.generate_original_responses(prompts, progress_bar=self.progress_bar)
        if self.multi_logprobs_scorer_names:
            sampled_responses = await self.generate_candidate_responses(prompts=prompts, num_responses=num_responses, progress_bar=self.progress_bar)
        result = self.score(prompts=prompts, responses=responses, sampled_responses=sampled_responses, logprobs_results=self.logprobs, sampled_logprobs_results=self.multiple_logprobs)

        self._stop_progress_bar()
        self.progress_bar = None  # if re-run ensure the same progress object is not used
        return result

    async def score(self, logprobs_results: List[List[Dict[str, Any]]], prompts: Optional[List[str]] = None, responses: Optional[List[str]] = None, sampled_responses: Optional[List[List[str]]] = None, sampled_logprobs_results: Optional[List[List[List[Dict[str, Any]]]]] = None) -> UQResult:
        """
        Compute white-box confidence scores from provided logprobs.

        Parameters
        ----------
        logprobs_results : list of logprobs_result
            List of dictionaries, each returned by BaseChatModel.agenerate

        prompts : list of str, default=None
            A list of input prompts for the model.

        responses : list of str, default=None
            A list of model responses for the prompts.

        Returns
        -------
        UQResult
            UQResult containing prompts, responses, logprobs, and white-box UQ scores
        """

        data = {"prompts": prompts, "responses": responses, "logprobs_results": logprobs_results, "sampled_responses": sampled_responses, "sampled_logprobs_results": sampled_logprobs_results}
        data = {key: val for key, val in data.items if val}

        if self.single_logprobs_scorer_names:
            single_logprobs_scores_dict = self.single_generation_scorer.evaluate_from_logprobs(logprobs_results)
            data.update(single_logprobs_scores_dict)
        if self.multi_logprobs_scorer_names:
            multip_logprobs_scores_dict = self.single_generation_scorer.evaluate_from_top_logprobs(logprobs_results)
            data.update(multip_logprobs_scores_dict)
        if self.multi_generation_scorer_names:
            multi_generation_scores_dict = self.multi_generation_scorer.evaluate(logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results, responses=responses, sampled_responses=sampled_responses)
            data.update(multi_generation_scores_dict)
        if "p_true" in self.scorers:
            p_true_scores_dict = await self.p_true_scorer.evaluate(prompts=prompts, responses=responses)
            data.update(p_true_scores_dict)
        result = {"data": data, "metadata": {"temperature": None if not self.llm else self.llm.temperature}}
        return UQResult(result)

    def _validate_scorers(self, scorers: List[str], top_k_logprobs: int) -> None:
        """Validate and store scorer list"""
        self.scorers = self.white_box_names if not scorers else []
        for scorer in scorers:
            if scorer in ALL_WHITE_BOX_SCORER_NAMES:
                self.scorers.append(scorer)
            else:
                raise ValueError(f"Invalid scorer provided: {scorer}")
        self.single_logprobs_scorer_names = list(set(SINGLE_LOGPROBS_SCORER_NAMES) & set(scorers))
        self.top_logprobs_scorer_names = list(set(TOP_LOGPROBS_SCORER_NAMES) & set(scorers))
        self.sampled_logprobs_scorer_names = list(set(SAMPLED_LOGPROBS_SCORER_NAMES) & set(scorers))
        if self.single_logprobs_scorer_names:
            self.single_logprobs_scorer = SingleLogprobsScorer(scorers=self.single_logprobs_scorer_names)
        if self.top_logprobs_scorer_names:
            self.top_logprobs_scorer = TopLogprobsScorer(scorers=self.top_logprobs_scorer_names)
        if self.sampled_logprobs_scorer_names:
            self.sampled_logprobs_scorer = SampledLogprobsScorer(scorers=self.sampled_logprobs_scorer_names)
        if "p_true" in scorers:
            self.p_true_scorer = PTrueScorer(llm=self.llm, max_calls_per_min=self.max_calls_per_min)
