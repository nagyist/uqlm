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


from typing import Any, List, Optional
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
import numpy as np


class LongFormUQ(UncertaintyQuantifier):
    def __init__(self, llm: Any = None, device: Any = None, system_prompt: Optional[str] = None, max_calls_per_min: Optional[int] = None, use_n_param: bool = False) -> None:
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
        """
        super().__init__(llm=llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param)

    async def uncertainty_aware_decode(self, claim_sets: List[List[str]], claim_scores: List[List[float]], claim_refinement_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> List[str]:
        """
        Parameters
        ----------
        claim_sets : List[List[str]]
            List of original responses decomposed into lists of claims

        claim_scores : List[List[float]]
            List of lists of claim-level confidence scores to be used for uncertainty-aware filtering

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses
        """
        self._construct_progress_bar(show_progress_bars)
        self._display_reconstruction_header(show_progress_bars)
        uad_result = await self.reconstructor.reconstruct_responses(claim_sets=claim_sets, claim_scores=claim_scores, responses=self.responses, progress_bar=self.progress_bar)
        self._stop_progress_bar()
        self.progress_bar = None

        for scorer in self.scorers:
            filtered_claim_scores = []
            for i in range(len(self.claim_sets)):
                filtered_claim_scores_i = []
                for j in range(len(self.claim_sets[i])):
                    if not uad_result["removed"][i][j]:
                        filtered_claim_scores_i.append(self.claim_scores[scorer][i][j])
                filtered_claim_scores.append(filtered_claim_scores_i)

            uad_result["refined_" + scorer] = self._aggregate_scores(filtered_claim_scores)

        return uad_result

    async def _decompose_responses(self, show_progress_bars) -> None:
        """Decompose original responses into claims or sentences"""
        self._display_decomposition_header(show_progress_bars)
        if self.granularity == "sentence":
            self.claim_sets = self.decomposer.decompose_sentences(responses=self.responses, progress_bar=self.progress_bar)
        elif self.granularity == "claim":
            self.claim_sets = await self.decomposer.decompose_claims(responses=self.responses, progress_bar=self.progress_bar)

    async def _decompose_candidate_responses(self, show_progress_bars) -> None:
        """Display header and decompose responses"""
        if self.granularity == "sentence":
            self.sampled_claim_sets = self.decomposer.decompose_candidate_sentences(sampled_responses=self.sampled_responses, progress_bar=self.progress_bar)
        elif self.granularity == "claim":
            self.sampled_claim_sets = await self.decomposer.decompose_candidate_claims(sampled_responses=self.sampled_responses, progress_bar=self.progress_bar)

    def _aggregate_scores(self, claim_scores: List[List[float]]) -> List[float]:
        """Aggregate claim scores to response level scores"""
        if self.aggregation == "mean":
            return [np.mean(cs) for cs in claim_scores]
        elif self.aggregation == "min":
            return [np.min(cs) for cs in claim_scores]

    def _display_decomposition_header(self, show_progress_bars: bool) -> None:
        """Displays decomposition header"""
        if show_progress_bars:
            self.progress_bar.start()
            self.progress_bar.add_task("")
            self.progress_bar.add_task("✂️ Decomposition")

    def _display_reconstruction_header(self, show_progress_bars: bool) -> None:
        """Displays decomposition header"""
        if show_progress_bars:
            self.progress_bar.start()
            self.progress_bar.add_task("")
            self.progress_bar.add_task("✅️ Refinement")
