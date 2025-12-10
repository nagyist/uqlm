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

import asyncio
import time
from typing import List, Optional
import numpy as np
from uqlm.utils.prompts import get_response_reconstruction_prompt
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel


class UncertaintyAwareDecoder:
    def __init__(self, reconstructor_llm: Optional[BaseChatModel] = None, threshold: float = 1 / 3, aggregation: str = "mean") -> None:
        """
        Class for decomposing responses into individual claims or sentences. This class is used as an intermediate
        step for longform UQ methods.

        Parameters
        ----------
        reconstructor_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        threshold : float, default=1/3
            Threshold used for uncertainty-aware filtering
        """
        self.reconstructor_llm = reconstructor_llm
        self.threshold = threshold
        self.aggregation = aggregation
        self.reconstruction_template = get_response_reconstruction_prompt

    async def reconstruct_responses(self, claim_sets: List[List[str]], claim_scores: List[List[float]], progress_bar: Optional[Progress] = None) -> List[str]:
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
        filtered_claim_scores, filtered_claim_sets, retain_indicators = [], [], []
        for i in range(len(claim_sets)):
            filtered_claim_scores_i, filtered_claim_set_i, retain_indicators_i = [], [], []
            for j in range(len(claim_scores[i])):
                if claim_scores[i][j] > self.threshold:
                    filtered_claim_scores_i.append(claim_scores[i][j])
                    filtered_claim_set_i.append(claim_sets[i][j])
                    retain_indicators_i.append(True)
                else:
                    retain_indicators_i.append(False)
            filtered_claim_scores.append(filtered_claim_scores_i)
            filtered_claim_sets.append(filtered_claim_set_i)
            retain_indicators.append(retain_indicators_i)

        if progress_bar:
            self.progress_task = progress_bar.add_task(" - Reconstructing responses with high-confidence claims...", total=len(claim_sets))
        tasks = [self._reconstruct_single_response(claim_set=filtered_claim_sets[i], progress_bar=progress_bar) for i in range(len(claim_sets))]
        reconstructed_responses = await asyncio.gather(*tasks)
        time.sleep(0.1)
        return {"uad_responses": reconstructed_responses, "claim_retained": retain_indicators, "uad_response_scores": self._aggregate_scores(filtered_claim_scores)}

    async def _reconstruct_single_response(self, claim_set: List[str], progress_bar: Optional[Progress] = None) -> str:
        """Decompose single response into claims using LLM and extract claims from the result"""
        reconstructed_response = await self.reconstructor_llm.ainvoke(self.reconstruction_template(claim_set))
        if progress_bar:
            progress_bar.update(self.progress_task, advance=1)
        return reconstructed_response.content

    def _aggregate_scores(self, claim_scores: List[List[float]]) -> List[float]:
        """Aggregate filtered claim scores to response level scores"""
        if self.aggregation == "mean":
            return [np.mean(cs) for cs in claim_scores]
        elif self.aggregation == "min":
            return [np.min(cs) for cs in claim_scores]
