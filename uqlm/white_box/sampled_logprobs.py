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


from typing import List, Dict, Any, Optional
from rich.progress import Progress
from uqlm.black_box.cosine import CosineScorer
from uqlm.white_box.baseclass.logprobs_scorer import LogprobsScorer


SAMPLED_LOGPROBS_SCORER_NAMES = [
    # "semantic_negentropy", "semantic_density",
    "monte_carlo_negentropy",
    "consistency_and_confidence",
]


class SampledLogprobsScorer(LogprobsScorer):
    def __init__(self, scorers: List[str] = SAMPLED_LOGPROBS_SCORER_NAMES):
        super().__init__()
        self.scorers = scorers

    def evaluate(self, logprobs_results: List[List[Dict[str, Any]]], responses: List[str], sampled_responses: List[List[str]], sampled_logprobs_results: Optional[List[List[List[Dict[str, Any]]]]] = None, progress_bar: Optional[Progress] = None):
        scores_dict = {}
        if "monte_carlo_negentropy" in self.scorers:
            scores_dict["monte_carlo_negentropy"] = self.compute_monte_carlo_sequence_entropy(logprobs_results=logprobs_results, responses=responses, sampled_responses=sampled_responses, sampled_logprobs_results=sampled_logprobs_results)
        if "consistency_and_confidence" in self.scorers:
            scores_dict["consistency_and_confidence"] = self.compute_consistency_confidence(logprobs_results=logprobs_results, responses=responses, sampled_responses=sampled_responses, progress_bar=progress_bar)
        return {k: scores_dict[k] for k in self.scorers}

    def compute_consistency_confidence(self, logprobs_results: List[List[Dict[str, Any]]], responses: List[str], sampled_responses: List[List[str]], progress_bar: Optional[Progress] = None) -> List[float]:
        cosine_scores = CosineScorer().evaluate(responses=responses, sampled_responses=sampled_responses, progress_bar=progress_bar)
        response_probs = self._compute_single_generation_scores(logprobs_results, self._norm_prob)
        cocoa_scores = [cs * rp for cs, rp in zip(cosine_scores, response_probs)]
        return cocoa_scores

    def compute_monte_carlo_sequence_entropy(self, logprobs_results: List[List[Dict[str, Any]]], sampled_logprobs_results: List[List[List[Dict[str, Any]]]], responses: List[str], sampled_responses: List[List[str]]) -> List[float]:
        num_responses = len(sampled_responses[0]) + 1
        monte_carlo_negentropy_scores = []
        for i in range(len(responses)):
            all_logprobs_response_i = [logprobs_results[i]] + sampled_logprobs_results[i]
            all_responses_i = [responses[i]] + sampled_responses[i]

            all_sampled_sequence_probs_response_i = self._compute_single_generation_scores(all_logprobs_response_i, self._norm_prob)
            monte_carlo_entropy_i = self._entropy_from_probs(probs_list=all_sampled_sequence_probs_response_i, texts=all_responses_i)
            monte_carlo_negentropy_i = 1 - monte_carlo_entropy_i / num_responses
            monte_carlo_negentropy_scores.append(monte_carlo_negentropy_i)
        return monte_carlo_negentropy_scores
