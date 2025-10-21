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


import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from uqlm.white_box.baseclass.white_box_scorer import WhiteBoxScorer
    

class SingleGenerationScorer(WhiteBoxScorer):
    def __init__(self):
        """Class for computing WhiteBox UQ scores with a single generation"""
        super().__init__()
    
    def evaluate_from_logprobs(self, logprobs_results: List[List[Dict[str, Any]]]) -> Dict[str, List[float]]:
        """Compute scores from logprobs results"""
        scores_dict = {
            "normalized_probability": self._compute_single_generation_scores(logprobs_results, self._norm_prob),
            "min_probability": self._compute_single_generation_scores(logprobs_results, self._min_prob),
            "sequence_probability": self._compute_single_generation_scores(logprobs_results, self._seq_prob),
        }
        return scores_dict
    
    def evaluate_from_top_logprobs(self, logprobs_results: List[List[Dict[str, Any]]]) -> Dict[str, List[float]]:
        """Compute scores from top logprobs results"""
        scores_dict = {
            "mean_token_negentropy": self._compute_single_generation_scores(logprobs_results, self._mean_token_negentropy),
            "min_token_negentropy": self._compute_single_generation_scores(logprobs_results, self._min_token_negentropy),
            "probability_margin": self._compute_single_generation_scores(logprobs_results, self._probability_margin),
        }
        return scores_dict        
    
    def _min_prob(self, single_response_logprobs: List[Dict[str, Any]]) -> float:
        """Compute minimum token probability"""
        probs = self.extract_probs(single_response_logprobs)
        return np.min(probs)
    
    def _compute_token_entropies(self, single_response_logprobs: List[Dict[str, Any]]) -> np.ndarray:
        """Compute entropy for each token in the sequence"""
        top_logprobs_list = self.extract_top_logprobs(single_response_logprobs)
        return np.array([self._entropy_from_logprobs(top_logprobs) for top_logprobs in top_logprobs_list])
    
    def _compute_token_negentropies(self, single_response_logprobs: List[Dict[str, Any]]) -> np.ndarray:
        """Compute negentropy for each token in the sequence"""
        entropies = self._compute_token_entropies(single_response_logprobs)
        top_logprobs_list = self.extract_top_logprobs(single_response_logprobs)
        k_values = np.array([len(top_logprobs) for top_logprobs in top_logprobs_list])
        max_entropies = np.log(k_values)
        negentropies = 1 - entropies / max_entropies
        return negentropies
    
    def _mean_token_negentropy(self, single_response_logprobs: List[Dict[str, Any]]) -> float:
        """Compute mean token negentropy across the sequence"""
        negentropies = self._compute_token_negentropies(single_response_logprobs)
        return np.mean(negentropies)
    
    def _min_token_negentropy(self, single_response_logprobs: List[Dict[str, Any]]) -> float:
        """Compute minimum token negentropy across the sequence"""
        negentropies = self._compute_token_negentropies(single_response_logprobs)
        return np.min(negentropies)
    
    def _probability_margin(self, single_response_logprobs: List[Dict[str, Any]]) -> float:
        """Compute mean probability margin (difference between top two probabilities)"""
        top_logprobs_list = self.extract_top_logprobs(single_response_logprobs)
        margins = []
        for top_logprobs in top_logprobs_list:
            probs = np.exp(top_logprobs)
            probs = np.sort(probs)[::-1]
            margin = probs[0] - probs[1]
            margins.append(margin)
        return np.mean(margins)
