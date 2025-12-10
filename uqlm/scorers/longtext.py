from typing import Any, Dict, List, Optional, Union
import numpy as np
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
from uqlm.utils.results import UQResult
from uqlm.longform.black_box.matched_unit import MatchedUnitScorer
from uqlm.longform.black_box.unit_response import UnitResponseScorer
from uqlm.longform.decomposition import ResponseDecomposer
from uqlm.longform.uad import UncertaintyAwareDecoder

# SENTENCE_BLACKBOX_SCORERS = ["response_sent_entail", "response_sent_noncontradict", "response_sent_contrast_entail"]
# CLAIM_BLACKBOX_SCORERS = ["response_claim_entail", "response_claim_noncontradict", "response_claim_contrast_entail"]
# MATCHED_CLAIM_BLACKBOX_SCORERS = ["matched_claim_entail", "matched_claim_noncontradict", "matched_claim_contrast_entail"]
# MATCHED_SENTENCE_BLACKBOX_SCORERS = ["matched_sent_entail", "matched_sent_noncontradict", "matched_sent_contrast_entail"]

# UNIT_RESPONSE_SCORERS = SENTENCE_BLACKBOX_SCORERS + CLAIM_BLACKBOX_SCORERS
# MATCHED_UNIT_SCORERS = MATCHED_CLAIM_BLACKBOX_SCORERS + MATCHED_SENTENCE_BLACKBOX_SCORERS
# DATACLASS_NAMES = ["claim_entail_scores", "claim_noncontradict_scores", "claim_constrast_entail_scores"]
# DATACLASS_TO_SCORER_MAP = {scorer: dataclass_name for scorer, dataclass_name in zip(UNIT_RESPONSE_SCORERS + MATCHED_UNIT_SCORERS, DATACLASS_NAMES * 4)}
UNIT_RESPONSE_SCORERS = ["entailment", "noncontradiction", "contrasted_entailment"]
MATCHED_UNIT_SCORERS = UNIT_RESPONSE_SCORERS + ["bert_score", "cosine_sim"]

class LongTextUQ(UncertaintyQuantifier):
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        granularity: str = "claim",
        mode: str = "unit_response",
        scorers: Optional[List[str]] = None,
        aggregation: Optional[str] = None,
        uad_filtering: bool = False,
        uad_threshold: float = 1 / 3,
        claim_decomposition_llm: Optional[BaseChatModel] = None,
        device: Any = None,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        system_prompt: str = "You are a helpful assistant.",
        max_calls_per_min: Optional[int] = None,
        sampling_temperature: float = 1.0,
        use_n_param: bool = False,
        max_length: int = 2000,
    ) -> None:
        """
        Class for longform uncertainty quantification. Implements claimwise analogs of other UQ-based
        scorers for improved performance with longform tasks.

        Parameters
        ----------
        llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        scorers : subset of {"entailment", "noncontradiction", "contrasted_entailment", "bert_score", "cosine_sim"}, default=None
            Specifies which black box (consistency) scorers to include. If None, defaults to ["entailment"].
            
        granularity : str, default="claim"
            Specifies whether to decompose and score at claim or sentence level granularity. Must be either "claim" or "sentence"
            
        mode : str, default="unit_response"
            Specifies whether to implement unit-response (LUQ-style) scoring or matched-unit (LUQ-pair-style) scoring 

        aggregation : str, default="mean"
            Specifies how to aggregate claim/sentence-level scores to response-level scores. Must be one of 'min' or 'mean'.

        claim_decomposition_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel` to be used for decomposing responses into individual claims.
            If granularity="claim" and claim_decomposition_llm is None, the provided `llm` will be used for claim decomposition.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that NLI model use for prediction. Applies to 'luq', 'luq_atomic'
            scorers. Pass a torch.device to leverage GPU.

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        system_prompt : str or None, default="You are a helpful assistant."
            Optional argument for user to provide custom system prompt

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        sampling_temperature : float, default=1.0
            The 'temperature' parameter for llm model to generate sampled LLM responses. Must be greater than 0.

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when num_responses > 1.

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError
        """
        # TODO: enable scorer specification
        # TODO: return claim scores and response scores?
        # TODO: enable UAD filtering and reconstruction

        super().__init__(llm=llm, device=device, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param)
        self.mode = mode
        self.granularity = granularity
        self.aggregation = aggregation
        self.max_length = max_length
        self.sampling_temperature = sampling_temperature
        self.nli_model_name = nli_model_name
        self.claim_decomposition_llm = claim_decomposition_llm
        self.decomposer = ResponseDecomposer(claim_decomposition_llm=claim_decomposition_llm if claim_decomposition_llm else llm)
        self.scorers = scorers
        self.uad_filtering = uad_filtering
        self.uad_threshold = uad_threshold
        self._validate_scorers()
        self.prompts = None
        self.responses = None
        self.claim_sets = None
        self.sentence_sets = None
        self.sampled_responses = None
        self.sampled_claim_sets = None
        self.sampled_sentence_sets = None
        self.num_responses = None
        self.uad_result = {}

    async def generate_and_score(self, prompts: List[str], num_responses: int = 5, uad_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Generate LLM responses, sampled LLM (candidate) responses, and compute confidence scores with specified scorers for the provided prompts.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        num_responses : int, default=5
            The number of sampled responses used to compute consistency.

        show_progress_bars : bool, default=True
            If True, displays progress bars while generating and scoring responses

        Returns
        -------
        UQResult
            UQResult containing data (prompts, responses, and scores) and metadata
        """
        self.prompts = prompts
        self.num_responses = num_responses

        self._construct_progress_bar(show_progress_bars)
        self._display_generation_header(show_progress_bars)

        responses = await self.generate_original_responses(prompts=prompts, progress_bar=self.progress_bar)
        sampled_responses = await self.generate_candidate_responses(prompts=prompts, progress_bar=self.progress_bar, num_responses=self.num_responses)
        result = await self.score(responses=responses, sampled_responses=sampled_responses, show_progress_bars=show_progress_bars)
        return result

    async def score(self, responses: List[str], sampled_responses: List[List[str]], uad_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Compute confidence scores with specified scorers on provided LLM responses. Should only be used if responses and sampled responses
        are already generated. Otherwise, use `generate_and_score`.

        Parameters
        ----------
        responses : list of str, default=None
            A list of model responses for the prompts.

        sampled_responses : list of list of str, default=None
            A list of lists of sampled LLM responses for each prompt. These will be used to compute consistency scores by comparing to
            the corresponding response from `responses`.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while scoring responses

        Returns
        -------
        UQResult
            UQResult containing data (responses and scores) and metadata
        """
        self.responses = responses
        self.sampled_responses = sampled_responses
        self.num_responses = len(sampled_responses[0])
        self._construct_progress_bar(show_progress_bars)

        await self._decompose_responses(show_progress_bars)
        self._display_scoring_header(show_progress_bars)

        self.scores_dict = self._score_from_decomposed(claim_sets=self.claim_sets, sentence_sets=self.sentence_sets, sampled_responses=self.sampled_responses, sampled_claim_sets=self.sampled_claim_sets, progress_bar=self.progress_bar)
        
        if self.uad_filtering:
            claim_scores_for_uad = self.scores_dict["claim_" + self.scorers[0]]
            self.uad_result = await self.uncertainty_aware_decode(claim_sets=self.claim_sets, claim_scores=claim_scores_for_uad, show_progress_bars=show_progress_bars)

        self._stop_progress_bar()
        self.progress_bar = None
            
        return self._construct_result()
    
    async def uncertainty_aware_decode(self, claim_sets: List[List[str]], claim_scores: List[List[float]], uad_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> List[str]:
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
        uad_result = await self.reconstructor.reconstruct_responses(claim_sets=claim_sets, claim_scores=claim_scores, progress_bar=self.progress_bar)
        self._stop_progress_bar()
        self.progress_bar = None
        return uad_result

    def _score_from_decomposed(self, claim_sets: List[List[str]], sentence_sets: List[List[str]], sampled_responses: Optional[List[List[str]]] = None, sampled_claim_sets: Optional[List[List[List[str]]]] = None, progress_bar: Optional[Progress] = None) -> UQResult:
        """
        Compute confidence scores with specified scorers on provided LLM responses. Should only be used if responses and sampled responses
        are already generated. Otherwise, use `generate_and_score`.

        Parameters
        ----------
        claim_sets : list of list of strings
            List of original responses decomposed into lists of either claims or sentences

        sampled_responses : list of list of strings
            Candidate responses to be compared to the decomposed original responses

        sampled_claim_sets : list of list of list of strings
            Decomposed responses to be compared to the decomposed original responses

        Returns
        -------
        UQResult
            UQResult containing data (responses and scores) and metadata
        """
        claim_sets = self.claim_sets if self.granularity == "claim" else self.sentence_sets
        if self.mode == "unit_response":
            score_results = self.unit_response_scorer.evaluate(claim_sets=claim_sets, sampled_responses=sampled_responses, progress_bar=progress_bar).to_dict()
        elif self.mode == "matched_unit":
            sampled_claim_sets = self.sampled_claim_sets if self.granularity == "claim" else self.sampled_sentence_sets
            score_results = self.matched_unit_scorer.evaluate(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets, progress_bar=progress_bar).to_dict()

        scores_dict = {}
        for scorer, scores in score_results.items():
            if scorer in self.scorers:
                scores_dict[self.granularity + "_" + scorer] = [list(claim_scores) for claim_scores in scores]
                scores_dict["aggregated" + "_" + scorer] = self._aggregate_scores(scores)
        return scores_dict
                
    async def _decompose_responses(self, show_progress_bars) -> None:
        """Display header and decompose responses"""
        self._display_decomposition_header(show_progress_bars)
        if self.granularity == "sentence":
            self.sentence_sets = self.decomposer.decompose_sentences(responses=self.responses, progress_bar=self.progress_bar)
            if self.mode == "matched_unit":
                self.sampled_sentence_sets = self.decomposer.decompose_candidate_sentences(sampled_responses=self.sampled_responses, progress_bar=self.progress_bar)
        elif self.granularity == "claim":
            self.claim_sets = await self.decomposer.decompose_claims(responses=self.responses, progress_bar=self.progress_bar)
            if self.mode == "matched_unit":
                self.sampled_claim_sets = await self.decomposer.decompose_candidate_claims(sampled_responses=self.sampled_responses, progress_bar=self.progress_bar)

    def _aggregate_scores(self, claim_scores: List[List[float]]) -> List[float]:
        """Aggregate claim scores to response level scores"""
        if self.aggregation == "mean":
            return [np.mean(cs) for cs in claim_scores]        
        elif self.aggregation == "min":
            return [np.min(cs) for cs in claim_scores]

    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        data = {"responses": self.responses, "sampled_responses": self.sampled_responses}
        if self.prompts:
            data["prompts"] = self.prompts
        if self.claim_sets:
            data["claim_sets"] = self.claim_sets
        if self.sentence_sets:
            data["sentence_sets"] = self.sentence_sets
        data.update(self.scores_dict)
        data.update(self.uad_result)
        result = {"data": data, "metadata": {"mode": self.mode, "granularity": self.granularity, "aggregation": self.aggregation, "temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": None if not self.sampling_temperature else self.sampling_temperature, "num_responses": self.num_responses}}
        return UQResult(result)

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
            self.progress_bar.add_task("✅️ Uncertainty-Aware Decoding")

    def _validate_scorers(self) -> None:
        """Validate scorers"""
        if not self.scorers:
            self.scorers = ["entailment"]
        
        self.matched_unit_scorer = None
        self.unit_response_scorer = None
        if self.granularity not in ["sentence", "claim"]:
            raise ValueError(
                f"""
                Invalid granularity: {self.granularity}. Must be one of "sentence", "claim"
                """
            )
        if self.uad_filtering:
            if self.granularity != "claim":
                raise ValueError("Uncertainty aware decoding is only possible with claim-level scoring. Please set uad_filtering=False or granularity='claim'")
            self.reconstructor = UncertaintyAwareDecoder(
                reconstructor_llm=self.decomposer.claim_decomposition_llm, 
                threshold=self.uad_threshold,
                aggregation=self.aggregation,
            )
        if self.mode == "unit_response":
            if set(self.scorers) - set(UNIT_RESPONSE_SCORERS):
                raise ValueError(
                f"""
                Invalid scorers: {set(self.scorers) - set(UNIT_RESPONSE_SCORERS)}. Must be subset of {UNIT_RESPONSE_SCORERS} when mode="unit_response"
                """
            )
            self.unit_response_scorer = UnitResponseScorer(nli_model_name=self.nli_model_name, device=self.device, max_length=self.max_length)
        elif self.mode == "matched_unit":
            self.matched_unit_scorer = MatchedUnitScorer(nli_model_name=self.nli_model_name, device=self.device, max_length=self.max_length)
            if set(self.scorers) - set(MATCHED_UNIT_SCORERS):
                raise ValueError(
                f"""
                Invalid scorers: {set(self.scorers) - set(MATCHED_UNIT_SCORERS)}. Must be subset of {MATCHED_UNIT_SCORERS} when mode="matched_unit"
                """
            )
        else:
            raise ValueError(
                f"""
                Invalid mode: {self.mode}. Must be one of "unit_response", "matched_unit"
                """
            )