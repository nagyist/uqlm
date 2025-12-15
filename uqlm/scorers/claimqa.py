import numpy as np
from typing import List, Optional, Any
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.longform.decomposition.response_decomposer import ResponseDecomposer
from uqlm.longform.claim_qa.question_generator import QuestionGenerator
from uqlm.utils.prompts.claim_qa import get_answer_template
from uqlm.utils.results import UQResult
from uqlm.scorers import BlackBoxUQ
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
from uqlm.longform.uad import UncertaintyAwareDecoder


class ClaimQA(UncertaintyQuantifier):
    def __init__(
        self,
        llm: BaseChatModel,
        claim_decomposition_llm: BaseChatModel = None,
        question_generator_llm: BaseChatModel = None,
        scorers: Optional[List[str]] = None,
        granularity: str = "claim",
        aggregation: str = "mean",
        claim_refinement: bool = False,
        claim_refinement_threshold: float = 1 / 3,
        system_prompt: str = "You are a helpful assistant.",
        sampling_temperature: float = 1.0,
        max_calls_per_min: Optional[int] = None,
        questioner_max_calls_per_min: Optional[int] = None,
        num_questions: int = 1,
        num_claim_qa_responses: int = 5,
        max_length: int = 1000,
        device: Any = None,
        use_n_param: bool = False,
    ):
        """
        Initialize the ClaimQAScorer.

        Parameters
        ----------
        llm : BaseChatModel
            The original LLM to use for generating responses.
        llm_decomposer : BaseChatModel
            The LLM to use for decomposing the claims.
        question_generator_llm : BaseChatModel
            The LLM to use for generating questions.
        response_template : str, default="atomic"
            The template to use for generating responses. Choose from "atomic" or "factoid".
        system_prompt : str
            The system prompt to use for generating responses. Default is "You are a helpful assistant."
        sampling_temperature : float, default=1.0
            The temperature to use for sampling the responses.
        max_calls_per_min : int
            The maximum number of calls per minute to the LLM.
        use_n_param : bool
            Whether to use the n parameter for the LLM.
        num_questions : int, default=1
            The number of questions to generate for each factoid.
        num_claim_qa_responses : int, default=2
            The number of responses to generate for each claim-inverted question.
        """
        super().__init__(llm=llm, system_prompt=system_prompt, max_calls_per_min=max_calls_per_min, use_n_param=use_n_param)
        self.num_questions = num_questions
        self.num_claim_qa_responses = num_claim_qa_responses
        self.granularity = granularity
        self.claim_refinement = claim_refinement
        self.claim_refinement_threshold = claim_refinement_threshold
        self.aggregation = aggregation
        self.scorers = scorers

        self.decomposer = ResponseDecomposer(claim_decomposition_llm=claim_decomposition_llm if claim_decomposition_llm else llm)
        self.question_generator = QuestionGenerator(question_generator_llm=question_generator_llm if question_generator_llm is not None else self.decomposer.claim_decomposition_llm, max_calls_per_min=questioner_max_calls_per_min, num_questions=self.num_questions)
        self.bb_object = BlackBoxUQ(llm=llm, scorers=scorers, device=device, max_calls_per_min=max_calls_per_min, sampling_temperature=sampling_temperature, max_length=max_length)
        if self.claim_refinement:
            if self.granularity != "claim":
                raise ValueError("Uncertainty aware decoding is only possible with claim-level scoring. Please set claim_refinement=False or granularity='claim'")
            self.reconstructor = UncertaintyAwareDecoder(reconstructor_llm=self.decomposer.claim_decomposition_llm, threshold=self.claim_refinement_threshold, aggregation=self.aggregation)
            self.uad_scorer = self.scorers[0]
        self.uad_result = {}

    async def generate_and_score(self, prompts: List[str], claim_refinement_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Generate and score the responses.

        Parameters
        ----------
        prompts : List[str]
            A list of prompts to generate responses from LLM.
        progress_bar : Optional[Progress], default=None
            A progress bar to display the progress of the generation.
        """
        self._construct_progress_bar(show_progress_bars)
        self._display_generation_header(show_progress_bars)

        responses = await self.generate_original_responses(prompts=prompts, progress_bar=self.progress_bar)
        return await self.score(prompts=prompts, responses=responses, show_progress_bars=show_progress_bars)

    async def score(self, prompts: List[str], responses: List[str], claim_refinement_threshold: float = 1 / 3, show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Evaluate the QuesAns scores for a given set of claim_sets.

        Parameters
        ----------
        responses : List[str]
            A list of responses to be scored.
        progress_bar : Optional[Progress], default=None
            A progress bar to display the progress of the evaluation.
        """
        self.prompts = prompts
        self.responses = responses

        self._construct_progress_bar(show_progress_bars)
        await self._decompose_responses(show_progress_bars)

        result = await self._score_from_decomposed(prompts=self.prompts, responses=responses, claim_sets=self.claim_sets, progress_bar=self.progress_bar)
        return result

    async def _score_from_decomposed(self, claim_sets: List[List[str]], responses: Optional[List[str]] = None, prompts: Optional[List[str]] = None, progress_bar: Optional[Progress] = None):
        """
        Evaluate the ClaimQA scores for a given set of prompts, responses, and claim_sets.
        """
        self.claim_sets = claim_sets
        self.prompts = [None] * len(claim_sets) if not prompts else prompts
        self.responses = [None] * len(claim_sets) if not responses else responses

        responses_flat, prompts_flat = [], []
        for prompt, response, claim_set in zip(self.prompts, self.responses, self.claim_sets):
            prompts_flat.extend([prompt] * len(claim_set) * self.num_questions)
            responses_flat.extend([response] * len(claim_set) * self.num_questions)
        num_claims = [len(claim_set) for claim_set in self.claim_sets]

        generated_questions = await self.question_generator.generate_questions(
            claim_sets=self.claim_sets,
            # responses=self.responses,
            progress_bar=progress_bar,
        )
        formatted_claim_questions = [
            get_answer_template(
                claim_question=generated_questions[i],
                original_question=prompts_flat[i],
                # original_response=responses_flat[i]
            )
            for i in range(len(generated_questions))
        ]

        self.bb_object.progress_bar = progress_bar
        self.bb_object.generation_type = "claim_qa"
        bb_result = await self.bb_object.generate_and_score(prompts=formatted_claim_questions, num_responses=self.num_claim_qa_responses, show_progress_bars=True if progress_bar else False)
        self.scores_dict = self._process_bb_result(bb_result=bb_result, formatted_claim_questions=generated_questions, num_claims=num_claims)

        if self.claim_refinement:
            self.uad_result = await self.uncertainty_aware_decode(claim_sets=self.claim_sets, claim_scores=self.claim_scores[self.uad_scorer], show_progress_bars=True if progress_bar else False)

        self.scores_dict["claims_data"] = self._extract_claim_data()

        if "removed" in self.uad_result:
            del self.uad_result["removed"]

        self._stop_progress_bar()
        self.progress_bar = None

        return self._construct_result()

    def _process_bb_result(self, bb_result: Any, formatted_claim_questions: List[str], num_claims: List[float]) -> None:
        """Format BlackBoxUQ output data"""
        self.claim_scores = {key: [] for key in self.bb_object.scorers}
        self.response_fact_questions, self.response_fact_questions_responses, self.response_fact_questions_sampled_responses = [], [], []

        initial_index = 0
        for i in range(len(self.claim_sets)):
            self.response_fact_questions.append([formatted_claim_questions[j : j + self.num_questions] for j in range(initial_index, initial_index + num_claims[i] * self.num_questions, self.num_questions)])
            tmp_data = bb_result.to_dict()["data"]
            self.response_fact_questions_responses.append([tmp_data["responses"][j : j + self.num_questions] for j in range(initial_index, initial_index + num_claims[i] * self.num_questions, self.num_questions)])
            self.response_fact_questions_sampled_responses.append([tmp_data["sampled_responses"][j : j + self.num_questions] for j in range(initial_index, initial_index + num_claims[i] * self.num_questions, self.num_questions)])
            for key in self.bb_object.scorers:
                tmp = bb_result.to_dict()["data"][key][initial_index : initial_index + num_claims[i] * self.num_questions]
                if self.num_questions == 1:
                    tmp_claim_scores = tmp
                else:
                    tmp_claim_scores = [np.mean(tmp[j * self.num_questions : (j + 1) * self.num_questions]) for j in range(num_claims[i])]
                self.claim_scores[key].append(tmp_claim_scores)
            initial_index += num_claims[i] * self.num_questions
        scores_dict = {key: self._aggregate_scores(scores) for key, scores in self.claim_scores.items()}
        return scores_dict

    def _extract_claim_data(self) -> None:
        """Extract claims data"""
        claims_data = []
        for i in range(len(self.claim_sets)):
            claim_i_data = []
            for j in range(len(self.claim_sets[i])):
                claims_dict = {self.granularity: self.claim_sets[i][j], "removed": False if not self.uad_result else self.uad_result["removed"][i][j], "claim_questions": self.response_fact_questions[i][j], "claim_qa_responses": self.response_fact_questions_responses[i][j], "claim_qa_sampled_responses": self.response_fact_questions_sampled_responses[i][j]}
                for scorer in self.bb_object.scorers:
                    claims_dict[scorer] = self.claim_scores[scorer][i][j]
                claim_i_data.append(claims_dict)
            claims_data.append(claim_i_data)
        return claims_data

    def _construct_result(self) -> Any:
        """Constructs UQResult object"""
        data = {}
        if self.prompts:
            data["prompts"] = self.prompts
        if self.responses:
            data["responses"] = self.responses

        data.update(self.scores_dict)
        data.update(self.uad_result)
        result = {"data": data, "metadata": {"granularity": self.granularity, "aggregation": self.aggregation, "temperature": None if not self.llm else self.llm.temperature, "sampling_temperature": self.bb_object.sampling_temperature, "num_claim_qa_responses": self.num_claim_qa_responses, "claim_refinement_threshold": self.claim_refinement_threshold}}
        return UQResult(result)

    async def _decompose_responses(self, show_progress_bars) -> None:
        """Display header and decompose responses"""
        self._display_decomposition_header(show_progress_bars)
        if self.granularity == "sentence":
            self.claim_sets = self.decomposer.decompose_sentences(responses=self.responses, progress_bar=self.progress_bar)
        elif self.granularity == "claim":
            self.claim_sets = await self.decomposer.decompose_claims(responses=self.responses, progress_bar=self.progress_bar)

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

    def _aggregate_scores(self, claim_scores: List[List[float]]) -> List[float]:
        """Aggregate claim scores to response level scores"""
        if self.aggregation == "mean":
            return [np.mean(cs) for cs in claim_scores]
        elif self.aggregation == "min":
            return [np.min(cs) for cs in claim_scores]

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
