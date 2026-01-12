import contextlib
import io
from typing import Any, Optional, List
import warnings

import numpy as np
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from rich.progress import Progress
from uqlm.utils.prompts.entailment_prompts import get_entailment_prompt
from uqlm.utils.response_generator import ResponseGenerator


SYSTEM_PROMPT = "You are a helpful assistant that evaluates natural language inference relationships."
STR_SCORE_MAP = { "yes": 1.0, "no": 0.0}


class EntailmentClassifier(ResponseGenerator):
    def __init__(self, nli_llm: Optional[BaseChatModel] = None, max_calls_per_min: Optional[int] = None) -> None:
        """
        A class to compute NLI predictions.

        Parameters
        ----------
        nli_llm : BaseChatModel, default=None
            A LangChain chat model for LLM-based NLI inference. If provided, takes precedence over nli_model_name.

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.
        """
        super().__init__(llm=llm, max_calls_per_min=max_calls_per_min)


    async def judge_entailment(self, responses: List[str], claims: List[str], retries: int = 5, progress_bar: Optional[Progress] = None) -> Dict[str, Any]:
        """
        Async version of predict() for single NLI prediction.

        This method computes NLI predictions on the provided inputs asynchronously.
        For LangChain models, this enables concurrent LLM calls which significantly improves performance.
        For HuggingFace models, this wraps the synchronous call for API consistency.

        Parameters
        ----------
        responses : List[str]
            The premise text for NLI classification.

        claims : List[str]
            The hypothesis text for NLI classification.

        Returns
        -------
        Dict[str, Any]
            The entailment prompts, raw LLM outputs, and extracted entailment/contradiction scores
        """
        prompts = self._construct_prompts(responses=responses, claims=claims)
        with contextlib.redirect_stdout(io.StringIO()):
            data = await self.generate_responses(prompts=prompts, count=1, system_prompt=SYSTEM_PROMPT, progress_bar=progress_bar)

        # Extract scores
        scores = [self._extract_score(response_text) for response_text in data["data"]["response"]]
        df = pd.DataFrame({"judge_prompts": data["data"]["prompt"], "judge_responses": data["data"]["response"], "scores": scores})

        # Retry logic for failed extractions
        retry = 0
        while retry <= retries:
            retry += 1

            # Find any failures
            score_failures = df[pd.isna(df.scores)]

            # If ANY failures exist, retry 
            if len(score_failures) > 0:
                # Get all failure indices
                failure_indices = set(score_failures.index)

                with contextlib.redirect_stdout(io.StringIO()):
                    tmp = await self.generate_responses(prompts=list(df.loc[list(failure_indices), "judge_prompts"]), count=1, system_prompt=SYSTEM_PROMPT, progress_bar=False)

                retry_data = self._extract_answers(responses=tmp["data"]["response"])

                df.loc[list(failure_indices), "scores"] = retry_data

            # Exit if no more failures
            if len(score_failures) == 0:
                break
        return {col: list(df[col]) for col in df.columns}

    def _extract_score(self, response_text: str) -> float:
        """
        Map response text to score
        """
        clean_text = response_text.strip().lower()
        for word, score in STR_SCORE_MAP.items():
            # Best: response starts with the value
            if clean_text.startswith(word):
                return score
            
        for word, score in STR_SCORE_MAP.items():
            # fallback: substring search
            if word in response_text:
                return score
            
        return np.nan

    
    @staticmethod
    def _construct_prompts(responses: List[str], claims: list[str]) -> List[str]:
        """Construct prompt for entailment evaluation"""
        return [get_entailment_prompt(claim=claims[i], source_text=responses[i], style="binary") for i in range(len(responses))]
    

# from pydantic import BaseModel, Field
# from typing import Any, Optional, Literal, List, Tuple, Union
# class NLIResult(BaseModel):
#     """
#     Result from NLI prediction with probabilities.

#     This unified model supports both binary and ternary NLI styles.
#     The structure adapts based on the `style` field.
#     """

#     style: Literal["binary", "ternary"] = Field(..., description="The NLI style used")

#     # Binary fields (populated when style="binary")
#     binary_label: Optional[bool] = Field(None, description="True if entailed, False otherwise (binary style only)")
#     binary_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Probability of entailment (binary style only)")

#     # Ternary fields (populated when style="ternary")
#     ternary_label: Optional[Literal["contradiction", "neutral", "entailment"]] = Field(None, description="Predicted NLI class (ternary style only)")
#     ternary_probabilities: Optional[Tuple[float, float, float]] = Field(None, description="Probabilities for [contradiction, neutral, entailment] (ternary style only)")

#     @property
#     def label(self) -> Union[bool, str]:
#         """Get the label regardless of style."""
#         if self.style == "binary":
#             return self.binary_label
#         else:  # ternary
#             return self.ternary_label

#     @property
#     def entailment_probability(self) -> Optional[float]:
#         """Get entailment probability regardless of style."""
#         if self.style == "binary" and self.binary_probability:
#             return self.binary_probability
#         elif self.style == "ternary" and self.ternary_probabilities:
#             return self.ternary_probabilities[2]
#         return None

#     @property
#     def contradiction_probability(self) -> Optional[float]:
#         """Get contradiction probability (ternary only)."""
#         if self.style == "ternary" and self.ternary_probabilities:
#             return self.ternary_probabilities[0]
#         return None

#     @property
#     def neutral_probability(self) -> Optional[float]:
#         """Get neutral probability (ternary only)."""
#         if self.style == "ternary" and self.ternary_probabilities:
#             return self.ternary_probabilities[1]
#         return None