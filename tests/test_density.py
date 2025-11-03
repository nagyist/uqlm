import pytest
from unittest.mock import AsyncMock, MagicMock
from uqlm.scorers.density import SemanticDensity
from uqlm.utils.results import UQResult


@pytest.mark.asyncio
async def test_generate_and_score():
    # Mock dependencies
    mock_llm = MagicMock()
    mock_llm.logprobs = True
    mock_nli_scorer = MagicMock()
    mock_nli_scorer.num_responses = 5

    # Create an instance of SemanticDensity
    semantic_density = SemanticDensity(llm=mock_llm)
    semantic_density._setup_nli = MagicMock()
    semantic_density._construct_progress_bar = MagicMock()
    semantic_density._display_generation_header = MagicMock()
    semantic_density.generate_original_responses = AsyncMock(return_value=["response1", "response2"])
    semantic_density.generate_candidate_responses = AsyncMock(return_value=[["sample1", "sample2"], ["sample3", "sample4"]])
    semantic_density.score = MagicMock(return_value=UQResult({"data": {}, "metadata": {}}))

    # Test the method
    prompts = ["prompt1", "prompt2"]
    result = await semantic_density.generate_and_score(prompts, num_responses=2)

    # Assertions
    assert semantic_density.prompts == prompts
    assert semantic_density.num_responses == 2
    semantic_density.generate_original_responses.assert_called_once_with(prompts, progress_bar=semantic_density.progress_bar)
    semantic_density.generate_candidate_responses.assert_called_once_with(prompts, num_responses=2, progress_bar=semantic_density.progress_bar)
    semantic_density.score.assert_called_once()


def test_score():
    # Mock dependencies
    mock_nli_scorer = MagicMock()
    mock_nli_scorer._semantic_density_process = MagicMock(return_value=("density_value", None))

    semantic_density = SemanticDensity()
    semantic_density.nli_scorer = mock_nli_scorer
    semantic_density._construct_progress_bar = MagicMock()
    semantic_density._display_scoring_header = MagicMock()
    semantic_density._stop_progress_bar = MagicMock()
    semantic_density._construct_black_box_return_data = MagicMock(return_value={})
    semantic_density.progress_bar = MagicMock()
    semantic_density.progress_bar.add_task = MagicMock(return_value="task_id")
    semantic_density.progress_bar.update = MagicMock()

    # Test data
    responses = ["response1", "response2"]
    sampled_responses = [["sample1", "sample2"], ["sample3", "sample4"]]
    semantic_density.responses = responses
    semantic_density.sampled_responses = sampled_responses
    semantic_density.prompts = ["prompt1", "prompt2"]
    semantic_density.multiple_logprobs = [["logprob1", "logprob2"], ["logprob3", "logprob4"]]

    # Call the method
    result = semantic_density.score(responses, sampled_responses)

    # Assertions
    assert "semantic_density_values" in result.data
    assert "multiple_logprobs" in result.data
    mock_nli_scorer._semantic_density_process.assert_called()


if __name__ == "__main__":
    pytest.main()