import pytest
import numpy as np
from unittest.mock import MagicMock
from uqlm.white_box.sampled_logprobs import SampledLogprobsScorer, SAMPLED_LOGPROBS_SCORER_NAMES


@pytest.fixture
def mock_logprobs_results():
    """Fixture to provide mock logprobs results."""
    return [
        [
            {"token": "a", "logprobs": [-0.1, -1.0, -2.0]},
            {"token": "b", "logprobs": [-0.2, -0.5, -1.5]},
        ],
        [
            {"token": "c", "logprobs": [-0.3, -0.7, -1.2]},
            {"token": "d", "logprobs": [-0.4, -0.8, -1.0]},
        ],
    ]


@pytest.fixture
def mock_sampled_logprobs_results():
    """Fixture to provide mock sampled logprobs results."""
    return [
        [
            [
                {"token": "a", "logprobs": [-0.15, -1.1, -2.1]},
                {"token": "b", "logprobs": [-0.25, -0.6, -1.6]},
            ],
            [
                {"token": "a", "logprobs": [-0.12, -1.05, -2.05]},
                {"token": "b", "logprobs": [-0.22, -0.55, -1.55]},
            ],
        ],
        [
            [
                {"token": "c", "logprobs": [-0.35, -0.75, -1.25]},
                {"token": "d", "logprobs": [-0.45, -0.85, -1.05]},
            ],
            [
                {"token": "c", "logprobs": [-0.32, -0.72, -1.22]},
                {"token": "d", "logprobs": [-0.42, -0.82, -1.02]},
            ],
        ],
    ]


@pytest.fixture
def mock_responses():
    """Fixture to provide mock responses."""
    return ["response_1", "response_2"]


@pytest.fixture
def mock_sampled_responses():
    """Fixture to provide mock sampled responses."""
    return [["sampled_response_1", "sampled_response_2"], ["sampled_response_3", "sampled_response_4"]]


@pytest.fixture
def scorer():
    """Fixture to create a SampledLogprobsScorer instance."""
    return SampledLogprobsScorer()


def test_evaluate(mock_logprobs_results, mock_responses, mock_sampled_responses, mock_sampled_logprobs_results, scorer, monkeypatch):
    """Test the evaluate method of SampledLogprobsScorer."""
    # Mock the _compute_single_generation_scores method
    monkeypatch.setattr(
        scorer,
        "_compute_single_generation_scores",
        lambda logprobs, func: [0.9 for _ in logprobs],
    )

    # Mock the CosineScorer's evaluate method
    monkeypatch.setattr(
        "uqlm.black_box.cosine.CosineScorer.evaluate",
        lambda self, responses, sampled_responses, progress_bar: [0.8, 0.7],
    )

    result = scorer.evaluate(
        logprobs_results=mock_logprobs_results,
        responses=mock_responses,
        sampled_responses=mock_sampled_responses,
        sampled_logprobs_results=mock_sampled_logprobs_results,
    )

    # Verify the result contains all scorer names
    assert set(result.keys()) == set(SAMPLED_LOGPROBS_SCORER_NAMES)

    # Verify the length of the results matches the number of responses
    for key in result:
        assert len(result[key]) == len(mock_responses)


def test_compute_consistency_confidence(mock_logprobs_results, mock_responses, mock_sampled_responses, scorer, monkeypatch):
    """Test the compute_consistency_confidence method."""
    # Mock the _compute_single_generation_scores method
    monkeypatch.setattr(
        scorer,
        "_compute_single_generation_scores",
        lambda logprobs, func: [0.9 for _ in logprobs],
    )

    # Mock the CosineScorer's evaluate method
    monkeypatch.setattr(
        "uqlm.black_box.cosine.CosineScorer.evaluate",
        lambda self, responses, sampled_responses, progress_bar: [0.8, 0.7],
    )

    result = scorer.compute_consistency_confidence(
        logprobs_results=mock_logprobs_results,
        responses=mock_responses,
        sampled_responses=mock_sampled_responses,
    )

    # Verify the result is a list of floats
    assert isinstance(result, list)
    assert all(isinstance(score, float) for score in result)

    # Verify the length of the result matches the number of responses
    assert len(result) == len(mock_responses)


def test_monte_carlo_probability(mock_logprobs_results, mock_sampled_logprobs_results, mock_responses, mock_sampled_responses, scorer, monkeypatch):
    """Test the monte_carlo_probability method."""
    # Mock the _compute_single_generation_scores method
    monkeypatch.setattr(
        scorer,
        "_compute_single_generation_scores",
        lambda logprobs, func: [0.9 for _ in logprobs],
    )

    result = scorer.monte_carlo_probability(
        logprobs_results=mock_logprobs_results,
        sampled_logprobs_results=mock_sampled_logprobs_results,
        responses=mock_responses,
        sampled_responses=mock_sampled_responses,
    )

    # Verify the result is a list of floats
    assert isinstance(result, list)
    assert all(isinstance(score, float) for score in result)

    # Verify the length of the result matches the number of responses
    assert len(result) == len(mock_responses)