import pytest
from unittest.mock import AsyncMock, MagicMock
from uqlm.utils.grader import LLMGrader

@pytest.mark.asyncio
async def test_grade_responses():
    """Test the grade_responses method"""
    mock_llm = MagicMock()
    mock_response_generator = AsyncMock()
    mock_response_generator.generate_responses.return_value = {
        "data": {"response": ["yes", "no", "yes"]}
    }
    mock_llm.response_generator = mock_response_generator

    grader = LLMGrader(llm=mock_llm)
    grader.response_generator = mock_response_generator

    prompts = ["What is 2+2?", "What is the capital of France?", "What is 5*5?"]
    responses = ["4", "Berlin", "25"]
    answers = [["4"], ["Paris"], ["25"]]

    result = await grader.grade_responses(prompts, responses, answers)
    assert result == [True, False, True]


def test_extract_grades():
    """Test the _extract_grades method"""
    assert LLMGrader._extract_grades("yes") is True
    assert LLMGrader._extract_grades("no") is False
    assert LLMGrader._extract_grades("YES") is True
    assert LLMGrader._extract_grades("NO") is False
    assert LLMGrader._extract_grades("maybe") is False


def test_construct_grader_prompt():
    """Test the _construct_grader_prompt method"""
    prompt = "What is 2+2?"
    response = "4"
    acceptable_answers = ["4", "four"]

    expected_prompt = """
        Your task is to grade the following proposed answer against the provided answer key. The ground truth is the gold standard regardless of any other information you may have. Return ONLY the word "yes" or "no", with no additional text, based on whether the proposed answer aligns with any of the ground truth answers. Answer "yes" if correct, "no" if incorrect.

        **Question:**
        What is 2+2?

        **Ground Truth Answers (Answer Key):**
        ['4', 'four']

        **Proposed Answer to Grade:**
        4

        Now your answer is (yes or no):
        """
    result = LLMGrader._construct_grader_prompt(prompt, response, acceptable_answers)
    assert result.strip() == expected_prompt.strip()