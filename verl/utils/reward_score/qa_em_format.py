# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
import string
import random

def normalize_answer(s):
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    """Check if prediction matches any golden answer."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_kbqa_sequence(text):
    """
    Check if the text follows valid KBQA reasoning sequence format.
    Expected format: <|im_start|>assistant ... <think>...</think> <sparql>...</sparql> <information>...</information> <answer>...</answer>
    
    Updated to support multi-round reasoning: think -> sparql -> information cycles can repeat before final answer.
    """
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False, "Missing assistant marker"
    
    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Check for balanced tags - KBQA uses think, sparql, information, answer
    tags_to_check = ["think", "sparql", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Check for proper sequence pattern: think -> sparql -> information -> answer (with multi-round support)
    split_pattern = r"(</?(?:think|sparql|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # Track the current position in the expected sequence
    state = "start"  # start -> think -> sparql -> information -> think -> ... -> answer -> end
    
    # Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|sparql|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<sparql>" and state == "after_think":
                state = "in_sparql"
            elif part == "</sparql>" and state == "in_sparql":
                state = "after_sparql"
            elif part == "<information>" and state == "after_sparql":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "information":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_sparql", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_sparql", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"


def extract_answer(solution_str):
    """Extract the final answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    
    # If there are no matches, return None
    if not matches:
        return None
    
    # If there are multiple matches, return the last one
    return matches[-1].strip()


def extract_sparql_query(text: str) -> str:
    """Extract SPARQL query from the text."""
    pattern = r"<sparql>(.*?)</sparql>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return the last match
    return None


def extract_information_blocks(text: str) -> list[str]:
    """Extract information blocks from the text."""
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_information_relevant(text: str, golden_answers: list[str]) -> bool:
    """Check if the information blocks contain relevant information about the golden answers."""
    info_blocks = extract_information_blocks(text)
    for info_block in info_blocks:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(info_block):
                return True
    return False


def is_sparql_correct(text: str, golden_sparql: str) -> bool:
    """Check if the extracted SPARQL query matches the golden SPARQL."""
    extracted_sparql = extract_sparql_query(text)
    if not extracted_sparql or not golden_sparql:
        return False
    
    # Normalize SPARQL queries for comparison
    def normalize_sparql(sparql):
        # Remove extra whitespace and normalize formatting
        sparql = re.sub(r'\s+', ' ', sparql).strip()
        return sparql.lower()
    
    return normalize_sparql(extracted_sparql) == normalize_sparql(golden_sparql)


def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0, final_format_score=0, sparql_bonus_score=0, information_bonus_score=0, score=1.):
    """
    The scoring function for exact match (EM) with format rewards for KBQA.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth dictionary containing 'target' answers and optionally 'sparql' query
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        structure_format_score: the score for correct structural format (proper tag sequence)
        final_format_score: the score for incorrect format but some structure
        sparql_bonus_score: bonus score for correct SPARQL query structure
        information_bonus_score: bonus score for relevant information extraction
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_kbqa_sequence(solution_str)
    
    # Check if SPARQL query is correct (if provided in ground truth)
    sparql_correct = False
    if 'sparql' in ground_truth and ground_truth['sparql']:
        sparql_correct = is_sparql_correct(solution_str, ground_truth['sparql'])
    
    # Check if information is relevant to the golden answers
    information_relevant = False
    if is_valid_format and 'target' in ground_truth:
        information_relevant = is_information_relevant(solution_str, ground_truth['target'])
    
    # Extract the final answer
    answer = extract_answer(solution_str=solution_str)
    
    # Random printing for debugging
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth.get('target', [])}")
        print(f"Extracted answer: {answer}")
        print(f"Valid format: {is_valid_format}")
        print(f"SPARQL correct: {sparql_correct}")
        print(f"Information relevant: {information_relevant}")
        print(f"Solution string: {solution_str}")
    
    # Scoring logic
    if answer is None:
        # No answer extracted
        if is_valid_format:
            bonus = 0
            if sparql_correct:
                bonus += sparql_bonus_score
            if information_relevant:
                bonus += information_bonus_score
            return structure_format_score + bonus  # e.g., 0.2 + 0.1 + 0.1 = 0.4
        else:
            return 0  # No structure, no answer
    else:
        # Answer was extracted, check if it's correct
        if em_check(answer, ground_truth.get('target', [])):
            # Correct answer
            if is_valid_format:
                bonus = 0
                if sparql_correct:
                    bonus += sparql_bonus_score
                if information_relevant:
                    bonus += information_bonus_score
                return score + bonus  # e.g., 1.0 + 0.1 + 0.1 = 1.2 (full score plus bonuses)
            else:
                return score - structure_format_score  # e.g., 1.0 - 0.2 = 0.8 (penalty for bad format)
        else:
            # Wrong answer
            if is_valid_format:
                bonus = 0
                if sparql_correct:
                    bonus += sparql_bonus_score
                if information_relevant:
                    bonus += information_bonus_score
                return structure_format_score + bonus  # e.g., 0.2 + 0.1 + 0.1 = 0.4 (points for good format and process)
            else:
                return final_format_score  # e.g., 0.1 (minimal points for attempt) 