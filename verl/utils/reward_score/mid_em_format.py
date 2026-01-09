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
import json
import random
import logging
import os
from datetime import datetime
from . import mid_reward
from . import qa_em_format

# Configure logger for MID extraction statistics
def _setup_mid_extraction_logger():
    """Setup logger for MID extraction statistics."""
    # Get experiment name
    experiment_name = os.getenv('EXPERIMENT_NAME', 'unknown_experiment')
    
    # Create experiment-specific logs directory
    base_log_dir = "logs"
    if experiment_name and experiment_name != 'unknown_experiment':
        log_dir = os.path.join(base_log_dir, experiment_name)
    else:
        log_dir = base_log_dir
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger('mid_extraction_stats')
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler for detailed logs
        log_file = os.path.join(log_dir, f'mid_extraction_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler for important info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Global statistics tracker
class MIDExtractionStats:
    def __init__(self):
        self.total_calls = 0
        self.successful_extractions = 0
        self.failed_extractions = 0
        self.valid_format_count = 0
        self.invalid_format_count = 0
        self.sparql_correct_count = 0
        self.information_relevant_count = 0
        self.last_report_count = 0
        self.report_interval = 100  # Report every 100 calls
        
    def update(self, predicted_mids, is_valid_format, sparql_correct, information_relevant):
        """Update statistics with new extraction result."""
        self.total_calls += 1
        
        if predicted_mids is not None:
            self.successful_extractions += 1
        else:
            self.failed_extractions += 1
            
        if is_valid_format:
            self.valid_format_count += 1
        else:
            self.invalid_format_count += 1
            
        if sparql_correct:
            self.sparql_correct_count += 1
            
        if information_relevant:
            self.information_relevant_count += 1
    
    def get_success_rate(self):
        """Calculate MID extraction success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_extractions / self.total_calls
    
    def get_format_valid_rate(self):
        """Calculate format validation success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.valid_format_count / self.total_calls
    
    def should_report(self):
        """Check if it's time to generate a report."""
        return self.total_calls - self.last_report_count >= self.report_interval
    
    def reset_report_counter(self):
        """Reset the report counter."""
        self.last_report_count = self.total_calls

# Initialize global stats and logger
_mid_stats = MIDExtractionStats()
_mid_logger = _setup_mid_extraction_logger()

def _log_extraction_stats(predicted_mids, is_valid_format, sparql_correct, information_relevant, 
                         ground_truth_mids, solution_str):
    """Log MID extraction statistics."""
    global _mid_stats, _mid_logger
    
    # Update statistics
    _mid_stats.update(predicted_mids, is_valid_format, sparql_correct, information_relevant)
    
    # Log detailed extraction info
    extraction_status = "SUCCESS" if predicted_mids is not None else "FAILED"
    _mid_logger.info(f"MID_EXTRACTION | Status: {extraction_status} | "
                    f"ValidFormat: {is_valid_format} | SPARQLCorrect: {sparql_correct} | "
                    f"InfoRelevant: {information_relevant} | "
                    f"ExtractedMIDs: {predicted_mids} | GroundTruthMIDs: {ground_truth_mids}")
    
    # Generate periodic report
    if _mid_stats.should_report():
        success_rate = _mid_stats.get_success_rate()
        format_valid_rate = _mid_stats.get_format_valid_rate()
        
        report = f"""
========== MID EXTRACTION STATISTICS REPORT ==========
Total Calls: {_mid_stats.total_calls}
Successful Extractions: {_mid_stats.successful_extractions}
Failed Extractions: {_mid_stats.failed_extractions}
MID Extraction Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)

Valid Format Count: {_mid_stats.valid_format_count}
Invalid Format Count: {_mid_stats.invalid_format_count}  
Format Validation Success Rate: {format_valid_rate:.4f} ({format_valid_rate*100:.2f}%)

SPARQL Correct Count: {_mid_stats.sparql_correct_count}
Information Relevant Count: {_mid_stats.information_relevant_count}
=====================================================
        """
        
        _mid_logger.warning(report)  # Use WARNING level to ensure console output
        _mid_stats.reset_report_counter()


def is_mid_information_relevant(text: str, golden_mids: list[str]) -> bool:
    """
    Check if the information blocks contain relevant information about the golden MIDs.
    
    This function is specifically designed for MID matching and does NOT apply
    normalize_answer() processing, since MIDs are strict identifiers that should
    be matched exactly (including the dot notation like 'm.0123abc').
    
    Args:
        text: The solution text containing information blocks
        golden_mids: List of golden MID strings to search for
        
    Returns:
        bool: True if any golden MID is found in any information block
    """
    info_blocks = qa_em_format.extract_information_blocks(text)
    for info_block in info_blocks:
        for golden_mid in golden_mids:
            # Direct string matching without normalization to preserve MID format
            if golden_mid in info_block:
                return True
    return False


def compute_score_mid_em(solution_str, ground_truth, method='strict', structure_format_score=0, sparql_bonus_score=0, information_bonus_score=0, score=1.):
    """
    The scoring function for MID-based evaluation with format rewards for KBQA.
    
    This function combines:
    - MID-based evaluation from mid_reward.py (F1 score on MID lists)
    - Format checking from qa_em_format.py (KBQA sequence validation)
    - Structured rewards for proper formatting and reasoning

    Args:
        solution_str: the solution text
        ground_truth: the ground truth dictionary containing 'target' MID list and optionally 'sparql' query
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        structure_format_score: the score for correct structural format (proper tag sequence)
        sparql_bonus_score: bonus score for correct SPARQL query structure
        information_bonus_score: bonus score for relevant information extraction
        format_score: backward compatibility parameter for basic format penalty
        score: the score multiplier for the correct answer
    """
    # Check if the solution follows valid KBQA reasoning sequence format
    is_valid_format, _ = qa_em_format.is_valid_kbqa_sequence(solution_str)
    
    # Check if SPARQL query is correct (if provided in ground truth)
    sparql_correct = False
    if 'sparql' in ground_truth and ground_truth['sparql']:
        sparql_correct = qa_em_format.is_sparql_correct(solution_str, ground_truth['sparql'])
    
    # Extract predicted MIDs from the answer tag
    predicted_mids = mid_reward.extract_mid_list(solution_str)
    
    # Get ground truth MIDs - support both single and multiple answer formats
    if 'target' in ground_truth:
        # Check if target contains multiple answer candidates (WebQSP-style)
        if isinstance(ground_truth['target'], list) and ground_truth['target'] and isinstance(ground_truth['target'][0], list):
            # Multiple answer candidates: list of lists
            ground_truth_mids = ground_truth['target']
        elif isinstance(ground_truth['target'], list):
            # Single answer list
            ground_truth_mids = [str(mid).strip() for mid in ground_truth['target'] if str(mid).strip()]
        else:
            # Single answer string
            ground_truth_mids = [str(ground_truth['target']).strip()]
    else:
        ground_truth_mids = []
    
    # Check if information is relevant to the golden MIDs
    # Use MID-specific relevance check that preserves MID format (no normalization)
    information_relevant = False
    if is_valid_format and ground_truth_mids:
        # Handle both single and multiple answer formats for information relevance check
        if isinstance(ground_truth_mids[0], list) if ground_truth_mids else False:
            # Multiple answer candidates - check if any candidate is relevant
            for candidate_mids in ground_truth_mids:
                if is_mid_information_relevant(solution_str, candidate_mids):
                    information_relevant = True
                    break
        else:
            # Single answer format
            information_relevant = is_mid_information_relevant(solution_str, ground_truth_mids)
    
    # Log MID extraction statistics
    _log_extraction_stats(predicted_mids, is_valid_format, sparql_correct, 
                         information_relevant, ground_truth_mids, solution_str)
    
    # Random printing for debugging
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Golden MIDs: {ground_truth_mids}")
        print(f"Extracted MIDs: {predicted_mids}")
        print(f"Valid format: {is_valid_format}")
        print(f"SPARQL correct: {sparql_correct}")
        print(f"Information relevant: {information_relevant}")
        print(f"Solution string: {solution_str}")
    
    # Calculate bonus scores
    bonus = 0
    if sparql_correct:
        bonus += sparql_bonus_score
    if information_relevant:
        bonus += information_bonus_score
    
    # Scoring logic
    if predicted_mids is None:
        # No MIDs extracted (format error)
        if is_valid_format:
            # Good structure but no answer
            return structure_format_score + bonus
        else:
            # Bad structure and no answer - return 0
            return 0
    else:
        # MIDs were extracted, calculate F1 score
        # Handle both single and multiple answer formats
        if isinstance(ground_truth_mids[0], list) if ground_truth_mids else False:
            # Multiple answer candidates - use best F1 approach
            precision, recall, f1 = mid_reward.calculate_best_mid_f1(ground_truth_mids, predicted_mids)
        else:
            # Single answer format
            precision, recall, f1 = mid_reward.calculate_mid_prf1(ground_truth_mids, predicted_mids)
        
        if do_print:
            print(f"MID F1 Score: {f1:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
        
        if is_valid_format:
            # Good structure, use F1 score as base reward + structure bonus
            return (f1 * score) + structure_format_score + bonus
        else:
            # Bad structure but correct MIDs, apply format penalty
            return max(0, (f1 * score) - structure_format_score + bonus)


def extract_mid_list_with_validation(solution_str: str) -> tuple[list[str], bool]:
    """
    Extract MID list and validate format.
    Returns (mid_list, is_valid_format)
    """
    is_valid_format, _ = qa_em_format.is_valid_kbqa_sequence(solution_str)
    predicted_mids = mid_reward.extract_mid_list(solution_str)
    
    return predicted_mids, is_valid_format


def compute_mid_f1_with_format_bonus(predicted_mids: list[str], ground_truth_mids: list[str], 
                                   is_valid_format: bool, sparql_correct: bool, information_relevant: bool,
                                   structure_format_score: float = 0.2, sparql_bonus_score: float = 0.1,
                                   information_bonus_score: float = 0.1, score: float = 1.0) -> float:
    """
    Compute F1 score with format bonuses.
    Separated for easier testing and reuse.
    """
    if not predicted_mids:
        return 0.0
    
    precision, recall, f1 = mid_reward.calculate_mid_prf1(ground_truth_mids, predicted_mids)
    
    bonus = 0
    if is_valid_format:
        bonus += structure_format_score
    if sparql_correct:
        bonus += sparql_bonus_score  
    if information_relevant:
        bonus += information_bonus_score
    
    return (f1 * score) + bonus 