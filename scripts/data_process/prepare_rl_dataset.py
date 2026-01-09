import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import datasets
import pandas as pd
from tqdm import tqdm

# Import necessary functions
try:
    from .webqsp_sparql import get_label_with_odbc_safe
except ImportError:
    # Handle case when run as script
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    from webqsp_sparql import get_label_with_odbc_safe

def ent_type(ent_id: str) -> str:
    """Determine the type of entity ID"""
    if ent_id.startswith('m.') or ent_id.startswith('g.'):
        return 'entity'
    elif ent_id.startswith('"') and (ent_id.endswith('"') or ent_id.endswith('"@en')):
        return 'name'
    elif '^^' in ent_id:
        return 'literal'
    elif '<http:' in ent_id:
        return 'url'
    elif ent_id.isdigit() or (ent_id[0] == '-' and ent_id[1:].isdigit()):
        return 'int'
    elif '.' in ent_id:
        return 'onto'
    else:
        return 'unknown'

def load_existing_entities_cache(output_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Load existing dataset and extract entity mappings to avoid re-using ODBC"""
    entity_cache = {}    
    if not os.path.exists(output_path):
        print(f"Output file {output_path} does not exist, will use ODBC if needed.")
        return entity_cache
        
    try:
        print(f"Loading existing file {output_path} to check for cached entities...")
        df = pd.read_parquet(output_path)
        
        cached_count = 0
        for _, row in df.iterrows():
            extra_info = row.get('extra_info', {})
            if isinstance(extra_info, str):
                try:
                    extra_info = json.loads(extra_info)
                except (json.JSONDecodeError, ValueError):
                    continue
            
            # Use question or ID as cache key
            question = extra_info.get('original_question', '')
            item_id = extra_info.get('id', '')
            extracted_entities = extra_info.get('extracted_entities', [])
            
            if extracted_entities is not None:
                # Use question as primary key, ID as fallback
                cache_key = question if question else item_id
                if cache_key:
                    entity_cache[cache_key] = extracted_entities
                    cached_count += 1
        
        print(f"Loaded {cached_count} cached entity mappings from existing file.")
        return entity_cache
        
    except Exception as e:
        print(f"Error loading existing file {output_path}: {e}")
        print("Will proceed without entity cache.")
        return {}

# Import prompt_function from webqsp_sparql.py to extract entities
def prompt_function(function_list: List[str], use_odbc: bool = False, entity_cache: Dict[str, List[Tuple[str, str]]] = None, cache_key: str = None): #! very important function
    scratchpad = ''
    entities = []
    
    # Check if we can use cached entities
    if entity_cache and cache_key and cache_key in entity_cache:
        print(f"Using cached entities for key: {cache_key[:50]}...")
        cached_entities = entity_cache[cache_key]
        use_odbc = False  # Force disable ODBC when using cache
    else:
        cached_entities = None
    
    for step_n, func in enumerate(function_list):
        step_n += 1
        if "START" in func:
            argument = re.findall(r"expression(.*?) = START\('(.*?)'\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should identify a topic entity from the question to start a new expression.\nAction{step_n}: Extract_entity '
            
            entity_id = argument[1]
            # Try to get entity label from cache first
            if cached_entities is not None:
                # Find entity by ID in cached entities
                entity_label = None
                for cached_name, cached_id in cached_entities:
                    if cached_id == entity_id:
                        entity_label = cached_name
                        break
                
                if entity_label:
                    entity = entity_label
                else:
                    # Fallback: handle different entity types
                    entity_type = ent_type(entity_id)
                    if entity_type == 'onto':
                        # Convert ontology types to readable format
                        entity = entity_id.replace('.', ' ').replace('_', ' ').title()
                    else:
                        entity = entity_id
            else:
                # Use original ODBC logic with proper entity type handling
                entity_type = ent_type(entity_id)
                if entity_type == 'entity':
                    # Regular entities (m.*, g.*)
                    entity = get_label_with_odbc_safe(entity_id, use_odbc)
                elif entity_type == 'onto':
                    # Ontology types (e.g., measurement_unit.conductance_unit)
                    # Convert dot notation to more readable format
                    entity = entity_id.replace('.', ' ').replace('_', ' ').title()
                else:
                    # Other types (literals, names, etc.)
                    entity = entity_id
            
            action = f'[ {entity} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = START('{entity_id}')\n" # | Start Excuted Answers: {entity} (total 1 answers)
            scratchpad += f'{plan}{action}{observation}'
            entities.append((entity, entity_id))
        elif "JOIN(" in func:
            argument = re.findall(r"expression(.*?) = JOIN\('(.*?)', expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should find the one-hop relation that is connected to the current expression.\nAction{step_n}: Find_relation '
            relation = argument[1] if '(R ' not in argument[1] else re.findall(r"\(R (.*?)\)", argument[1])[0]
            action = f'[ {relation} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = JOIN('{argument[1]}', expression{argument[2]})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "AND" in func:
            argument = re.findall(r"expression(.*?) = AND\(expression(.*?), expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should merge these two expressions.\nAction{step_n}: Merge '
            expression1, expression2 = f'expression{argument[1]}', f'expression{argument[2]}'
            action = f'[ {expression1} | {expression2} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = AND({expression1}, {expression2})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "ARG" in func:
            argument = re.findall(r"expression(.*?) = ARG\('(.*?)', expression(.*?), '(.*?)'\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should perform a sorting operation and impose a constraint to output either the maximum or minimum value.\nAction{step_n}: Order '
            mode, relation = argument[1], argument[3]
            action = f'[ {mode} | {relation} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = ARG('{mode}', expression{argument[2]}, '{relation}')\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'                
        elif "CMP" in func:
            argument = re.findall(r"expression(.*?) = CMP\('(.*?)', '(.*?)', expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should perform a numerical comparison to determine the range.\nAction{step_n}: Compare '
            mode_dict = {'le':'LESS EQUAL', 'ge':'GREATER EQUAL', 'lt':'LESS THAN', 'gt':'GREATER THAN'}
            mode, relation = mode_dict[argument[1]], argument[2]
            action = f'[ {mode} | {relation} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = CMP('{argument[1]}', '{relation}', expression{argument[3]})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "TC" in func:
            argument = re.findall(r"expression(.*?) = TC\(expression(.*?), '(.*?)', '(.*?)'\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should add a time constraint.\nAction{step_n}: Time_constraint '
            relation, time = argument[2], argument[3]
            action = f'[ {relation} | {time} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = TC(expression{argument[1]}, '{relation}', '{time}')\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
            if time != 'NOW':
                entities.append((time, argument[3]))
        elif "COUNT" in func:
            argument = re.findall(r"expression(.*?) = COUNT\(expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should perform a counting operation to determine the number of answers.\nAction{step_n}: Count '
            expression = f'expression{argument[1]}'
            action = f'[ {expression} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = COUNT({expression})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "STOP" in func:
            argument = re.findall(r"expression(.*?) = STOP\(expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we conclude that it is appropriate to end and output the expression.\nAction{step_n}: Finish '
            expression = f'expression{argument[1]}'
            action = f'[ {expression} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = STOP({expression})\n" # | Final Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        else:
            pass
    # print(scratchpad)
    return scratchpad, entities



def create_action_training_prompt(question: str, entities: List[Tuple[str, str]] = None, dataset_name: str = "webqsp") -> str:
    """Create prompt for Action-based reasoning model training following KBQA-o1 format

    Dataset-specific constraints:
    - webqsp: exclude Compare and Count actions
    - grailqa/graphq: exclude Time_constraint action
    """
    
    entities_str = ""
    if entities:
        entities_parts = [f"'{name}' ({mid})" for name, mid in entities[:10]]
        entities_display = ", ".join(entities_parts)
        if len(entities) > 50:
            entities_display += ", ..."
        entities_str = f"Candidate Entities: [{entities_display}]\n"
    else:
        entities_str = "Entities: []\n"
    
    # Assemble dataset-specific Available Actions
    actions_sections: List[str] = []

    # Find_relation (always)
    actions_sections.append(
        """* Find_relation [entity_id | relation]: Find entities connected to the given entity through the specified relation. The entity can be:
  - A MID (e.g., m.02mjmr) for entity types
  - A literal value (e.g., 2011-01-17^^http://www.w3.org/2001/XMLSchema#date, 62.0^^http://www.w3.org/2001/XMLSchema#float) for literal types
  - An expression reference for chaining operations. Use it when the referenced expression has many results.
  The relation should be a Freebase relation (e.g., people.person.place_of_birth, government.election_poll.poll_end_date). This will return a list of entities connected by the relation. 
  Examples: 
  - Find_relation [m.02mjmr | people.person.place_of_birth]
  - Find_relation [62.0^^http://www.w3.org/2001/XMLSchema#float | architecture.lighthouse.focal_height_of_light]
  - Find_relation [expression_id | media_common.quotation.author] (for chaining operations). 
  Chain example: for the query "Who authored quotations about Barack Obama?", you can chain operations: Find_relation [m.02mjmr | media_common.quotation.subjects] to get quotations about Obama, then Find_relation [expression1 | media_common.quotation.author] to get the authors of those quotations."""
    )

    # Merge (always)
    actions_sections.append(
        """* Merge [expression_id1 | expression_id2]: Merge two expressions using logical AND. This can be used in two scenarios:
  1. Merge two expressions from two early actions and get the intersection.(e.g., Merge [expression_id1 | expression_id2], expression_id1 and expression_id2 are the function_id of two previous actions)
  2. Merge an expression with an ontology type (e.g., Merge [expression_id | measurement_unit.conductance_unit])
Note: When using the merge operation, carefully verify the expression IDs to ensure you are merging the correct ones."""
    )

    # Order (always)
    actions_sections.append(
        """* Order [MAX/MIN | expression_id/ontology_type | relation]: Sort by relation and return max/min
Use example: 
  1. Order a existing expression: "When did Manchester United first win a trophy?", you should first get all of the team's championships with Find_relation [m.050fh | sports.sports_team.championships]. Then, apply the ordering constraint using Order [MIN | expression1 | time.event.end_date], which finds the championship with the earliest end date (i.e., the first trophy won).
  2. Order a ontology type: "which film director is the heaviest?", you should do the action  Order [MAX | film.director | people.person.weight_kg] to get the heaviest film director."""
    )

    is_webqsp = (dataset_name.lower() == "webqsp")
    is_grail_or_graphq = (dataset_name.lower() in ["grailqa", "graphq"])

    # Compare (exclude for webqsp)
    if not is_webqsp:
        actions_sections.append(
            """* Compare [operator | relation | number]: Apply numerical comparison. Available mode: le, lt, ge, gt.
Use example: for the query "what unit of conductance has equal to or smaller than 0.001 conductance in siemens?", you should apply the numerical comparison using Compare [ le | measurement_unit.conductance_unit.conductance_in_siemens | 0.001], which filters entities to keep only those with conductance values ≤ 0.001. Then, merge with the ontology type using Merge [expression1 | measurement_unit.conductance_unit] to ensure results are of the correct conductance unit type."""
        )

    # Time_constraint (exclude for grailqa/graphq)
    if not is_grail_or_graphq:
        actions_sections.append(
            """* Time_constraint [relation | time]: Filters a list of CVT entities, keeping only those where the <time> falls within the CVT's time range. Note that the relation for time_constraint should be of the type *.*.from, *.*.to, *.*.end_date. If the time constraint is now, use NOW as time parameter.
Use example: for the query "What team did Kaká play for in 2009?", you should first get all of the player's team memberships (CVTs) with Find_relation [m.04qv66 | sports.pro_athlete.teams]. Then, apply the time constraint using Time_constraint [sports.sports_team_roster.from | 2009], which returns a new expression representing the filtered results. Finally, use Find_relation on this expression to find the exact team he played for: Find_relation [expression2 | sports.sports_team_roster.team]."""
        )

    # Count (exclude for webqsp)
    if not is_webqsp:
        actions_sections.append("""* Count [expression]: Count results"""
        )

    available_actions_block = "\n\n".join(actions_sections)

    # Use KBQA-o1 format for action-based reasoning
    prompt = f"""You are an expert assistant for querying the Freebase knowledge base using structured reasoning actions.

Answer the given question about Freebase knowledge base. \
You **must** conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, provide structured actions inside <action> and </action> tags that will be executed to query the knowledge base. \
The system will return query results between <information> and </information>. \
You can query as many times as you want. And you need to answer the question with the appropriate identifier or value (MID for entities like m.02mjmr for Barack Obama, or literal values like dates, numbers, etc.).\
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, e.g. <answer> m.*** </answer> or <answer> 2012-02-07 </answer>, without detailed illustrations. For multiple answers, use space to separate them, e.g. <answer> m.01428y m.04ygk0 </answer>.

Available Actions (use exact format):

{available_actions_block}

Begin from the candidate entities detected in the question (you must start from one of these entities):
{entities_str}


Question: {question}
"""
    
    return prompt


def create_action_response_from_function_list(function_list: List[str], sexpr: str, answer: List[str], use_odbc: bool = False, entity_cache: Dict[str, List[Tuple[str, str]]] = None, cache_key: str = None) -> str:
    """Create structured training response with Action sequences following KBQA-o1 format"""
    
    action_scratchpad, entities = prompt_function(function_list, use_odbc=use_odbc, entity_cache=entity_cache, cache_key=cache_key)
    
    # 提取纯Action序列（去掉Thought和Observation）
    action_lines = []
    for line in action_scratchpad.split('\n'):
        if line.strip().startswith('Action'):
            action_lines.append(line.strip())
    
    # Format answer
    if isinstance(answer, list):
        answer_str = answer[0] if len(answer) == 1 else ", ".join(answer)
    else:
        answer_str = str(answer)
    
    # Create structured response with <action> tags for easy parsing
    response = f"""<think>
Let me analyze this question step by step and determine the appropriate actions to find the answer.
</think>

<action>
{chr(10).join(action_lines)}
</action>

<answer>
{answer_str}
</answer>"""
    
    return response

def process_sexpr_data(data: List[Dict], output_dir: str, training_mode: str = "action", use_odbc: bool = False, dataset_name: str = "webqsp", entity_cache: Dict[str, List[Tuple[str, str]]] = None):
    """Process KBQA-o1 format data"""
    
    processed_data = []
    
    for item in tqdm(data, desc="Processing KBQA-o1 data"):
        question = item.get('question', '')
        if question and not question.endswith('?'):
            question += '?'
        
        sexpr = item.get('sexpr', '')
        function_list = item.get('function_list', [])
        answer = item.get('answer', [])
        
        # Create cache key for entity lookup
        cache_key = question if question else item.get('ID', '')
        
        # Extract entities using KBQA-o1's method with caching
        _, entities = prompt_function(function_list, use_odbc=use_odbc, entity_cache=entity_cache, cache_key=cache_key)
        
        if training_mode == "action":
            # Action-based reasoning training (following KBQA-o1 format)
            prompt_content = create_action_training_prompt(question, entities, dataset_name)
            response = create_action_response_from_function_list(function_list, sexpr, answer, use_odbc, entity_cache, cache_key)
            
            data_item = {
                "data_source": dataset_name,
                "prompt": [{"role": "user", "content": prompt_content}],
                "response": response,  # Add expected response for SFT
                "ability": "kbqa-action-generation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "sexpr": sexpr,
                        "target": answer,
                        "sparql": item.get('sparql', ''),
                        "function_list": function_list,
                        "candidate_entities": entities
                    }
                },
                "extra_info": {
                    'id': item.get('ID', ''),
                    'function_list': function_list,
                    'extracted_entities': entities,
                    'original_question': question,
                    'level': item.get('level', None),  # GrailQA specific
                    'dataset_type': dataset_name,
                    'sparql_conversion_type': dataset_name  # For SPARQL converter optimization
                }
            }
        
        else:  # sparql mode (original)
            raise NotImplementedError("SPARQL mode is not implemented yet")
        processed_data.append(data_item)
    
    return processed_data

def process_original_webqsp_data(data: List[Dict], output_dir: str):
    """Process original WebQSP format data (existing functionality)"""
    # Use existing webqsp_sparql.py logic here
    # ... (existing implementation)
    pass

def main():
    parser = argparse.ArgumentParser(description='Process train.json files with enhanced entity extraction for KBQA-R1')
    parser.add_argument('--data_path', type=str, default='dataset/WebQSP/processed/WebQSP_train.json', 
                        help='Path to input JSON data file')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for parquet files (auto-generated based on dataset if not specified)')
    parser.add_argument('--template_type', type=str, default='base', 
                        help='Template type for prompt generation')
    parser.add_argument('--use_odbc', action='store_true', 
                        help='Use ODBC to get entity labels (requires ODBC setup)')
    parser.add_argument('--dataset', type=str, choices=['webqsp', 'grailqa', 'graphq'], default='webqsp',
                        help='Dataset type (webqsp, grailqa, or graphq)')
    parser.add_argument('--original_data_path', type=str, default='dataset/WebQSP/origin/WebQSP.train.json', 
                        help='Path to original WebQSP data (deprecated)')
    parser.add_argument('--auto_detect', action='store_true', default=True,
                        help='Auto-detect data format')
    
    args = parser.parse_args()
    data_format = "sexpr"
    
    # Load data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    # Detect format

    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = "webqsp"
    
    # Auto-generate output_dir based on dataset if not specified
    if args.output_dir is None:
        args.output_dir = f'data/{dataset_name}_rl_dataset'
        print(f"Output directory auto-generated: {args.output_dir}")
    
    print(f"Processing {dataset_name} dataset in {data_format} format")
    
    # Load entity cache from existing output file if available
    split = 'train' if 'train' in args.data_path else 'test'
    output_path = os.path.join(args.output_dir, f'{split}.parquet')
    entity_cache = load_existing_entities_cache(output_path)
    
    # If we have cached entities and not explicitly forcing ODBC, disable ODBC
    if entity_cache and not args.use_odbc:
        print("Found cached entities, will skip ODBC calls for cached items.")
        use_odbc_final = False
    else:
        use_odbc_final = args.use_odbc
        if args.use_odbc:
            print("ODBC explicitly enabled, will use ODBC for new/missing entities.")
    
    # Process based on format
    if dataset_name in ["webqsp", "grailqa", "graphq"]:
        processed_data = process_sexpr_data(data, args.output_dir, "action", use_odbc_final, dataset_name, entity_cache)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = datasets.Dataset.from_list(processed_data)
    
    dataset.to_parquet(output_path)
    
    print(f"Processed {len(processed_data)} items")
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
