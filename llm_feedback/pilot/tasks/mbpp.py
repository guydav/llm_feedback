from typing import List, Dict, Optional, Callable
import re

import datasets
from evaluate import load
import tqdm

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)

from .base import BaseTask


MARKDOWN_PATTERN = re.compile(r"```\w*")


ALLOWED_IMPORTS = ['typing', 'collections', 'math', 're', 'heapq', 'itertools', 'sys']
DEFAULT_ADDITIONAL_GLOBALS = {
    'all': all,
    'dict': dict,
    'filter': filter,
    'map': map,
    'max': max,
    'min': min,
    'sum': sum,
    'enumerate': enumerate,
    'reversed': reversed,
    'iter': iter,
}


class MBPPTask(BaseTask):
    def __init__(self, task_args_str: Optional[str] = None, allowed_imports: List[str] = ALLOWED_IMPORTS, 
                 additional_globals: Dict[str, Callable] = DEFAULT_ADDITIONAL_GLOBALS):
        super().__init__()
        self.task_args_str = task_args_str
        self.tqdm = 'tqdm' in task_args_str if task_args_str else False

        self.allowed_imports = allowed_imports
        self.additional_globals = additional_globals
        self.code_eval = load("guydav/restrictedpython_code_eval")

    def get_dataset(self, phase: str):
        return datasets.load_dataset("mbpp", split=phase)

    def get_chain(self, generation_llm: str, feedback_llm: str, refinement_llm: str,
                  chain_name: Optional[str] = "regular"):
        # 0. Setup
        initial_llm = ChatOpenAI(model_name=generation_llm)
        feedback_llm = ChatOpenAI(model_name=feedback_llm)
        refinement_llm = ChatOpenAI(model_name=refinement_llm)

        initial_solution_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful Python coding assistant."),
            HumanMessagePromptTemplate.from_template("""
You will be given a Python programming task and one unit test. Write a function that satisfies the specification in task description and passes the unit test. Imporant: Do not include the test case in your solution! Output just the improved solution, without any additional comments. Your entire output should be ready to be copy-pasted into a Python console and run.
Instruction:
{text}
Unit test:
{test_list_0}
Solution:
            """.strip(), input_variables=["text", "test_list_0"]),
        ])

        if chain_name == "regular":
            feedback_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful Python coding assistant."),
                HumanMessagePromptTemplate.from_template("""
You will be given a Python programming task, unit tests and a candidate solution. Your job is to provide short feedback on how to improve the candidate solution such that it satisfies the specification in task description and passes the unit test. Be as concise as possible! Do not provide the corrected solution, limit yourself to short feedback in natural language. Focus on correctness, not on following Python style guide or good variable naming. Don't comment on the provided unit tests, they're fixed and not meant to be changed. Your feedback should be understandable to someone who doesn't see these unit tests. If the solution is already okay, just output \"OK\".
Instruction:
{text}

Unit tests:
{test_list_0}
{test_list_1}
{test_list_2}

Solution:
{initial_solution}
            """.strip(), input_variables=["text", "test_list_0", "test_list_1", "test_list_2" "initial_solution"]),
            ])
            refinement_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful Python coding assistant."),
                HumanMessagePromptTemplate.from_template("""
Feedback:
You will be given a Python programming task, one unit test, an initial solution and feedback an expert provided on that initial solution. Your job is to rewrite the initial solution based on the feedback. Output just the improved solution, without any additional comments. Don't include unit test in your improved solution, they are not part of the solution. Your entire output should be ready to be copy-pasted into a Python console and run.

Instruction:
{text}

Unit test:
{test_list_0}

Initial solution:
{initial_solution}

Feedback:
{feedback}
Improved solution:
            """.strip(), input_variables=[
                    "text", "test_list_0", "test_list_1", "test_list_2", "initial_solution", "feedback"
                ]),
            ])
        elif chain_name == "chat":
            feedback_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful Python coding assistant. A human will show you a Python programming task, unit tests for this task and a candidate solution that human wrote. Your job is to provide short feedback on how to improve human's candidate solution such that it satisfies the specification in task description and passes the unit test. Be as concise as possible! Do not provide the corrected solution, limit yourself to short feedback in natural language. Focus on correctness, not on following Python style guide or good variable naming. Don't comment on the provided unit tests, they're fixed and not meant to be changed. Your feedback should be understandable to someone who doesn't see these unit tests. If the solution is already okay, just output \"OK\"."),
                HumanMessagePromptTemplate.from_template("""
Here is my task:
{text}

The function should pass the following tests:
{test_list_0}
{test_list_1}
{test_list_2}

Here is my solution:
{initial_solution}

How can I improve it? Just give be a short feedback, I don't need the improved solution.
            """.strip(), input_variables=["text", "test_list_0", "test_list_1", "test_list_2" "initial_solution"]),
            ])
            refinement_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful Python coding assistant. Human will be giving Python programming tasks paired with one unit test. Your job is to write a function that satisfies the specification in task description and passes the unit test. Your replies should consist purely of the improved solution, without any additional comments. Imporant: Do not include the test case in your solution! Output just the improved solution Your entire output should be ready to be copy-pasted into a Python console and run. Human will be giving you feedback on your solution. You should use this feedback to improve your solution. Again, your output should consist purely of the improved solution, without any additional comments. Sometimes human's feedback will be just \"OK\". This means that your solution is already correct and you should repeat it verbatim."),
                HumanMessagePromptTemplate.from_template("""
{text}
        
The function should pass the following tests:
{test_list_0}
{test_list_1}
{test_list_2}
                """.strip(), input_variables=["text", "test_list_0", "test_list_1", "test_list_2"]),
                AIMessagePromptTemplate.from_template("{initial_solution}", input_variables=["initial_solution"]),
                HumanMessagePromptTemplate.from_template("{feedback}", input_variables=["feedback"]),
            ])
        else:
            raise KeyError(chain_name)

        # === 1. Initial solution === #

        initial_solution_chain = LLMChain(
            llm=initial_llm,
            prompt=initial_solution_prompt,
            output_key="initial_solution",
        )
        feedback_chain = LLMChain(llm=feedback_llm, prompt=feedback_prompt, output_key="feedback")
        refinement_chain = LLMChain(llm=refinement_llm, prompt=refinement_prompt, output_key="refinement")
        ilf_chain = SequentialChain(
            chains=[initial_solution_chain, feedback_chain, refinement_chain],
            input_variables=["text", "test_list_0", "test_list_1", "test_list_2"],
            output_variables=["initial_solution", "feedback", "refinement"],
        )
        return ilf_chain

    def process(self, chain, example):
        output = chain({
            "text": example["text"],
            # HumanMessagePromptTemplate appears to not be able to handle lists,
            # so we need to pass each element separately.
            "test_list_0": example["test_list"][0],
            "test_list_1": example["test_list"][1],
            "test_list_2": example["test_list"][2],

        })
        output["test_setup_code"] = example["test_setup_code"]
        return output

    def evaluate(self, phase: str, outputs: List[Dict]):
        all_results = []

        if self.tqdm:
            outputs = tqdm.tqdm(outputs, desc='Evaluating')  # type: ignore

        for output in outputs:
            test_cases = [output[key] for key in ('test_list_0', 'test_list_1', 'test_list_2')]
            solutions = [output['initial_solution'], output['refinement']]
            solutions = [MARKDOWN_PATTERN.sub('', solution).strip() for solution in solutions]
            solutions = [solution.replace('(object)', '') for solution in solutions]
            if output['test_setup_code']:
                solutions = ['\n'.join([output['test_setup_code'], solution]) for solution in solutions]
            
            results = self.code_eval.compute(
                references=test_cases, 
                predictions=[solutions] * len(test_cases), 
                k=[len(solutions)],
                allowed_imports=self.allowed_imports,
                additional_globals=self.additional_globals,
                timeout=60,
                allow_str_format=True,
                allow_underscore_variable_names=True,
                )[1]  # type: ignore
            
            all_results.append({**output, **results})

        return all_results
            


class MBPPTestGenerationTask(BaseTask):
    def __init__(self, task_args_str: Optional[str] = None, allowed_imports: List[str] = ALLOWED_IMPORTS, 
                 additional_globals: Dict[str, Callable] = DEFAULT_ADDITIONAL_GLOBALS):
        super().__init__()
        self.task_args_str = task_args_str
        self.tqdm = 'tqdm' in task_args_str if task_args_str else False

        self.allowed_imports = allowed_imports
        self.additional_globals = additional_globals
        self.code_eval = load("guydav/restrictedpython_code_eval")

    def get_dataset(self, phase: str):
        return datasets.load_dataset("mbpp", split=phase)

    def get_chain(self, generation_llm: str, feedback_llm: str, refinement_llm: str,
                  chain_name: Optional[str] = "regular"):
        # 0. Setup
        initial_llm = ChatOpenAI(model_name=generation_llm)  
        feedback_llm = ChatOpenAI(model_name=feedback_llm)  
        # refinement_llm = ChatOpenAI(model_name=refinement_llm)  # type: ignore

        initial_solution_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful Python coding assistant."),
            HumanMessagePromptTemplate.from_template("""
You will be given a Python programming task and one unit test. Write a function that satisfies the specification in task description and passes the unit test. Imporant: Do not include the test case in your solution! Output just the improved solution, without any additional comments. Your entire output should be ready to be copy-pasted into a Python console and run.
Instruction:
{text}
Unit test:
{test_list_0}
Solution:
            """.strip(), input_variables=["text", "test_list_0"]),
        ])

        if chain_name == "regular":
            feedback_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful Python test generator, capable of generating tests that are helpful and evaluate code coverage and edge cases."),
    HumanMessagePromptTemplate.from_template("""
You will be given a Python programming task, a candidate solution, and 3 unit tests. Please write 3 additional unit tests to test the provided soltuion and verify if it is correct. 
The tests should be as helpful as possible, and should cover as many edge cases as possible. The tests should be written in the same style as the provided tests. 
Imporant: Do not include the existing test cases in your solution! Output just the new test cases. Ensure that you use the same function and class names as in the existing tests. Each test case should be ready to copy-paste Python console and run.
Instruction:
"{text}"
Candidate solution:
```python
{initial_solution}
```
Current unit tests:
```python 
{test_list_0}
{test_list_1}
{test_list_2}
```
New unit tests:
```python
""".strip(), input_variables=["text", "test_list_0", "test_list_1", "test_list_2"]),
])
            
        else:
            raise KeyError(chain_name)

        # === 1. Initial solution === #

        initial_solution_chain = LLMChain(
            llm=initial_llm,
            prompt=initial_solution_prompt,
            output_key="initial_solution",
        )
        feedback_chain = LLMChain(llm=feedback_llm, prompt=feedback_prompt, output_key="feedback")
        # refinement_chain = LLMChain(llm=refinement_llm, prompt=refinement_prompt, output_key="refinement")
        ilf_chain = SequentialChain(
            chains=[initial_solution_chain, feedback_chain], # , refinement_chain],
            input_variables=["text", "test_list_0", "test_list_1", "test_list_2"],
            output_variables=["initial_solution", "feedback"],  # , "refinement"],
        )
        return ilf_chain

    def process(self, chain, example):
        output = chain({
            "text": example["text"],
            # HumanMessagePromptTemplate appears to not be able to handle lists,
            # so we need to pass each element separately.
            "test_list_0": example["test_list"][0],
            "test_list_1": example["test_list"][1],
            "test_list_2": example["test_list"][2],

        })
        output["gold_code"] = example["code"]
        output["test_setup_code"] = example["test_setup_code"]

        return output

    def evaluate(self, phase: str, outputs: List[Dict]):
        all_results = []

        if self.tqdm:
            outputs = tqdm.tqdm(outputs, desc='Evaluating')  # type: ignore

        for output in outputs:
            gold_tests = [output[key] for key in ('test_list_0', 'test_list_1', 'test_list_2')]
            model_test_cases = _parse_test_cases(output['feedback'])

            model_tests_set = set(model_test_cases)
            gold_tests_set = set(gold_tests)

            # TODO: decide if at any point we want a softer match
            model_tests_set.difference_update(gold_tests_set)
            model_test_cases = list(model_tests_set)

            test_cases = gold_tests + model_test_cases
            solutions = [output['initial_solution'], output['gold_code']]
            solutions = [MARKDOWN_PATTERN.sub('', solution).strip() for solution in solutions]
            solutions = [solution.replace('(object)', '') for solution in solutions]
            if output['test_setup_code']:
                solutions = ['\n'.join([output['test_setup_code'], solution]) for solution in solutions]
            
            results = self.code_eval.compute(
                references=test_cases, 
                predictions=[solutions] * len(test_cases), 
                k=[len(solutions)],
                allowed_imports=self.allowed_imports,
                additional_globals=self.additional_globals,
                timeout=60,
                allow_str_format=True,
                allow_underscore_variable_names=True,
                )[1]  # type: ignore
            
            all_results.append({**output, 'test_cases': test_cases[:3], 'model_test_cases': model_test_cases, **results})

        return all_results


def _parse_test_cases(text: str):
    current_start = 0
    text = MARKDOWN_PATTERN.sub('', text).strip()
    test_cases = []

    while True:
        next_assert = text.find('assert', current_start)
        next_linebreak = text.find('\n', next_assert)
        if next_linebreak == -1:
            next_linebreak = len(text)

        test_cases.append(text[current_start:next_linebreak].strip())

        if next_linebreak == len(text):
            break

        current_start = next_linebreak + 1

    return test_cases
