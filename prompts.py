def _get_system_message(args):
    enforce_qwen = not getattr(args, "do_not_enforce_qwen", False)
    if enforce_qwen:
        assert "qwen" in getattr(args, "model_name", "").lower(), \
            "Qwen-specific prompt; use --do_not_enforce_qwen to allow non-Qwen models."
        return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    return "You are a helpful assistant."


def build_agent_messages_single_agent(question: str, args=None):
    """Build prompt for baseline single-agent method."""
    system_message = _get_system_message(args)
    task = args.task

    if task in ["gsm8k", "aime2024", "aime2025"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["mbppplus", "humanevalplus"]:
        user_content = f"""
Target Question: {question}

You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
Now, reason step by step and output the final answer:
"""

    elif task in ["winogrande"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    else:
        user_content = f"""
Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the question without outputting other irrelevant information.
Present your reasoning, and then clearly state your final answer at the end.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def _resolve_system_message(args, custom_system: str = None):
    if custom_system is not None:
        return custom_system if custom_system != "" else None
    return _get_system_message(args)


def _render_custom_prompt(role: str, question: str, context: str, args):
    prompts_cfg = getattr(args, "custom_prompts", None)
    if not isinstance(prompts_cfg, dict):
        return None, None
    custom_system = prompts_cfg.get("system", None)
    template = prompts_cfg.get(role)
    if template is None:
        return custom_system, None
    try:
        user_content = template.format(question=question, context=context)
    except Exception:
        user_content = template
    return custom_system, user_content


def build_agent_message_sequential_latent_mas(role: str, question: str, context: str = "", method=None, args=None):

    custom_system, user_prompt = _render_custom_prompt(role, question, context, args)
    system_message = _resolve_system_message(args, custom_system)

    if args.custom_prompt_file or user_prompt is None:

        if role == "planner":
            user_prompt = f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {question}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:
"""

        elif role == "judger":
            if args.task in ['gsm8k', 'aime2024', 'aime2025']:
                user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

            elif args.task in ["arc_easy", "arc_challenge", "gpqa", 'medqa']:
                user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

            elif args.task in ["mbppplus", "humanevalplus"]:
                user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve.

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block.

Now, reason step by step and output the final answer inside ```python
YOUR_PYTHON_CODE
```.
"""

            elif args.task in ["winogrande"]:
                user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

            else:
                raise NotImplementedError(f"Task {args.task} not implemented in judger prompt.")

        if user_prompt is None:
            user_prompt = f"""
Question: {question}

You are a helpful assistant. Reason step-by-step and provide a clear, concise response.
"""

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_agent_message_hierarchical_latent_mas(role: str, question: str, context: str = "", method=None, args=None):

    custom_system, user_content = _render_custom_prompt(role, question, context, args)
    system_message = _resolve_system_message(args, custom_system)

    if args.custom_prompt_file or user_content is None:

        if args.task in ['gsm8k', 'aime2024', 'aime2025']:
            if role == "math":
                user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
            elif role == "science":
                user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
            elif role == "code":
                user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
            elif role == "summarizer":
                user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""

        elif args.task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:

            if args.task == "medqa":
                if role == "math":
                    user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:
"""
                elif role == "science":
                    user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}     

Your response:
"""
                elif role == "code":
                    user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:       
"""
                elif role == "summarizer":
                    user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:
"""

            else:
                if role == "math":
                    user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:
"""
                elif role == "science":
                    user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}     

Your response:
"""
                elif role == "code":
                    user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:       
"""
                elif role == "summarizer":
                    user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:
"""

        elif args.task in ["mbppplus", "humanevalplus"]:
            if role == "math":
                user_content = f"""
You are a math agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
            elif role == "science":
                user_content = f"""
You are a science agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
            elif role == "code":
                user_content = f"""
You are a code agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:       
"""
            elif role == "summarizer":
                user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 
    
Input Question: {question}

Your response:
"""

        elif args.task in ["winogrande"]:
            if role == "math":
                user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""
            elif role == "science":
                user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}     

Your response:
"""
            elif role == "code":
                user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:       
"""
            elif role == "summarizer":
                user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""

        if user_content is None:
            user_content = f"""
Question: {question}

You are a helpful assistant. Reason step-by-step and provide a clear, concise response.
"""

    return [
        *([{"role": "system", "content": system_message}] if system_message else []),
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_sequential_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    ctx = context[: args.text_mas_context_length]

    custom_system, user_content = _render_custom_prompt(role, question, ctx, args)
    system_message = _resolve_system_message(args, custom_system)
    
    if args.custom_prompt_file or user_content is None:

        if role == "planner":
            user_content = f"""
You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

## Input Question:
{question}

Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

## Format your response as follows:
Planner Agent's Output:
[Your detailed plan here]

Now output your plan to solve the question below:
"""

        elif role == "judger":
            task = getattr(args, "task", None)

            if task in ["gsm8k", "aime2024", "aime2025"]:
                user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> solver).
You are provided with the Agent's plan as reference.

Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

            elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
                user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> solver).
You are provided with the Agent's plan as reference.

Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

            elif task in ["mbppplus", "humanevalplus"]:
                user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> solver).
You are provided with the Agent's plan as reference.

Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
"""

            elif task in ["winogrande"]:
                user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> solver).
You are provided with the Agent's plan as reference.

Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
            else:
                user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> solver).
You are provided with the Agent's plan as reference.

Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and present your final answer clearly at the end.
"""

        if user_content is None:
            user_content = f"""
Question: {question}

Context:
{ctx}

You are a helpful assistant. Reason step-by-step and provide a clear, concise response.
"""

    return [
        *([{"role": "system", "content": system_message}] if system_message else []),
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_hierarchical_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    ctx = context[: args.text_mas_context_length]
    custom_system, user_content = _render_custom_prompt(role, question, ctx, args)
    system_message = _resolve_system_message(args, custom_system)

    if args.custom_prompt_file or user_content is None:

        if args.task in ['gsm8k', 'aime2024', 'aime2025']:
            if role == "math":
                user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
            elif role == "science":
                user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
            elif role == "code":
                user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
            elif role == "summarizer":
                user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{ctx}

Input Question: {question}

Your response:
"""

        elif args.task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
            if role == "math":
                user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
            elif role == "science":
                user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
            elif role == "code":
                user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
            elif role == "summarizer":
                user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{ctx}

Input Question: {question}

Your response:
"""

        elif args.task in ["mbppplus", "humanevalplus"]:
            if role == "math":
                user_content = f"""
You are a math agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
            elif role == "science":
                user_content = f"""
You are a science agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
            elif role == "code":
                user_content = f"""
You are a code agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
            elif role == "summarizer":
                user_content = f"""
You are a task summarizer. Given the final answer in markdown python code block.

Content from Previous Agent:
{ctx}

Input Question: {question}

Your response:
"""

        elif args.task in ["winogrande"]:
            if role == "math":
                user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""
            elif role == "science":
                user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}     

Your response:
"""
            elif role == "code":
                user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:       
"""
            elif role == "summarizer":
                user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{ctx}

"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""

        if user_content is None:
            user_content = f"""
Question: {question}

Context:
{ctx}

You are a helpful assistant. Reason step-by-step and provide a clear, concise response.
"""

    return [
        *([{"role": "system", "content": system_message}] if system_message else []),
        {"role": "user", "content": user_content},
    ]
