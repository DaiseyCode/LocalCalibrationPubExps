import os
import re
from numpy import mat
from synthegrator.lang_specs.lang_spec_python import PythonLangSpec
# LAZY IMPORT: transformers is now imported only when needed in get_or_load_tokenizer()
from synthegrator.synthdatasets import DatasetName, DatasetSpec
from pprint import pprint
from synthegrator.lang_specs.lang_spec_python import LsFunctionDefPython

os.environ["TOKENIZERS_PARALLELISM"] = "false"

lang_spec = PythonLangSpec()


#def strip_non_semantic(text: str):
#    pass


def solve_to_text(solve, dataset: DatasetSpec) -> str:
    if dataset.get_base_collection_name() in (
        DatasetName.humaneval.get_base_collection_name(), 
        DatasetName.humaneval_plus.get_base_collection_name(), 
        DatasetName.mbpp.get_base_collection_name(), 
        DatasetName.mbpp_plus.get_base_collection_name(),
        DatasetName.livecodebench.get_base_collection_name(),
        DatasetName.repocod.get_base_collection_name(),
    ):
        try:
            text = solve.apply().get_only_file(only_consider_dirty = True).content_str
        except ValueError as e:
            if "spaces" in str(e):
                return None
            raise
        except TypeError as e:
            print("TypeError")
            print(e)
            print("Solve steps")
            pprint(solve.solve_steps)
            raise

        funcs = PythonLangSpec.find_functions(text)
        matching_funcs = []
        for func in funcs:
            if solve.solve_steps[0].value.strip() in func.get_full_function_src():
                return func.get_body_src(include_prefix_comment=False).strip("\n")
        if dataset.get_base_collection_name() == DatasetName.repocod.get_base_collection_name():
            return solve.solve_steps[0].value.strip()
            print("--- TEXT")
            print(text)
            print("--- Solve step")
            print(solve.solve_steps[0].value.strip())
            print("--- Problem id")
            print(solve.problem.problem_id)
            # Save the text to a file
            with open("repocod_answer.txt", "w") as f:
                f.write(text)
            # Save the solve step to a file
            with open("repocod_solve_step.txt", "w") as f:
                f.write(solve.solve_steps[0].value.strip())
            # Save the problem id to a file
            raise ValueError(f"No function found in repocod answer")

        # Get the reference solution to get the reference function name we are supposed to be solving
        number_of_first_level_funcs = text.count("\ndef")
        if solve.problem.known_solutions:
            ref_text = solve.problem.known_solutions[0].apply().get_only_file().content_str
            ref_text_func, ref_func_name = get_func_body(
                ref_text,
                func_name_or_index=-1 if number_of_first_level_funcs > 1 else 0,   # Unless just one func
                must_have_doc_str=False,
            )
            if ref_text_func is None:
                print("ref_text_func is None")
                print(ref_text)
                raise ValueError(f"No function found")
            try:
                text_func, func_name = get_func_body(text, ref_func_name)
            except ValueError as e:
                print("known sol")
                print(solve.problem.known_solutions[0].apply().get_only_file().content_str)
                return None
        else:
            text_func, func_name = get_func_body(text, func_name_or_index=0)
            if text_func is None:
                #raise ValueError(f"No function found in\n{text}")
                text_func = text
        text_func = text_func.lstrip("\n")
        #assert '"""' not in text_func[:min(100, len(text_func))]
        return text_func
    elif dataset == DatasetName.dypy_line_completion:
        text = solve.solve_steps[0].value
        text = text.rstrip()  # Filter out the new line character at the end
        return text
    else:
        raise ValueError(f"Unknown dataset {dataset}")


def get_func_body(
    text: str,
    func_name_or_index: str | int,
    must_have_doc_str: bool = False,
):
    """Gets the function body from the given func_name in the given python source text"""
    funcs = list(lang_spec.find_functions(text, include_nested_functions=True))
    assert isinstance(func_name_or_index, str) or isinstance(func_name_or_index, int), f"{type(func_name_or_index)=}"

    if must_have_doc_str:
        funcs = [func for func in funcs if func.get_prefix_comment_str() is not None]

    if len(funcs) == 0:
        print("no funcs found. Possibly unparsable code")
        return None, None

    def select_func(funcs):
        if not isinstance(func_name_or_index, str):
            return funcs[func_name_or_index]
        for func in funcs:
            if func.get_function_name() == func_name_or_index:
                return func

    func = select_func(funcs)
    if func is None:
        print("text")
        print(text)
        raise ValueError(f"Function {func_name_or_index} not found in text")
    body = func.get_body_src()
    doc_str = func.get_prefix_comment_str()
    if doc_str is not None:
        if body.strip().startswith(doc_str):
            body = body.strip()[len(doc_str):]
    return body, func.get_function_name()


_tokenizers = {}

def get_or_load_tokenizer(key: str):
    global _tokenizers
    if key in _tokenizers:
        return _tokenizers[key]
    
    # LAZY IMPORT: Only import transformers when we actually need it
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(key)
    _tokenizers[key] = tokenizer
    return tokenizer


def tokenize_text(key: str, text: str) -> list[str]:
    #return tokenize_heuristics(text)
    tokenizer = get_or_load_tokenizer(key)
    assert isinstance(text, str)
    if (
        "mistral" in key
        or "llama" in key
    ):
        return tokenize_llama(text, tokenizer)
    elif "Qwen" in key:
        return tokenize_qwen(text, tokenizer)
    else:
        raise NotImplementedError


def tokenize_qwen(text: str, tokenizer) -> list[str]:
    return [
        token.replace("Ġ", " ").replace("Ċ", "\n")
        for token in tokenizer.tokenize(text, add_special_tokens=False)
    ]


def tokenize_heuristics(text: str):
    """
    A heuristic tokenizer for splitting Python code tokens.
    It keeps symbols, literals (numbers, short strings), operators, etc.,
    and splits at spaces, brackets, and other delimiters.
    """
    # Define regex patterns for various token types
    token_patterns = [
        r'\s*\".*?\"|\'.*?\'',  # Match string literals
        r'\s*\d+\.\d+|\d+',  # Match numbers (integers or floats)
        r'\s*[A-Za-z_]\w*',  # Match identifiers (variable/function names)
        r'\s*[+\-*/%=&|^~<>!]=?',  # Match operators
        r'\s*[()\[\]{},.:;]',  # Match punctuation
        r'\s+',  # Match whitespace
    ]

    # Combine regex patterns into one
    combined_pattern = '|'.join(f'({p})' for p in token_patterns)

    # Find all matches in the text
    matches = re.finditer(combined_pattern, text)

    # Extract tokens, skipping whitespace matches
    tokens = [
        match.group()
        for match in matches
        #if not match.group().isspace()
    ]

    return tokens


def tokenize_llama(text, gen_tokenizer) -> list[str]:
    return [
        token.replace("▁", " ").replace("<0x0A>", "\n")
        for token in gen_tokenizer.tokenize(
            text,
            add_special_tokens=False,
        )
    ]


if __name__ == "__main__":
    code = """
    max_so_far = numbers[0]
    result = [max_so_far]
    for i in range(1, len(numbers)):
        max_so_far = max(numbers[i], max_so_far)
        result.append(max_so_far)
    return result
    """
    print(tokenize_heuristics(code))


