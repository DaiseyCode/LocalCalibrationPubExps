from typing import Sequence
from sympy import LM
from synthegrator.problem_rendering import (
    LmPromptRenderSingleEdit,
    PromptRenderer,
    PromptRendererSingleEditGeneric,
)
from synthegrator.code_solver import LmCodeSolverAutoRegressive
from synthegrator.code_problems import CodeProblem
from synthegrator.environments import ProjectWorkingDirectory
from synthegrator.few_shotting import FewShotConfig
from synthegrator.problem_rendering import LmPromptRender, PromptRenderer
from synthegrator.prompting_test_case_selection import (
    PromptingTestCaseSelectionStrategy,
)
from synthegrator.response_parser import (
    ResponseParser,
    format_return_val_for_node,
)
from synthegrator.synthdatasets import DatasetName
from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.structs import LmPrediction, LmPrompt
from lxml import etree
from enum import Enum, auto
import re

from protogrator import make_solver_partial


def elide_function_definitions(
    text: str,
    keep_body_functions: Sequence[str] = None,
) -> str:
    if keep_body_functions is None:
        keep_body_functions = set()
    elif not isinstance(keep_body_functions, set):
        keep_body_functions = set(keep_body_functions)
    
    def consume_docstring(text: str) -> tuple[str, str]:
        #print("consume_docstring", repr(text))
        # Match docstring pattern: whitespace + triple quotes + content + closing triple quotes + optional newline
        # This handles both """ and ''' style docstrings
        pattern = r'^(\s*)("""|\'\'\')(.*?)\2(\n?)'
        match = re.match(pattern, text, re.DOTALL | re.MULTILINE)
        
        if match:
            # Return the matched docstring and the remaining text
            docstring = match.group(0)
            remaining = text[len(docstring):]
            return docstring, remaining
        else:
            # No docstring found, return empty docstring and original text
            return "", text

    def consume_body_after_docstring(text: str, indent_level: int) -> tuple[str, str]:
        #print("consume_body_after_docstring", repr(text), indent_level)
        if text.strip() == "":
            return text, ""

        lines = text.split('\n')
        split_index = len(lines)  # Default to consuming all lines
        
        for i, line in enumerate(lines):
            # Skip empty lines - they don't affect indentation logic
            if line.strip() == "":
                continue
                
            # Check the indentation level of this line
            current_indent = len(line) - len(line.lstrip())
            
            # If indentation is less than expected, we've reached the end of the function body
            if current_indent < indent_level:
                split_index = i
                break
        
        # Split at the determined index
        consumed_lines = lines[:split_index]
        remaining_lines = lines[split_index:]

        #print("split_index", split_index)
        #print("consumed_lines", repr(consumed_lines))
        #print("remaining_lines", repr(remaining_lines))
        
        # Process consumed lines: replace with "..." but keep trailing whitespace-only lines
        if consumed_lines:
            # Find trailing whitespace-only lines
            trailing_whitespace = []
            last_content_index = len(consumed_lines)
            
            for line in reversed(consumed_lines):
                if line.strip() == "":
                    trailing_whitespace.append(line)
                else:
                    break
            trailing_whitespace.reverse()
            
            # Create the elided version: properly indented "..." + trailing whitespace
            #print("trailing_whitespace", repr(trailing_whitespace))
            elided_lines = [" " * indent_level + "..."] + trailing_whitespace
            consumed_lines = elided_lines
        
        #print("after consumed_lines", repr(consumed_lines))
        #print("afterremaining_lines", repr(remaining_lines))
        remaining = "\n".join(remaining_lines)
        if remaining and consumed_lines:
            remaining = "\n" + remaining
        return '\n'.join(consumed_lines), remaining

    def consume_empty_line(text: str) -> tuple[str, str]:
        #print("consume_empty_line", repr(text))
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip() != "":
                break
        return "\n".join(lines[:i]), "\n".join(lines[i:])
        
    def consume_function_body_start(text: str) -> tuple[str, str]:
        empty_line, after_empty_line = consume_empty_line(text)
        doc_str, after_doc_str = consume_docstring(after_empty_line)
        indent_level = max(
            len(doc_str) - len(doc_str.lstrip()),
            len(after_doc_str) - len(after_doc_str.lstrip()),
        )
        body_str, after_body_str = consume_body_after_docstring(after_doc_str, indent_level)
        return empty_line + doc_str + body_str, after_body_str

    def extract_function_name(text: str) -> str:
        """Extract function name from a function definition starting with 'def '"""
        # Pattern to match: def function_name( or def function_name(
        pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return ""

    def consume_def(text: str) -> tuple[str, str]:
        """
        Processes a single function definition.
        Returns (processed_function, remaining_text_after_function)
        """
        # Extract function name to check against ignore list
        function_name = extract_function_name(text)
        
        # Find the end of the function signature
        # Match either "):  or ") -> anything:"
        pattern = r'\)\s*(?:->.*?)?:'
        match = re.search(pattern, text)
        
        if match:
            signature_end = match.end()
            before_body = text[:signature_end]
            after_signature = text[signature_end:]
            if after_signature.startswith("\n"):
                before_body += "\n"
                after_signature = after_signature[1:]
            
            # If function is in ignore list, find the full function body and return it as-is
            if function_name in keep_body_functions:
                # We need to find the complete function body without eliding it
                # Use the same logic but don't replace the body with "..."
                empty_line, after_empty_line = consume_empty_line(after_signature)
                doc_str, after_doc_str = consume_docstring(after_empty_line)
                
                # Find the end of the function body by looking for lines with less indentation
                lines = after_doc_str.split('\n')
                if lines and lines[0].strip():  # If there's content on the first line
                    indent_level = len(lines[0]) - len(lines[0].lstrip())
                else:
                    # Find the first non-empty line to determine indent level
                    indent_level = None
                    for line in lines:
                        if line.strip():
                            indent_level = len(line) - len(line.lstrip())
                            break
                    if indent_level is None:
                        indent_level = 4  # Default indentation
                
                split_index = len(lines)
                for i, line in enumerate(lines):
                    if line.strip() == "":
                        continue
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent < indent_level:
                        split_index = i
                        break
                
                body_lines = lines[:split_index]
                remaining_lines = lines[split_index:]
                
                full_body = empty_line + doc_str + '\n'.join(body_lines)
                remaining_text = '\n'.join(remaining_lines)
                if remaining_text and not remaining_text.startswith('\n'):
                    remaining_text = '\n' + remaining_text
                
                processed_function = before_body + full_body
                return processed_function, remaining_text
            else:
                # Normal elision process
                processed_body, remaining_text = consume_function_body_start(after_signature)
                processed_function = before_body + processed_body
                return processed_function, remaining_text
        else:
            # No valid function signature found, return as-is
            return text, ""

    def consume_outside(text: str) -> str:
        def_point = text.find("def ")
        if def_point == -1:
            # No more functions found
            return text
        
        before_def = text[:def_point]
        after_def = text[def_point:]
        
        # Process this function
        processed_function, remaining_after_function = consume_def(after_def)
        
        # Recursively process any remaining text for more functions
        remaining_processed = consume_outside(remaining_after_function)
        
        return before_def + processed_function + remaining_processed

    return consume_outside(text)


def elide_function_bodies_real_parser(
    code: str,
    ignore_functions: list[str] = None,
) -> str:
    funcs = PythonLangSpec().find_functions(code)
    for func in funcs:
        if func.get_function_name() in ignore_functions:
            continue
        body = func.get_body_src(include_prefix_comment=True).lstrip("\n")
        indent_level = len(body) - len(body.lstrip())
        code = code.replace(
            body,
            " " * indent_level + "..." + ("\n" if body.endswith("\n") else "")
        )
    return code

safe_chars_per_token = 4.364 - 3 * 0.368

class ShorteningPromptRenderer(PromptRenderer):
    def __init__(
        self,
        wrapper_renderer: PromptRenderer,
        only_datasets: list[DatasetName] = None,
        target_chars: int = int(5000 * safe_chars_per_token),
    ):
        self.wrapper_renderer = wrapper_renderer
        self.only_datasets = only_datasets or []
        self.target_chars = target_chars

    def render(
        self,
        problem: CodeProblem,
        lm: LmPredictor,
        prompt_seed: int | None = None,
    ) -> LmPromptRender:
        prompt = self.wrapper_renderer.render(problem, lm, prompt_seed)
        if problem.dataset_name not in self.only_datasets:
            return prompt
        assert prompt.prompt.is_text_a_chat()
        last_turn = prompt.prompt.text[-1]
        last_turn_text = last_turn.content
        func_names = regex_func_names(last_turn_text)
        keep_n_last = 64
        # Iteratively elide more function bodies until under target_chars or out of functions
        while len(last_turn_text) > self.target_chars and keep_n_last > 4:
            func_names = func_names[-min(keep_n_last, len(func_names)):] if func_names else []
            last_turn_text = elide_function_definitions(
                last_turn_text, keep_body_functions=func_names)
            keep_n_last = max(1, keep_n_last // 2)
        if len(last_turn_text) > self.target_chars:
            sep = '\n... ELIDED ...\n'
            sep_len = len(sep)
            budget = int(self.target_chars - sep_len - 2)
            start_len = int(budget * 0.2)
            end_len = budget - start_len
            first = last_turn_text[:start_len]
            last = last_turn_text[-end_len:]
            last_turn_text = first + sep + last
        last_turn.content = last_turn_text
        return prompt


def regex_func_names(text: str) -> list[str]:
    """Match on likely names of functions"""
    pattern = r'\b(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    matches = re.findall(pattern, text)
    return matches



def make_shortening_solver() -> LmCodeSolverAutoRegressive:
    return make_solver_partial(
        LmCodeSolverAutoRegressive,
        prompt_renderer=ShorteningPromptRenderer(
            wrapper_renderer=PromptRendererSingleEditGeneric(),
            only_datasets=[DatasetName.repocod],
        ),
    )


def test_shorten_code_simplest():
    code = '''
def foo():
    a = 1
    return a
'''.strip()
    short = elide_function_definitions(code)
    print("RESULT")
    print(short)
    assert short == '''
def foo():
    ...
'''.strip()


def test_shorten_code_simplest_with_docstring():
    code = '''
def foo():
    """This is a docstring"""
    a = 1
    return a
'''.strip()
    short = elide_function_definitions(code)
    print("RESULT")
    print(short)
    assert short == '''
def foo():
    """This is a docstring"""
    ...
'''.strip()


def test_shorten_code_simplest_two_funcs():
    code = '''
def foo():
    a = 1
    return a

def bar():
    b = 2
    return b
'''.strip()
    short = elide_function_definitions(code)
    print("RESULT")
    print(short)
    assert short == '''
def foo():
    ...

def bar():
    ...
'''.strip()


def test_shorten_code_simplest_with_docstring_ml():
    code = '''
def foo():
    """This is a docstring
    with multiple lines
    """
    a = 1
    return a
'''.strip()
    short = elide_function_definitions(code)
    print("RESULT")
    print(short)
    assert short == '''
def foo():
    """This is a docstring
    with multiple lines
    """
    ...
'''.strip()


def test_shorten_code():
    code = '''
def foo():
    a = 1
    return a

def bar():
    """This is a docstring"""
    b = 2
    return b

def baz(
    b = 3
):
    c = 3
    return b


def qux():
    return 4
'''.strip()
    short = elide_function_definitions(code)
    print("RESULT")
    print(short)
    assert short == '''
def foo():
    ...

def bar():
    """This is a docstring"""
    ...

def baz(
    b = 3
):
    ...


def qux():
    ...
'''.strip()


def test_single_quote_docstring():
    code = """def test():
    '''Single quote docstring'''
    return 42"""
    
    result = elide_function_definitions(code)
    expected = """def test():
    '''Single quote docstring'''
    ..."""
    assert result == expected


def test_multiline_docstring():
    code = """def test():
    '''
    Multi-line
    docstring
    '''
    return 42"""
    
    result = elide_function_definitions(code)
    expected = """def test():
    '''
    Multi-line
    docstring
    '''
    ..."""
    assert result == expected


def test_no_docstring():
    code = """def simple():
    x = 1
    y = 2
    return x + y"""
    
    result = elide_function_definitions(code)
    expected = """def simple():
    ..."""
    assert result == expected


def test_class_methods():
    """Test that functions inside classes get elided but class structure is preserved"""
    code = """class MyClass:
    def __init__(self):
        '''Constructor docstring'''
        self.value = 42
        
    def method1(self):
        '''Method docstring'''
        return self.value
    
    def method2(self):
        x = 1
        y = 2
        return x + y
        
    @property
    def prop(self):
        return self.value * 2"""
    
    result = elide_function_definitions(code)
    expected = """class MyClass:
    def __init__(self):
        '''Constructor docstring'''
        ...
        
    def method1(self):
        '''Method docstring'''
        ...
    
    def method2(self):
        ...
        
    @property
    def prop(self):
        ..."""
    assert result == expected


def test_nested_functions():
    """Test that outer function gets completely elided, including nested functions"""
    code = """def outer_function():
    x = 1
    
    def inner_function():
        '''Inner docstring'''
        return x * 2
        
    def another_inner():
        y = 3
        return y
    
    result = inner_function() + another_inner()
    return result"""
    
    result = elide_function_definitions(code)
    expected = """def outer_function():
    ..."""
    assert result == expected


def test_mixed_functions_and_classes():
    """Test mixed top-level functions and classes"""
    code = """def top_level_func():
    return "hello"

class MyClass:
    def method(self):
        '''Method with docstring'''
        return 42
        
def another_top_level():
    '''Another function'''
    x = 1
    return x

class AnotherClass:
    pass"""
    
    result = elide_function_definitions(code)
    expected = """def top_level_func():
    ...

class MyClass:
    def method(self):
        '''Method with docstring'''
        ...
        
def another_top_level():
    '''Another function'''
    ...

class AnotherClass:
    pass"""
    assert result == expected


def test_functions_with_decorators():
    """Test functions with decorators"""
    code = """@decorator
def decorated_func():
    return "decorated"

class MyClass:
    @staticmethod
    def static_method():
        '''Static method docstring'''
        return 123
        
    @classmethod  
    def class_method(cls):
        return cls"""
    
    result = elide_function_definitions(code)
    expected = """@decorator
def decorated_func():
    ...

class MyClass:
    @staticmethod
    def static_method():
        '''Static method docstring'''
        ...
        
    @classmethod  
    def class_method(cls):
        ..."""
    assert result == expected


def test_complex_signatures():
    """Test functions with complex multi-line signatures"""
    code = """def complex_function(
    param1: str,
    param2: int = 42,
    param3: Optional[List[Dict[str, Any]]] = None,
    *args,
    **kwargs
) -> Tuple[str, int]:
    '''Complex function with type hints'''
    if param3 is None:
        param3 = []
    return param1, param2"""
    
    result = elide_function_definitions(code)
    expected = """def complex_function(
    param1: str,
    param2: int = 42,
    param3: Optional[List[Dict[str, Any]]] = None,
    *args,
    **kwargs
) -> Tuple[str, int]:
    '''Complex function with type hints'''
    ..."""
    assert result == expected


def test_empty_functions():
    """Test functions with just pass or ellipsis"""
    code = """def empty1():
    pass

def empty2():
    ...

def empty3():
    '''Just a docstring'''
    pass"""
    
    result = elide_function_definitions(code)
    expected = """def empty1():
    ...

def empty2():
    ...

def empty3():
    '''Just a docstring'''
    ..."""
    assert result == expected


def test_ignore_functions():
    """Test that functions in ignore_functions list are not elided"""
    code = '''
def foo():
    """This function should be ignored"""
    a = 1
    b = 2
    return a + b

def bar():
    """This function should be elided"""
    c = 3
    d = 4
    return c * d

def baz():
    """This function should also be ignored"""
    e = 5
    return e
'''.strip()
    
    short = elide_function_definitions(code, keep_body_functions=['foo', 'baz'])
    print("RESULT")
    print(short)
    expected = '''
def foo():
    """This function should be ignored"""
    a = 1
    b = 2
    return a + b

def bar():
    """This function should be elided"""
    ...

def baz():
    """This function should also be ignored"""
    e = 5
    return e
'''.strip()
    assert short == expected


def test_ignore_functions_empty_list():
    """Test with empty ignore_functions list - should elide all functions"""
    code = '''
def foo():
    a = 1
    return a

def bar():
    b = 2
    return b
'''.strip()
    
    short = elide_function_definitions(code, keep_body_functions=[])
    expected = '''
def foo():
    ...

def bar():
    ...
'''.strip()
    assert short == expected


def test_ignore_functions_nonexistent():
    """Test with function names that don't exist in the code"""
    code = '''
def foo():
    a = 1
    return a

def bar():
    b = 2
    return b
'''.strip()
    
    short = elide_function_definitions(code, keep_body_functions=['nonexistent', 'also_missing'])
    expected = '''
def foo():
    ...

def bar():
    ...
'''.strip()
    assert short == expected


def test_ignore_functions_with_class_methods():
    """Test ignore_functions with class methods"""
    code = '''
class MyClass:
    def __init__(self):
        self.value = 42
        
    def method1(self):
        return self.value
    
    def method2(self):
        x = 1
        y = 2
        return x + y
'''.strip()
    
    short = elide_function_definitions(code, keep_body_functions=['method1'])
    expected = '''
class MyClass:
    def __init__(self):
        ...
        
    def method1(self):
        return self.value
    
    def method2(self):
        ...
'''.strip()
    assert short == expected


def test_empty_function_body():
    code = '''
def foo():
    return 1

def bar():
'''.strip()
    
    short = elide_function_definitions(code)
    expected = '''
def foo():
    ...

def bar():
'''.strip()
    assert short == expected


def test_empty_function_body_2():
    code = '''
    class Foo:
        def bar():
            """foo"""
    class Bar:
'''.strip()
    
    short = elide_function_definitions(code)
    expected = '''
    class Foo:
        def bar():
            """foo"""
    class Bar:
'''.strip()
    assert short == expected