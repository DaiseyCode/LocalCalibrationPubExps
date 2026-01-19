
from pprint import pprint
import dataclasses
from pathlib import Path
import shutil
from synthegrator.synthdatasets import DatasetName
from synthegrator.synthdatasets import yield_problems_from_name
from typing import TypeVar
import re
from synthegrator.synthdatasets import DatasetName
from localizing.filter_helpers import debug_str_filterables
from localizing.localizing_structs import BaseLocalization, LocalizationList
from localizing.probe.embedding_styles import EmbeddingStyle
from localizing.probe.probe_data_gather import get_or_serialize_localizations_embedded
from localizing.probe.agg_models.agg_config import ProbeConfig
from collections import Counter, defaultdict
from lmwrapper.openai_wrapper import OpenAiModelNames
import numpy as np
from itertools import combinations
import subprocess
import datetime

from localizing.probe.agg_models.agg_config import ProbeConfig
from pape.dataset_flow_stats import do_with_config
from pape.configs import BASE_PAPER_CONFIG


cur_file = Path(__file__).parent
TABLE_DIR = cur_file / "gen/tables"



def get_git_info():
    """Get current git commit SHA and timestamp."""
    try:
        # Get git SHA
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                    stderr=subprocess.DEVNULL).decode('utf-8').strip()
        # Get short SHA (first 7 characters)
        short_sha = sha[:7]
        # Check if there are any uncommitted changes
        changes = subprocess.check_output(['git', 'status', '--porcelain'], 
                                          stderr=subprocess.DEVNULL).decode('utf-8').strip()
        if changes:
            short_sha += " (dirty)"
        return short_sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def escape_latex(text):
    """Escape special LaTeX characters in text, but preserve LaTeX commands."""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.replace('%', '\\%')
    text = text.replace('#', '\\#')
    
    return text

def generate_latex_table(rows, full_width=False, caption_content="Caption", label="tab:dataset_stats", caption_at_top=False, resize_to_fit=False):
    """Generate a LaTeX table from the rows data with support for arbitrarily deep nested columns.
    
    Args:
        rows: List of dictionaries containing table data, or strings for LaTeX commands (e.g., "\\midrule")
        full_width: If True, use table* environment for full-width table
        caption_content: Text content for the table caption
        label: LaTeX label for the table
        caption_at_top: If True, place caption above table; if False, place below
        resize_to_fit: If True, use resizebox to scale table to fit column/text width
    """
    latex_output = []
    
    # Add metadata comment
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_sha = get_git_info()
    latex_output.append(f"% Generated on {timestamp} from git commit {git_sha}")
    latex_output.append("")
    
    # Table header
    if full_width:
        latex_output.append("\\begin{table*}[htbp]")
    else:
        latex_output.append("\\begin{table}[htbp]")
    
    # Add caption at top if requested
    if caption_at_top:
        latex_output.append(f"\\caption{{{caption_content}}}")
        latex_output.append(f"\\label{{{label}}}")
    
    if resize_to_fit:
        latex_output.append("% Note: Add \\usepackage{booktabs}, \\usepackage{amssymb}, \\usepackage{multirow}, \\usepackage{makecell}, and \\usepackage{graphicx} to document preamble")
    else:
        latex_output.append("% Note: Add \\usepackage{booktabs}, \\usepackage{amssymb}, \\usepackage{multirow}, and \\usepackage{makecell} to document preamble")
    
    # Find the first dictionary row to analyze structure
    sample_row = None
    for row in rows:
        if isinstance(row, dict):
            sample_row = row
            break
    
    if sample_row is None:
        raise ValueError("No dictionary rows found in data")
    
    # Analyze the column structure recursively
    column_structure = _analyze_column_structure(sample_row)
    flat_headers = _get_flat_headers(column_structure)
    total_columns = len(flat_headers)
    
    # Generate column specifications
    column_specs = []
    for header in flat_headers:
        if any(keyword in header.lower() for keyword in ['name', 'dataset', 'technique', '\\mutt']):
            column_specs.append('l')
        else:
            column_specs.append('c')
    
    # Add centering command
    latex_output.append("\\centering")
    
    # Add resizebox if requested
    if resize_to_fit:
        target_width = "\\textwidth" if full_width else "\\columnwidth"
        latex_output.append(f"\\resizebox{{{target_width}}}{{!}}{{%")
    
    # Create tabular environment
    latex_output.append(f"\\begin{{tabular}}{{{' '.join(column_specs)}}}")
    latex_output.append("\\toprule")
    
    # Generate headers with proper nesting
    _generate_nested_headers(latex_output, column_structure, total_columns)
    
    latex_output.append("\\midrule")
    
    # Data rows
    for row in rows:
        if isinstance(row, str):
            # Handle string rows (LaTeX commands like "\\midrule")
            latex_output.append(row)
        elif isinstance(row, dict):
            # Handle dictionary rows (actual data)
            values = _flatten_row_values(row)
            latex_row = " & ".join([escape_latex(str(v)) for v in values]) + " \\\\"
            latex_output.append(latex_row)
        else:
            # Handle unexpected row types
            raise ValueError(f"Unexpected row type: {type(row)}. Expected dict or str.")
    
    # Table footer
    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    
    # Close resizebox if used
    if resize_to_fit:
        latex_output.append("}%")
    
    # Add caption at bottom if requested
    if not caption_at_top:
        latex_output.append(f"\\caption{{{caption_content}}}")
        latex_output.append(f"\\label{{{label}}}")
    
    if full_width:
        latex_output.append("\\end{table*}")
    else:
        latex_output.append("\\end{table}")
    
    return "\n".join(latex_output)


def _analyze_column_structure(row_dict):
    """Recursively analyze the column structure to support arbitrary nesting depth."""
    structure = []
    
    for key, value in row_dict.items():
        if isinstance(value, dict):
            # Recursive case: nested dictionary
            sub_structure = _analyze_column_structure(value)
            structure.append({
                'name': key,
                'type': 'group',
                'children': sub_structure,
                'span': _count_leaf_columns(sub_structure)
            })
        else:
            # Base case: leaf value
            structure.append({
                'name': key,
                'type': 'leaf',
                'span': 1
            })
    
    return structure


def _count_leaf_columns(structure):
    """Count the total number of leaf columns in a structure."""
    total = 0
    for item in structure:
        if item['type'] == 'leaf':
            total += 1
        else:
            total += item['span']
    return total


def _get_flat_headers(structure):
    """Extract all leaf column headers in order."""
    headers = []
    for item in structure:
        if item['type'] == 'leaf':
            headers.append(item['name'])
        else:
            headers.extend(_get_flat_headers(item['children']))
    return headers


def _generate_nested_headers(latex_output, structure, total_columns):
    """Generate LaTeX headers with proper multicolumn and cmidrule commands."""
    # Find the maximum depth
    max_depth = _get_max_depth(structure)
    
    if max_depth == 1:
        # Simple case: no nesting
        headers = [escape_latex(item['name']) for item in structure]
        latex_output.append(" & ".join(headers) + " \\\\")
        return
    
    # Generate headers level by level
    for level in range(max_depth):
        header_row, cmidrules = _build_header_row_at_level(structure, level, max_depth)
        
        # Add the header row
        latex_output.append(" & ".join(header_row) + " \\\\")
        
        # Add cmidrules only for the second-to-last level (makes table less busy)
        if level == max_depth - 2 and cmidrules:
            for cmidrule in cmidrules:
                latex_output.append(cmidrule)


def _build_header_row_at_level(structure, target_level, max_depth):
    """Build a complete header row for a specific level."""
    header_parts = []
    cmidrules = []
    current_col = 1
    
    for item in structure:
        item_depth = _get_item_depth(item)
        
        if target_level == 0:
            # Top level
            if item['type'] == 'group' and item_depth > 1:
                # Show the group name spanning all its columns
                header_parts.append(f"\\multicolumn{{{item['span']}}}{{c}}{{{escape_latex(item['name'])}}}")
                if item['span'] > 1:
                    cmidrules.append(f"\\cmidrule(lr){{{current_col}-{current_col + item['span'] - 1}}}")
            elif item['type'] == 'leaf':
                # For single columns, add empty string at top level (will show name in bottom row)
                header_parts.append("")
            else:
                header_parts.append("")
        
        elif target_level == max_depth - 1:
            # Bottom level - show actual column names
            if item['type'] == 'leaf':
                # For single columns, add the column name in the final row
                header_parts.append(escape_latex(item['name']))
            else:
                # For groups, collect their leaf headers
                leaf_headers = []
                _collect_leaf_headers(item, leaf_headers)
                header_parts.extend(leaf_headers)
        
        else:
            # Middle levels
            middle_headers, middle_cmidrules = _get_headers_at_middle_level(item, target_level, current_col)
            header_parts.extend(middle_headers)
            cmidrules.extend(middle_cmidrules)
        
        current_col += item['span']
    
    return header_parts, cmidrules


def _get_headers_at_middle_level(item, target_level, start_col):
    """Get headers for middle levels of nesting."""
    headers = []
    cmidrules = []
    
    if item['type'] == 'leaf':
        headers.append("")
        return headers, cmidrules
    
    # For groups, we need to look at their children at the appropriate depth
    current_col = start_col
    level_depth = target_level
    
    def process_level(structure, remaining_depth, col_pos):
        level_headers = []
        level_cmidrules = []
        
        if remaining_depth == 1:
            # Show the direct children
            for child in structure:
                if child['type'] == 'group':
                    level_headers.append(f"\\multicolumn{{{child['span']}}}{{c}}{{{escape_latex(child['name'])}}}")
                    if child['span'] > 1:
                        level_cmidrules.append(f"\\cmidrule(lr){{{col_pos}-{col_pos + child['span'] - 1}}}")
                else:
                    level_headers.append("")
                col_pos += child['span']
        else:
            # Go deeper
            for child in structure:
                if child['type'] == 'group':
                    child_headers, child_cmidrules = process_level(child['children'], remaining_depth - 1, col_pos)
                    level_headers.extend(child_headers)
                    level_cmidrules.extend(child_cmidrules)
                else:
                    level_headers.append("")
                col_pos += child['span']
        
        return level_headers, level_cmidrules
    
    if item['type'] == 'group':
        return process_level(item['children'], level_depth, current_col)
    
    return headers, cmidrules


def _collect_leaf_headers(item, headers):
    """Collect all leaf headers from an item."""
    if item['type'] == 'leaf':
        headers.append(escape_latex(item['name']))
    else:
        for child in item['children']:
            _collect_leaf_headers(child, headers)


def _get_max_depth(structure):
    """Get the maximum depth of nesting in the structure."""
    max_depth = 1
    for item in structure:
        if item['type'] == 'group':
            item_depth = 1 + _get_max_depth(item['children'])
            max_depth = max(max_depth, item_depth)
    return max_depth


def _get_item_depth(item):
    """Get the depth of a specific item."""
    if item['type'] == 'leaf':
        return 1
    else:
        return 1 + _get_max_depth(item['children'])


def _flatten_row_values(row_dict):
    """Recursively flatten all values from a nested dictionary structure."""
    values = []
    for key, value in row_dict.items():
        if isinstance(value, dict):
            values.extend(_flatten_row_values(value))
        else:
            values.append(value)
    return values


def render_latex_table_to_file(latex_table: str, output_path: str, format: str = "pdf"):
    """
    Render a LaTeX table string to a PDF or SVG file.

    Args:
        latex_table: The LaTeX table code (from generate_latex_table).
        output_path: Path to save the rendered file (should end with .pdf or .svg).
        format: 'pdf' or 'svg'.
    """
    import tempfile
    import os
    import subprocess
    import shutil

    # Minimal LaTeX document with required packages
    minimal_tex = f"""
    \\documentclass[preview]{{standalone}}
    \\usepackage{{booktabs}}
    \\usepackage{{amssymb}}
    \\usepackage{{multirow}}
    \\usepackage{{makecell}}
    \\usepackage{{graphicx}}
    \\begin{{document}}
    {latex_table}
    \\end{{document}}
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "table.tex")
        pdf_path = os.path.join(tmpdir, "table.pdf")
        # Write the .tex file
        with open(tex_path, "w") as f:
            f.write(minimal_tex)
        # Run pdflatex
        try:
            subprocess.run([
                "pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"pdflatex failed: {e.stderr.decode('utf-8', errors='ignore')}")
        if format == "pdf":
            shutil.copyfile(pdf_path, output_path)
        elif format == "svg":
            svg_path = output_path
            try:
                subprocess.run([
                    "pdf2svg", pdf_path, svg_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"pdf2svg failed: {e.stderr.decode('utf-8', errors='ignore')}")
        else:
            raise ValueError("format must be 'pdf' or 'svg'")


def dict_to_latex_vars_cmds(d: dict[str, str]) -> str:
    """Use newcommand to create latex variables from a dictionary"""

    def snake_to_cammel(s):
        return "".join([w.capitalize() for w in s.split("_")])

    out = []
    for k, v in d.items():
        cmd_name = snake_to_cammel(k)
        cmd_name = re.sub(r'\d', 'N', cmd_name)
        out.append(f"\\newcommand{{\\{cmd_name}}}{{{escape_latex(v)}\\xspace}}")
    return "\n".join(out)