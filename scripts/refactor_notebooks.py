#!/usr/bin/env python3
"""
Notebook Refactoring Script
Applies the same refactoring rules used on Python files to Jupyter notebooks:
- Removes verbose/Italian comments
- Removes emojis
- Consolidates multi-line docstrings
- Cleans up whitespace
"""

import json
import re
import sys
from pathlib import Path


def remove_emojis(text: str) -> str:
    """Remove common emojis from text."""
    emojis = ['🔧', '📦', '📂', '✂️', '📊', '✅', '❌', '⚠️', '🌍', '🚀', '💾', '🔍', '🗑️', '🔄', '📈', '📉', '🎯', '💡', '🔥', '⭐', '🎉', '👍', '👎', '🤖', '📝', '📌', '🏆', '🛠️', '⚙️', '🧪', '🧹', '📁', '📋', '🔗', '🔒', '🔓', '⏰', '⏳', '🔃', '🔄', '➡️', '⬅️', '⬆️', '⬇️', '↔️', '↕️', '🔀', '🔂', '🔁', 'ℹ️']
    for emoji in emojis:
        text = text.replace(emoji + ' ', '')  # Remove emoji with trailing space
        text = text.replace(emoji, '')  # Remove standalone emoji
    return text


def is_verbose_comment(line: str) -> bool:
    """Check if a line is a verbose comment that should be removed."""
    stripped = line.strip()
    if not stripped.startswith('#'):
        return False
    
    # Keep structural comments
    structural_patterns = [
        r'^# ={3,}',  # Section headers like # ===...
        r'^# -{3,}',  # Dividers like # ---...
        r'^# TODO',
        r'^# FIXME',
        r'^# NOTE',
        r'^# type:',
        r'^# noqa',
        r'^# pylint',
        r'^# fmt:',
    ]
    for pattern in structural_patterns:
        if re.match(pattern, stripped, re.IGNORECASE):
            return False
    
    # Remove Italian comments
    italian_words = ['esempio', 'nota', 'attenzione', 'configurazione', 'trucco', 
                     'adattamento', 'creiamo', 'spezziamo', 'salviamo', 'passiamo',
                     'carichiamo', 'inizializziamo', 'verifichiamo']
    comment_lower = stripped.lower()
    for word in italian_words:
        if word in comment_lower:
            return True
    
    # Remove overly verbose comments (more than 80 chars)
    if len(stripped) > 100:
        return True
    
    return False


def clean_code_cell(source_lines: list) -> list:
    """Clean a code cell's source lines."""
    cleaned = []
    
    for line in source_lines:
        # Remove emojis
        line = remove_emojis(line)
        
        # Skip verbose comments
        if is_verbose_comment(line):
            continue
        
        # Compress multiple blank lines
        if not line.strip() and cleaned and not cleaned[-1].strip():
            continue
        
        cleaned.append(line)
    
    # Remove trailing blank lines
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    
    return cleaned


def clean_markdown_cell(source_lines: list) -> list:
    """Clean a markdown cell's source lines."""
    cleaned = []
    
    for line in source_lines:
        # Remove emojis
        line = remove_emojis(line)
        cleaned.append(line)
    
    return cleaned


def refactor_notebook(input_path: str, output_path: str = None) -> dict:
    """
    Refactor a Jupyter notebook.
    
    Args:
        input_path: Path to input notebook
        output_path: Path to output notebook (defaults to overwriting input)
    
    Returns:
        Stats dict with before/after cell counts
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path
    
    print(f"Processing: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    stats = {
        'code_cells': 0,
        'markdown_cells': 0,
        'lines_before': 0,
        'lines_after': 0
    }
    
    for cell in notebook.get('cells', []):
        cell_type = cell.get('cell_type', '')
        source = cell.get('source', [])
        
        # Handle source as string or list
        if isinstance(source, str):
            source_text = source
        else:
            source_text = ''.join(source)
        
        # Remove emojis from full text first
        source_text = remove_emojis(source_text)
        
        # Split back into lines
        source_lines = source_text.split('\n')
        
        lines_before = len(source_lines)
        stats['lines_before'] += lines_before
        
        if cell_type == 'code':
            stats['code_cells'] += 1
            cleaned = clean_code_cell(source_lines)
        elif cell_type == 'markdown':
            stats['markdown_cells'] += 1
            cleaned = clean_markdown_cell(source_lines)
        else:
            cleaned = source_lines
        
        stats['lines_after'] += len(cleaned)
        
        # Convert back to list format with newlines (except last line)
        if cleaned:
            cell['source'] = [line + '\n' for line in cleaned[:-1]] + [cleaned[-1]]
        else:
            cell['source'] = []
        
        # Clear outputs to reduce file size
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    reduction = stats['lines_before'] - stats['lines_after']
    print(f"  Code cells: {stats['code_cells']}, Markdown cells: {stats['markdown_cells']}")
    print(f"  Lines: {stats['lines_before']} -> {stats['lines_after']} (-{reduction})")
    print(f"  Saved to: {output_path}")
    
    return stats


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python refactor_notebooks.py <notebook.ipynb> [output.ipynb]")
        print("       python refactor_notebooks.py --all  # Process all notebooks")
        sys.exit(1)
    
    if sys.argv[1] == '--all':
        # Process all notebooks in the project
        project_root = Path(__file__).parent.parent
        notebooks = list(project_root.glob('notebooks/*.ipynb')) + list(project_root.glob('tests/*.ipynb'))
        
        total_stats = {'lines_before': 0, 'lines_after': 0}
        for nb in notebooks:
            stats = refactor_notebook(str(nb))
            total_stats['lines_before'] += stats['lines_before']
            total_stats['lines_after'] += stats['lines_after']
        
        print(f"\n=== TOTAL ===")
        print(f"Notebooks processed: {len(notebooks)}")
        print(f"Total lines: {total_stats['lines_before']} -> {total_stats['lines_after']}")
        print(f"Reduction: {total_stats['lines_before'] - total_stats['lines_after']} lines")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        refactor_notebook(input_path, output_path)


if __name__ == "__main__":
    main()
