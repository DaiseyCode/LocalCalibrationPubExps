import os
import json
import re
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
import inspect


def sanitize_str(s):
    """Convert to string and sanitize for use in paths (remove non-alphanumeric, replace spaces)"""
    s = str(s)
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', s)


def get_git_sha():
    """Get current git SHA if available"""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return "git_sha_unavailable"


def get_caller_info():
    """Get information about the caller of the function"""
    stack = traceback.extract_stack()
    # Go back 2 frames to get the caller of the caller (since this is called from plot_and_save)
    if len(stack) >= 3:
        frame = stack[-3]
        return {
            'file': frame.filename,
            'line': frame.lineno,
            'function': frame.name,
        }
    return {}


_root_path = Path(__file__).parent / "plots"

def save_plot_with_metadata(save_path, metadata, format='svg'):
    """Save the current matplotlib plot and associated metadata"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save plot
    plt.savefig(f"{save_path}.{format}", format=format)
    
    # Save metadata
    with open(f"{save_path}.meta.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def plot_and_save(dir_components, title=None, properties=None, archive=True, show=False, format=['png', 'svg'], plots_base_dir=None):
    """
    Save the current matplotlib plot to a file and optionally archive it.
    
    Args:
        dir_components: List of strings/values to build subdirectory path
        title: Title to use as the base filename (does NOT set plot title)
        properties: Dict of property keys to include in filename (optional)
        archive: Whether to also save to the archive directory
        show: Whether to call plt.show() after saving
        format: File format to save (default: svg)
        plots_base_dir: Base directory for all plots, defaults to 'plots' under caller's directory
    """
    # Properties defaults to empty dict
    if properties is None:
        properties = {}
    
    # Get base directory relative to caller
    if plots_base_dir is None:
        plots_base_dir = _root_path
    
    # Sanitize directory components and create subdir path
    sanitized_components = [sanitize_str(c) for c in dir_components]
    subdir = os.path.join(*sanitized_components)
    
    # Create base filename from title or use 'plot' if not provided
    base_filename = sanitize_str(title) if title else 'plot'
    
    # Add properties suffix if any properties provided
    if properties:
        # Just use the property keys in the filename, not their values
        prop_keys = [k for k in sorted(properties.keys()) if k != 'type']
        # If there's a 'type' property, put it first
        if 'type' in properties:
            props_suffix = sanitize_str(properties['type'])
            if prop_keys:
                props_suffix += '_' + '_'.join(sanitize_str(k) for k in prop_keys)
        else:
            props_suffix = '_'.join(sanitize_str(k) for k in prop_keys) if prop_keys else ''
        
        # Only add suffix if there's something to add
        filename = f"{base_filename}_{props_suffix}" if props_suffix else base_filename
    else:
        filename = base_filename
    
    # Prepare metadata (include full properties for reference)
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'git_sha': get_git_sha(),
        'caller': get_caller_info(),
        'dir_components': dir_components,
        'properties': properties,
        'title': title,
    }
    
    # Save to main plots directory
    main_dir = os.path.join(plots_base_dir, subdir)
    main_path = os.path.join(main_dir, filename)
    for fmt in (format if isinstance(format, list) else [format]):
        save_plot_with_metadata(main_path, metadata, fmt)
    
    # Save to archive with timestamp
    if archive:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_dir = os.path.join(plots_base_dir, 'archive', subdir)
        archive_path = os.path.join(archive_dir, f"{timestamp}_{filename}")
        for fmt in (format if isinstance(format, list) else [format]):
            save_plot_with_metadata(archive_path, metadata, fmt)
    
    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()  # Close plot to prevent memory issues


def plot(dir_components, title=None, properties=None, format=['png', 'svg'], plots_base_dir=None):
    """
    Show the plot and also save it (with archiving)
    
    This is a convenience wrapper around plot_and_save with show=True
    """
    return plot_and_save(
        dir_components=dir_components,
        title=title,
        properties=properties,
        archive=True,
        show=True,
        format=format,
        plots_base_dir=plots_base_dir
    )


if __name__ == "__main__":
    print(_root_path)