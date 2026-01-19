from typing import Iterable, TypeAlias


TAG_VALUE_TYPES: TypeAlias = str | float | None | tuple['TAG_VALUE_TYPES', ...]
TAG_TYPE: TypeAlias = tuple[str, TAG_VALUE_TYPES]



class FilterTrackingMixin:
    """A mixin for tracking certain filters that a certain object might go through.
    Supports recursive checking of child attributes that are also FilterTrackingMixin instances.

    This implementation uses properties to ensure filter tracking state is always properly
    initialized, regardless of inheritance order or whether __init__ is called.

    Example:
        class MyFilterableClass(FilterTrackingMixin, SomeOtherClass):
            def some_method(self):
                if self.annotate_filter("size_check", len(self.data) > 10):
                    # Handle passing case
                    pass
    """

    @property
    def _passed_filter_names(self) -> list[tuple[str, str | float | None]]:
        """List of filters this object has passed, with optional metadata."""
        if not hasattr(self, '_passed_filter_names_storage'):
            self._passed_filter_names_storage = []
        return self._passed_filter_names_storage

    @_passed_filter_names.setter
    def _passed_filter_names(self, value: list[tuple[str, str | float | None]]) -> None:
        self._passed_filter_names_storage = value

    @property
    def _failed_filter_names(self) -> list[tuple[str, str | float | None]]:
        """List of filters this object has failed, with optional metadata."""
        if not hasattr(self, '_failed_filter_names_storage'):
            self._failed_filter_names_storage = []
        return self._failed_filter_names_storage

    @_failed_filter_names.setter
    def _failed_filter_names(self, value: list[tuple[str, str | float | None]]) -> None:
        self._failed_filter_names_storage = value

    @property
    def _tags(self) -> list[TAG_TYPE]:
        """List of tags associated with this object."""
        if not hasattr(self, '_tags_storage'):
            self._tags_storage = []
        return self._tags_storage

    @_tags.setter
    def _tags(self, value: list[TAG_TYPE]) -> None:
        self._tags_storage = value

    def annotate_tag(self, tag: str, tag_value: TAG_VALUE_TYPES = None) -> None:
        """Add a tag to this object. Tags are multi-sets so can be added multiple times."""
        self._tags.append((tag, tag_value))

    def get_tags(self) -> list[TAG_TYPE]:
        """Get all tags associated with this object."""
        return self._tags.copy()

    def get_tag_values(self, tag: str) -> list[TAG_VALUE_TYPES]:
        """Get the values of a tag"""
        values = []
        for t, v in self._tags:
            if t == tag:
                values.append(v)
        return values

    def get_all_child_tracking(self) -> list['FilterTrackingMixin']:
        """Looks for any attributes that are FilterTrackingMixins.

        Returns:
            List of child objects that inherit from FilterTrackingMixin
        """
        children = []

        # Check all non-private attributes
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue

            value = getattr(self, attr_name, None)
            if value is not None:
                if isinstance(value, FilterTrackingMixin):
                    children.append(value)
                # Handle list/tuple of FilterTrackingMixin
                elif isinstance(value, (list, tuple)):
                    children.extend(
                        item for item in value
                        if isinstance(item, FilterTrackingMixin)
                    )
                # Handle dict values that are FilterTrackingMixin
                elif isinstance(value, dict):
                    children.extend(
                        v for v in value.values()
                        if isinstance(v, FilterTrackingMixin)
                    )

        return children

    def annotate_passed_filter(
        self,
        filter_name: str,
        filter_metadata: str | float | None = None
    ) -> None:
        """Record that this object passed a filter."""
        self._passed_filter_names.append((filter_name, filter_metadata))

    def annotate_failed_filter(
        self,
        filter_name: str,
        filter_metadata: str | float | None = None
    ) -> None:
        """Record that this object failed a filter."""
        self._failed_filter_names.append((filter_name, filter_metadata))

    def annotate_filter(
        self,
        filter_name: str,
        passed: bool,
        filter_metadata: str | float | None = None
    ) -> bool:
        """Record whether this object passed or failed a filter.

        Args:
            filter_name: Name of the filter being applied
            passed: Whether the filter passed
            filter_metadata: Optional metadata about the filter application

        Returns:
            The passed parameter (allowing for use in if statements)
        """
        if passed:
            self.annotate_passed_filter(filter_name, filter_metadata)
        else:
            self.annotate_failed_filter(filter_name, filter_metadata)
        return passed

    def check_passed_all_filters(self, recursive: bool = True) -> bool:
        """Check if this object and optionally all its children passed all filters.

        Args:
            recursive: If True, also check all child FilterTrackingMixin objects

        Returns:
            bool: True if no failed filters were recorded
        """
        # Check self first
        if self._failed_filter_names:
            return False

        # If not recursive, we're done
        if not recursive:
            return True

        # Check all children recursively
        return all(
            child.check_passed_all_filters(recursive=True)
            for child in self.get_all_child_tracking()
        )

    def get_all_failed_filters(self, recursive: bool = True) -> list[tuple[str, str | float | None]]:
        """Get all failed filters from this object and optionally its children.

        Args:
            recursive: If True, also get failed filters from all child FilterTrackingMixin objects

        Returns:
            List of (filter_name, metadata) tuples for all failed filters
        """
        failed = self._failed_filter_names.copy()

        if recursive:
            for child in self.get_all_child_tracking():
                failed.extend(child.get_all_failed_filters(recursive=True))

        return failed
    
    def get_all_passed_filters(self, recursive: bool = True) -> list[tuple[str, str | float | None]]:
        """Get all passed filters from this object and optionally its children.
        
        Args:
            recursive: If True, also get passed filters from all child FilterTrackingMixin objects
        """
        passed = self._passed_filter_names.copy()
        if recursive:
            for child in self.get_all_child_tracking():
                passed.extend(child.get_all_passed_filters(recursive=True))
        return passed

    def did_fail_filter(self, filter_name: str) -> bool:
        """Check if this object failed a specific filter.

        Args:
            filter_name: Name of the filter to check for

        Returns:
            bool: True if the filter failed
        """
        return any(name == filter_name for name, _ in self._failed_filter_names)
    

    def did_pass_filter(self, filter_name: str) -> bool:
        """Check if this object passed a specific filter.

        Args:
            filter_name: Name of the filter to check for

        Returns:
            bool: True if the filter failed
        """
        return any(name == filter_name for name, _ in self._passed_filter_names)

    def copy_filters_from(self, other: 'FilterTrackingMixin') -> None:
        """Copy all filter information from another FilterTrackingMixin object.

        Args:
            other: Another FilterTrackingMixin object to copy from

        Raises:
            ValueError: If other is not a FilterTrackingMixin
        """
        if not isinstance(other, FilterTrackingMixin):
            raise ValueError("Other must be a FilterTrackingMixin")

        self._passed_filter_names = other._passed_filter_names.copy()
        self._failed_filter_names = other._failed_filter_names.copy()
        self._tags = other._tags.copy()

    def get_filter_summary(self) -> dict:
        """Get a summary of all filters applied to this object.

        Returns:
            dict: Summary containing passed and failed filters, and tags
        """
        return {
            'passed_filters': self._passed_filter_names.copy(),
            'failed_filters': self._failed_filter_names.copy(),
            'tags': self._tags.copy(),
            'all_passed': len(self._failed_filter_names) == 0
        }


    def get_filter_sequence(self) -> list[tuple[str, bool, str | float | None]]:
        """Get the sequence of filters applied to this object in chronological order.

        Returns:
            List of (filter_name, passed, metadata) tuples in order of application
        """
        # Combine passed and failed filters with a passed/failed flag
        all_filters = [(name, True, meta) for name, meta in self._passed_filter_names]
        all_filters.extend((name, False, meta) for name, meta in self._failed_filter_names)
        return all_filters


from typing import Iterable
from collections import Counter, defaultdict


def debug_str_filterables(filterables: Iterable['FilterTrackingMixin']) -> str:
    """
    Generate a diagnostic string analyzing filter paths and outcomes for a collection of objects.

    Provides two views:
      1. Summary statistics including overall pass rate and the most common failure points.
      2. An aggregated flow diagram showing how objects traverse through filters.

    Args:
        filterables: Collection of FilterTrackingMixin objects to analyze

    Returns:
        Formatted string containing the analysis.
    """
    # Convert to list so we can iterate multiple times
    filterables = list(filterables)
    total_objects = len(filterables)
    if total_objects == 0:
        return "No objects to analyze"

    # --- Section 1: Summary Statistics ---

    # (a) Overall pass rate:
    passed_objects = sum(1 for obj in filterables if obj.check_passed_all_filters())

    # (b) Most common failure points:
    failure_counter = Counter()
    for obj in filterables:
        for name, meta in obj._failed_filter_names:
            failure_counter[name] += 1

    # (c) Per-filter stats: track, for each filter, how many objects reached it and how many passed.
    # This is a simple aggregate (not showing order beyond first occurrence).
    filter_stats = defaultdict(lambda: {'reached': 0, 'passed': 0})
    for obj in filterables:
        seen_filters = set()
        for name, passed, _ in obj.get_filter_sequence():
            if name not in seen_filters:
                filter_stats[name]['reached'] += 1
                if passed:
                    filter_stats[name]['passed'] += 1
                seen_filters.add(name)

    lines = []
    lines.append(f"Filter Analysis for {total_objects} objects")
    lines.append(
        f"Overall pass rate: {passed_objects}/{total_objects} ({passed_objects / total_objects * 100:.1f}%)")
    lines.append("")
    lines.append("Most Common Failure Points:")
    if failure_counter:
        for filter_name, count in failure_counter.most_common():
            lines.append(f"  {filter_name}: {count} failures ({count / total_objects * 100:.1f}%)")
    else:
        lines.append("  None")

    lines.append("")
    lines.append("Per-Filter Aggregate Stats:")
    for filter_name, stats in filter_stats.items():
        reached = stats['reached']
        passed = stats['passed']
        if reached:
            pass_rate = passed / reached * 100
        else:
            pass_rate = 0
        lines.append(
            f"  {filter_name}: reached {reached}/{total_objects} ({reached / total_objects * 100:.1f}%), "
            f"passed {passed}/{reached} ({pass_rate:.1f}%)"
        )

    # --- Section 2: Tag Statistics ---
    
    # Get objects that passed all filters
    passed_filterables = [obj for obj in filterables if obj.check_passed_all_filters()]
    
    # Count tags among passing objects
    tag_counter = Counter()
    all_tags_counter = Counter()  # For all objects regardless of pass/fail
    
    for obj in passed_filterables:
        for (tag, value) in obj.get_tags():
            tag_counter[tag] += 1
    
    for obj in filterables:
        for tag in obj.get_tags():
            all_tags_counter[tag] += 1
    
    lines.append("")
    lines.append(f"Tag Statistics (among {len(passed_filterables)} objects that passed all filters):")
    if tag_counter:
        for tag, count in tag_counter.most_common():
            percentage = count / len(passed_filterables) * 100 if passed_filterables else 0
            total_with_tag = all_tags_counter[tag]
            lines.append(f"  {tag}: {count}/{len(passed_filterables)} ({percentage:.1f}%) "
                        f"[{total_with_tag} total objects have this tag]")
    else:
        lines.append("  No tags found on passing objects")

    # --- Section 3: Flow Diagram ---
    # We build an aggregated tree structure based on each object's filter sequence.
    # (Note: this assumes that get_filter_sequence() returns the filters in the order they were applied.)
    tree = {"count": total_objects, "children": {}}

    for obj in filterables:
        seq = obj.get_filter_sequence()
        node = tree
        # For each filter event in the sequence, group by (filter name, passed flag)
        for name, passed, meta in seq:
            key = (name, passed)
            if key not in node["children"]:
                node["children"][key] = {"count": 0, "children": {}}
            node["children"][key]["count"] += 1
            node = node["children"][key]

    def print_tree(node: dict, indent: str = "") -> list[str]:
        """Recursively print the flow tree as a list of indented lines."""
        tree_lines = []
        for (name, passed), child in sorted(
                node.get("children", {}).items(), key=lambda x: x[1]["count"], reverse=True
        ):
            status = "Passed" if passed else "Failed"
            tree_lines.append(f"{indent}-> {name} ({status}) [n={child['count']}]")
            tree_lines.extend(print_tree(child, indent + "    "))
        return tree_lines

    lines.append("")
    lines.append("Filter Flow Diagram (aggregated across objects):")
    lines.append(f"Start [n={tree['count']}]")
    lines.extend(print_tree(tree, indent="  "))

    return "\n".join(lines)
