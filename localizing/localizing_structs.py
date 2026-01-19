import dataclasses
from lmwrapper.structs import LmPrompt, LmChatDialog, LmChatTurn, ChatGptRoles
from dataclasses import fields
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar, Generic, Protocol, Callable
from typing import Any
import numpy as np
from lmwrapper.utils import StrEnum
from synthegrator.code_problems import CodeSolution, LmCodeSolution
from synthegrator.solution_eval import SolutionEvaluation
from localizing.filter_helpers import FilterTrackingMixin
from localizing.problem_processing import solve_to_text, tokenize_text
from synthegrator.synthdatasets import DatasetName
from protogrator import LmCodeSolutionSet
from numpy.typing import NDArray


@dataclass(kw_only=True)
class AllCorrectEstimate:
    prob_all_correct: float
    prob_all_correct_name: str
    actual: bool = None


@dataclass(kw_only=True)
class BaseLocalization(FilterTrackingMixin):
    dataset_name: DatasetName
    gen_model_name: str
    base_solve: LmCodeSolution
    base_eval: SolutionEvaluation = None
    gt_fix_solve: CodeSolution = None
    gt_fix_eval: SolutionEvaluation = None
    gen_model_properties: dict[str, Any] = dataclasses.field(
        default_factory=dict,
    )
    all_correct_estimate: AllCorrectEstimate | None = None
    preset_base_text: str | None = None
    preset_gt_fix_text: str | None = None
    preset_full_convo: LmChatDialog | None = None

    def __post_init__(self):
        self._base_text_cache = None
        self._gt_fix_text_cache = None

    def get_base_text(self) -> str:
        if self.preset_base_text is not None:
            return self.preset_base_text
        if self._base_text_cache is None:
            self._base_text_cache = solve_to_text(self.base_solve, self.dataset_name)
        return self._base_text_cache

    def get_base_convo(
        self,
        include_last_answer: bool = True,
    ) -> LmChatDialog:
        if self.preset_full_convo is not None:
            if include_last_answer:
                return self.preset_full_convo
            return self.preset_full_convo[:-1]

        if self.base_solve is None:
            return None

        if not include_last_answer:
            vals = self.base_solve.prompt.get_text_as_chat()
            return vals
        else:
            vals = self.base_solve.lm_prediction.prompt.get_text_as_chat()
            return LmChatDialog([
                *vals,
                LmChatTurn(
                    role=ChatGptRoles.assistant,
                    content=self.base_solve.lm_prediction.completion_text,
                )
            ])

    @property
    def is_base_success(self) -> bool:
        if (
            self.preset_base_text is not None
            and self.preset_gt_fix_text is not None
            and self.base_eval is None
        ):
            return self.preset_base_text == self.preset_gt_fix_text
        if self.base_eval is None:
            return False
        if self.base_eval.main_metric is None:
            return False
        return self.base_eval.main_metric.is_success

    def get_gt_fix_text(self) -> str:
        if self.preset_gt_fix_text is not None:
            return self.preset_gt_fix_text
        if self.gt_fix_solve is None:
            if self.base_eval.main_metric.is_success:
                return self.get_base_text()
            raise ValueError("No ground truth fix solve")
        if self._gt_fix_text_cache is None:
            self._gt_fix_text_cache = solve_to_text(self.gt_fix_solve, self.dataset_name)
        return self._gt_fix_text_cache

    @classmethod
    def copy_from(cls, other: 'BaseLocalization', **extra_fields):
        """Creates a shallow copy of the localization"""
        # Gather all fields and their values from 'other'
        field_values = {
            f.name: getattr(other, f.name) 
            for f in fields(other)
        }
        # Create a new instance of cls using these values
        new_instance = cls(**field_values)
        # Copy any filter states or other attributes required
        new_instance.copy_filters_from(other)
        # Update any extra fields
        for field_name, field_value in extra_fields.items():
            setattr(new_instance, field_name, field_value)
        return new_instance

    def copy(self):
        return self.copy_from(self)

T_l = TypeVar('T_l', bound=BaseLocalization)
T_lo = TypeVar('T_lo', bound=BaseLocalization)


class LocalizationList(Generic[T_l]):
    def __init__(
        self,
        items: list[T_l] | None = None,
    ):
        if items is None:
            items = []
        self._items = items

    def __len__(self):
        return len(self._items)
        
    def len_passed_filtered(self) -> int:
        """Returns the number of items that pass all filters."""
        return len(list(self.iter_passed_filtered()))

    def get_dataset_name_set(self) -> set[DatasetName]:
        return {item.dataset_name for item in self._items}

    def get_only_datset_name(self) -> DatasetName:
        dataset_name_set = self.get_dataset_name_set()
        assert len(dataset_name_set) == 1
        return dataset_name_set.pop()

    def get_gen_model_name_set(self) -> set[str]:
        return {item.gen_model_name for item in self._items}

    def get_only_gen_model_name(self) -> str:
        gen_model_name_set = self.get_gen_model_name_set()
        assert len(gen_model_name_set) == 1
        return gen_model_name_set.pop()

    def iter_passed_filtered(self) -> Iterable[T_l]:
        for item in list(self._items):
            if item.check_passed_all_filters():
                yield item

    def iter_all(self) -> Iterable[T_l]:
        yield from self._items

    def append(self, item: T_l):
        self._items.append(item)
        assert isinstance(item, self._items[0].__class__)

    def extend(self, other: 'LocalizationList[T_l]'):
        """Extend this localization list with items from another list."""
        if len(self._items) > 0 and len(other._items) > 0:
            # Check that the items are compatible types
            assert (issubclass(other._items[0].__class__, self._items[0].__class__) or 
                   issubclass(self._items[0].__class__, other._items[0].__class__)), \
                f"Cannot extend localization list with items of incompatible type: "\
                f"{self._items[0].__class__} vs {other._items[0].__class__}"
        
        self._items.extend(other._items)

    def base_problems(self) -> Iterable[CodeSolution]:
        for item in self._items:
            yield item.base_solve.problem

    def get_gen_property_set(self, value: str):
        return {item.gen_model_properties[value] for item in self._items}

    def get_only_gen_property(self, value: str):
        gen_property_set = self.get_gen_property_set(value)
        assert len(gen_property_set) == 1
        return gen_property_set.pop()

    def copy_with_type_change(
        self,
        new_type: type[T_lo],
    ) -> 'LocalizationList[T_lo]':
        # check that the new type is a subclass of the old type
        if len(self) > 0 and not issubclass(new_type, self._items[0].__class__):
            raise ValueError(f"{new_type} is not a subclass of {self._items[0].__class__}")
        new_items = [new_type.copy_from(item) for item in self._items]
        return LocalizationList(new_items)

    def copy(self):
        """Makes a new list that contains a shallow copy of each item in the list"""
        return LocalizationList([item.copy() for item in self._items])

    def __iter__(self):
        raise TypeError(
            "Please use either iter_all() or iter_passed_filtered() "
            "(we avoid __iter__ to avoid unexpected behavior)"
        )

    def __str__(self):
        return (
            f"LocalizationList(\n"
            f"len {len(self)}\n"
            f"len passed {self.len_passed_filtered()}\n"
            f"dataset_name_set {self.get_dataset_name_set()}\n"
            f"gen_model_name_set {self.get_gen_model_name_set()}\n"
            f")\n"
        )
    
    def __repr__(self):
        return str(self)


class MultiSampleMode(StrEnum):
    from_prompt = 'from_prompt'
    repair = 'repair'


@dataclass(frozen=True)
class MultiSamplingConfig:
    multi_temperature: float 
    target_num_samples: int
    mode: MultiSampleMode


@dataclass(kw_only=True)
class MultiSamplingLocalization(BaseLocalization):
    """A localization that relies on having a sample
    multiple completions for the problem."""
    samples: LmCodeSolutionSet = None
    config: MultiSamplingConfig = None


class HasTokenizedInfo(Protocol):
    base_tokens: list[str]
    gt_fix_tokens: list[str]


@dataclass(kw_only=True)
class TokenizedLocalization(BaseLocalization):
    """A style localization which relies
    on a tokenization"""
    base_tokens: list[str] = None
    gt_fix_tokens: list[str] = None
    tokenizer_key: str = None
    _gt_base_token_keeps: NDArray[np.bool_] | None = dataclasses.field(default=None, repr=False)

    def __post_init__(self):
        super().__post_init__()
        # Validate initial values if provided through constructor
        if hasattr(self, 'gt_base_token_keeps'):
            self.gt_base_token_keeps = self.gt_base_token_keeps

    def tokenize_with_same_tokenizer(self, text: str):
        return tokenize_text(self.tokenizer_key, text)

    def get_line_boundries(self) -> list[int]:
        """Get the tokens that start a new line."""
        if self.base_tokens is None:
            return None
        if len(self.base_tokens) == 0:
            return []
        lines = [0]
        for i, token in enumerate(self.base_tokens):
            if "\n" in token:
                if token[-1] == "\n":
                    lines.append(i + 1)
                else:
                    lines.append(i)
        if lines[-1] > len(self.base_tokens):
            lines[-1] = len(self.base_tokens)
        if lines[-1] != len(self.base_tokens):
            lines.append(len(self.base_tokens))
        return lines

    def get_line_spans(self, ignore_empty_lines: bool = True) -> list[tuple[int, int]]:
        line_boundaries = self.get_line_boundries()
        
        # Initialize line spans
        line_spans = []
        
        # Convert boundaries to spans (start_idx, end_idx)
        start_idx = 0
        for end_idx in line_boundaries:
            if not ignore_empty_lines:
                line_spans.append((start_idx, end_idx))
            elif end_idx > start_idx:
                # Only add non-empty lines
                line_spans.append((start_idx, end_idx))
            start_idx = end_idx
        
        return line_spans

    @property
    def gt_base_token_keeps(self) -> NDArray[np.bool_] | None:
        """Get the array of ground truth keep flags for each token."""
        return self._gt_base_token_keeps

    @gt_base_token_keeps.setter
    def gt_base_token_keeps(self, value: NDArray[np.bool_] | list[bool] | None):
        """Set the array of ground truth keep flags for each token.

        Args:
            value: Array of boolean values representing ground truth keep flags, or None.
                  Must be same length as base_tokens if not None.

        Raises:
            ValueError: If value is not None and length doesn't match base_tokens.
        """
        if value is not None:
            # Convert to numpy array if necessary
            value_array = np.asarray(value, dtype=np.bool_)
            if len(value_array) != len(self.base_tokens):
                raise ValueError(
                    f"gt_base_token_keeps length ({len(value_array)}) must match "
                    f"base_tokens length ({len(self.base_tokens)})"
                )
            self._gt_base_token_keeps = value_array
        else:
            self._gt_base_token_keeps = None


@dataclass(kw_only=True)
class MultiTokenizedLocalization(
    MultiSamplingLocalization,
    TokenizedLocalization,
):
    pass


@dataclass(kw_only=True)
class TokenEqualsLocalization(TokenizedLocalization):
    """A style localization where every token has associated 
    estimates and ground truth values.

    Properties are validated to ensure estimate_keeps matches the length of base_tokens when set.
    All numeric arrays are stored as numpy arrays.
    """
    _estimate_keeps: NDArray[np.float32] | None = dataclasses.field(default=None, repr=False)

    def __post_init__(self):
        super().__post_init__()
        # Validate initial values if provided through constructor
        if hasattr(self, 'estimate_keeps'):
            self.estimated_keeps = self.estimated_keeps

    @property
    def estimated_keeps(self) -> NDArray[np.float32] | None:
        """Get the array of estimated keep probabilities for each token."""
        return self._estimate_keeps

    @estimated_keeps.setter
    def estimated_keeps(self, value: NDArray[np.float32] | list[float] | None):
        """Set the array of estimated keep probabilities for each token.

        Args:
            value: Array of float values representing keep probabilities, or None.
                  Must be same length as base_tokens if not None.

        Raises:
            ValueError: If value is not None and length doesn't match base_tokens.
        """
        if value is not None:
            # Convert to numpy array if necessary
            value_array = np.asarray(value, dtype=np.float32)
            if len(value_array) != len(self.base_tokens):
                raise ValueError(
                    f"estimate_keeps length ({len(value_array)}) must match "
                    f"base_tokens length ({len(self.base_tokens)})"
                )
            self._estimate_keeps = value_array
        else:
            self._estimate_keeps = None


@dataclass(kw_only=True)
class MultisampleTokenEqualsLocalization(
    MultiTokenizedLocalization,
    TokenEqualsLocalization,
):
    keep_tallys: list[list[int]] = None

