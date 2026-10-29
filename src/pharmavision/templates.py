"""Test templates: everything specific to one behaviour test."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TestTemplate(BaseModel):
    """What a run needs to know about the test that was filmed."""

    model_config = ConfigDict(frozen=True)

    test_type: Literal["open_field"]
    floor_prompt: str
    animal_prompt: str
    max_animals: int = Field(ge=1)
    arena_width_cm: float = Field(gt=0)
    arena_height_cm: float = Field(gt=0)
    centre_area_fraction: float = Field(gt=0, lt=1)
    """The centre zone: the floor polygon shrunk about its centroid to this fraction of its area."""
    protocol_duration_s: float = Field(gt=0)
    duration_tolerance_s: float = Field(default=0.5, ge=0)
    """Container rounding: a 10-min recording can probe as 599.96 s."""


# Sturman et al. 2020 (doi:10.1038/s41386-020-0776-y): 45 x 45 cm arena, 10 min, centre = the
# central 50% of the floor.
OPEN_FIELD = TestTemplate(
    test_type="open_field",
    floor_prompt="arena floor",
    animal_prompt="mouse",
    max_animals=1,
    arena_width_cm=45.0,
    arena_height_cm=45.0,
    centre_area_fraction=0.5,
    protocol_duration_s=600.0,
)
