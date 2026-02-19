"""Models for AI storyboard generation."""

from dataclasses import dataclass, field


@dataclass
class CharacterProfile:
    """A character in the storyboard with consistent visual appearance."""

    name: str
    appearance: str
    clothing: str
    accessories: str

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "appearance": self.appearance,
            "clothing": self.clothing,
            "accessories": self.accessories,
        }


@dataclass
class StoryboardScene:
    """A single scene in the storyboard."""

    scene_number: int
    description: str
    camera_angle: str
    character_action: str
    environment: str
    image_prompt: str

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "scene_number": self.scene_number,
            "description": self.description,
            "camera_angle": self.camera_angle,
            "character_action": self.character_action,
            "environment": self.environment,
            "image_prompt": self.image_prompt,
        }


@dataclass
class StoryboardPlan:
    """Complete storyboard plan with characters and scenes."""

    title: str
    characters: list[CharacterProfile]
    scenes: list[StoryboardScene]
    style_guide: str
    aspect_ratio: str

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "title": self.title,
            "characters": [c.to_dict() for c in self.characters],
            "scenes": [s.to_dict() for s in self.scenes],
            "style_guide": self.style_guide,
            "aspect_ratio": self.aspect_ratio,
        }


@dataclass
class StoryboardJob:
    """Job tracking storyboard generation progress."""

    job_id: str
    status: str
    plan: StoryboardPlan | None = None
    reference_images: dict[str, str] = field(default_factory=dict)
    scene_images: list[dict] = field(default_factory=list)
    error: str | None = None
    total_cost: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "plan": self.plan.to_dict() if self.plan else None,
            "reference_images": self.reference_images,
            "scene_images": self.scene_images,
            "error": self.error,
            "total_cost": self.total_cost,
        }
