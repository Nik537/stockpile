"""Pydantic request/response models for the Stockpile API."""

from pydantic import BaseModel, Field

# =============================================================================
# Response Models
# =============================================================================


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str

    model_config = {"json_schema_extra": {"examples": [{"message": "Operation completed successfully"}]}}


class RootResponse(BaseModel):
    """Root endpoint response."""

    message: str
    version: str

    model_config = {"json_schema_extra": {"examples": [{"message": "Stockpile API", "version": "1.0.0"}]}}


class HealthResponse(BaseModel):
    """Health check response."""

    status: str

    model_config = {"json_schema_extra": {"examples": [{"status": "healthy"}]}}


class JobProgressResponse(BaseModel):
    """Job progress details."""

    stage: str
    percent: int = Field(ge=0, le=100)
    message: str


class JobResponse(BaseModel):
    """B-roll processing job response."""

    id: str
    video_filename: str
    status: str
    created_at: str
    updated_at: str
    progress: JobProgressResponse
    preferences: dict | None = None
    error: str | None = None
    output_dir: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "video_filename": "my_video.mp4",
                    "status": "processing",
                    "created_at": "2026-02-08T12:00:00",
                    "updated_at": "2026-02-08T12:05:00",
                    "progress": {"stage": "transcribing", "percent": 25, "message": "Transcribing audio..."},
                }
            ]
        }
    }


class JobListResponse(BaseModel):
    """List of jobs response."""

    jobs: list[JobResponse]


class JobCreatedResponse(BaseModel):
    """Response when a job is created."""

    job_id: str
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "message": "Video uploaded successfully, processing started",
                }
            ]
        }
    }


class OutlierSearchCreatedResponse(BaseModel):
    """Response when an outlier search is started."""

    search_id: str
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"search_id": "550e8400-e29b-41d4-a716-446655440000", "message": "Outlier search started"}]
        }
    }


class ServiceStatusResponse(BaseModel):
    """Generic service status response."""

    configured: bool = False
    available: bool = False
    error: str | None = None


class TTSStatusResponse(BaseModel):
    """TTS service status response."""

    colab: dict
    runpod: dict


class TTSEndpointResponse(BaseModel):
    """Response after setting TTS endpoint."""

    message: str
    connected: bool = False
    server_url: str | None = None


class PublicTTSResponse(BaseModel):
    """Public TTS generation response."""

    audio_url: str
    cost: float
    voice: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "audio_url": "https://image.runpod.ai/chatterbox-turbo/abc123.wav",
                    "cost": 0.00432,
                    "voice": "Lucy",
                }
            ]
        }
    }


class ImageGenerationResultResponse(BaseModel):
    """Image generation result response."""

    images: list[dict]
    model: str
    seed: int | None = None
    cost: float | None = None


class BulkImagePromptsResponse(BaseModel):
    """Response for bulk image prompt generation."""

    job_id: str
    prompts: list[dict]
    estimated_cost: float
    estimated_time_seconds: float


class BulkImageGenerateResponse(BaseModel):
    """Response for starting bulk image generation."""

    job_id: str
    status: str
    total_count: int
    estimated_cost: float


# =============================================================================
# Request Models
# =============================================================================


class OutlierSearchParams(BaseModel):
    """Parameters for outlier search."""

    topic: str
    max_channels: int = 10
    min_score: float = 3.0
    days: int | None = None
    include_shorts: bool = False
    min_subs: int | None = None
    max_subs: int | None = None


class TTSEndpointRequest(BaseModel):
    """Request body for setting TTS endpoint."""

    url: str


class TTSGenerateRequest(BaseModel):
    """Request body for TTS generation."""

    text: str
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8


class ImageGenerateRequest(BaseModel):
    """Request body for image generation."""

    prompt: str
    model: str = "runware-flux-klein-4b"
    width: int = 1024
    height: int = 1024
    num_images: int = 1
    seed: int | None = None
    guidance_scale: float = 7.5


class ImageEditRequestBody(BaseModel):
    """Request body for image editing."""

    prompt: str
    image_url: str  # URL or base64 data URL
    model: str = "nano-banana-pro"
    strength: float = 0.75
    seed: int | None = None
    guidance_scale: float = 7.5
    mask_image: str | None = None  # base64 data URL for inpainting mask


class RunPodImageGenerateRequest(BaseModel):
    """Request body for RunPod image generation."""

    prompt: str
    model: str = "nano-banana-pro"
    width: int = 1024
    height: int = 1024
    seed: int | None = None


class PublicTTSRequest(BaseModel):
    """Request body for public TTS generation."""

    text: str
    voice: str = "Lucy"


class VoiceResponse(BaseModel):
    """Voice library entry response."""

    id: str
    name: str
    is_preset: bool
    audio_path: str
    created_at: str
    duration_seconds: float


class BulkImagePromptsRequest(BaseModel):
    """Request body for generating bulk image prompts."""

    meta_prompt: str
    count: int = 50


class BulkImagePromptItem(BaseModel):
    """A single prompt item for bulk generation."""

    index: int
    prompt: str
    rendering_style: str
    mood: str
    composition: str
    has_text_space: bool = False


class BulkImageGenerateRequest(BaseModel):
    """Request body for generating bulk images."""

    job_id: str
    prompts: list[BulkImagePromptItem]
    model: str = "runware-flux-klein-4b"
    width: int = 1024
    height: int = 1024
