"""Service singletons and dependency injection for the Stockpile API."""

from services.ai_service import AIService
from services.bulk_image_service import BulkImageService
from services.dataset_generator_service import DatasetGeneratorService
from services.image_generation_service import ImageGenerationService
from services.music_service import MusicService
from services.storyboard_service import StoryboardService
from services.tts_service import TTSService
from services.voice_library import VoiceLibrary
from utils.config import load_config

# Service singletons
_tts_service: TTSService | None = None
_image_gen_service: ImageGenerationService | None = None
_bulk_image_service: BulkImageService | None = None
_ai_service: AIService | None = None
_voice_library: VoiceLibrary | None = None
_music_service: MusicService | None = None
_dataset_gen_service: DatasetGeneratorService | None = None
_storyboard_service: StoryboardService | None = None


def get_image_gen_service() -> ImageGenerationService:
    """Get or create the image generation service instance."""
    global _image_gen_service
    if _image_gen_service is None:
        config = load_config()
        _image_gen_service = ImageGenerationService(
            api_key="",  # fal.ai no longer used
            runpod_api_key=config.get("runpod_api_key", ""),
            runware_api_key=config.get("runware_api_key", ""),
        )
    return _image_gen_service


def get_tts_service() -> TTSService:
    """Get or create the TTS service instance."""
    global _tts_service
    if _tts_service is None:
        config = load_config()
        _tts_service = TTSService(config.get("tts_server_url", ""))
    return _tts_service


def get_ai_service() -> AIService:
    """Get or create the AI service instance."""
    global _ai_service
    if _ai_service is None:
        config = load_config()
        _ai_service = AIService(
            api_key=config.get("gemini_api_key", ""),
            model_name=config.get("gemini_model", "gemini-3-flash-preview"),
        )
    return _ai_service


def get_bulk_image_service() -> BulkImageService:
    """Get or create the bulk image service instance."""
    global _bulk_image_service
    if _bulk_image_service is None:
        _bulk_image_service = BulkImageService(
            ai_service=get_ai_service(),
            image_gen_service=get_image_gen_service(),
        )
    return _bulk_image_service


def get_voice_library() -> VoiceLibrary:
    """Get or create the voice library instance."""
    global _voice_library
    if _voice_library is None:
        _voice_library = VoiceLibrary()
    return _voice_library


def get_music_service() -> MusicService:
    """Get or create the music service instance."""
    global _music_service
    if _music_service is None:
        _music_service = MusicService()
    return _music_service


def get_dataset_gen_service() -> DatasetGeneratorService:
    """Get or create the dataset generator service instance."""
    global _dataset_gen_service
    if _dataset_gen_service is None:
        _dataset_gen_service = DatasetGeneratorService(
            ai_service=get_ai_service(),
            image_gen_service=get_image_gen_service(),
        )
    return _dataset_gen_service


def get_storyboard_service() -> StoryboardService:
    """Get or create the storyboard service instance."""
    global _storyboard_service
    if _storyboard_service is None:
        _storyboard_service = StoryboardService(
            ai_service=get_ai_service(),
            image_gen_service=get_image_gen_service(),
        )
    return _storyboard_service
