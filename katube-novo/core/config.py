"""
Configuration settings for the audio processing pipeline.
CPU-First architecture - Zero GPU dependencies.
"""
import os
from pathlib import Path

# Importacao opcional de dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Continua sem .env se dotenv nao disponivel


class Config:
    # Audio settings
    AUDIO_FORMAT = os.getenv('AUDIO_FORMAT', 'flac')
    AUDIO_QUALITY = os.getenv('AUDIO_QUALITY', 'best')
    SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '24000'))
    SEGMENT_MIN_DURATION = float(os.getenv('SEGMENT_MIN_DURATION', '10.0'))
    SEGMENT_MAX_DURATION = float(os.getenv('SEGMENT_MAX_DURATION', '15.0'))
    SEGMENT_OVERLAP = float(os.getenv('SEGMENT_OVERLAP', '0.5'))

    # Diarization settings
    PYANNOTE_MODEL = os.getenv('PYANNOTE_MODEL', 'pyannote/speaker-diarization-3.1')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

    # Voice overlap detection
    OVERLAP_THRESHOLD = float(os.getenv('OVERLAP_THRESHOLD', '0.9'))  # 90% overlap required
    MIN_SPEECH_DURATION = float(os.getenv('MIN_SPEECH_DURATION', '0.5'))

    # Audio segmentation limits
    MAX_SEGMENTS = int(os.getenv('MAX_SEGMENTS', '5000'))  # Maximum segments per audio

    # MOS Quality Filter settings (OBRIGATORIO)
    MOS_THRESHOLD = float(os.getenv('MOS_THRESHOLD', '3.0'))  # Minimum MOS score to accept
    # ENABLE_MOS_FILTER sempre True - filtro e obrigatorio

    # YouTube API settings
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

    # Directories
    BASE_DIR = Path(__file__).parent.parent
    AUDIOS_BAIXADOS_DIR = Path(os.getenv('AUDIOS_BAIXADOS_DIR', r'C:\Users\Usuario\Desktop\katube-novo\audios_baixados'))
    OUTPUT_DIR = AUDIOS_BAIXADOS_DIR / "output"

    # YouTube download settings
    YOUTUBE_FORMAT = "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best/worst"

    # STT preparation settings
    MAX_SEGMENT_SIZE = 25 * 1024 * 1024  # 25MB max per segment for STT

    # Resource Manager settings (NEW)
    MAX_CACHE_SIZE_MB = int(os.getenv('MAX_CACHE_SIZE_MB', '4096'))  # 4GB cache limit
    ENABLE_LAZY_LOADING = os.getenv('ENABLE_LAZY_LOADING', 'True').lower() == 'true'

    # Logging settings (NEW)
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
    LOG_TO_FILE = os.getenv('LOG_TO_FILE', 'False').lower() == 'true'
    LOG_FILE_PATH = Path(os.getenv('LOG_FILE_PATH', 'logs/katube.log'))

    # Device settings - CPU ONLY (NO GPU)
    DEVICE = 'cpu'  # Fixed to CPU for CPU-first architecture
    TORCH_DTYPE = 'float32'  # Fixed to float32 for CPU

    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        for dir_path in [cls.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create log directory if logging to file
        if cls.LOG_TO_FILE:
            cls.LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_memory_config(cls) -> dict:
        """
        Retorna configuracoes de memoria.

        Returns:
            Dict com configuracoes de cache e lazy loading
        """
        return {
            'max_cache_size_mb': cls.MAX_CACHE_SIZE_MB,
            'enable_lazy_loading': cls.ENABLE_LAZY_LOADING,
            'device': cls.DEVICE,
        }

    @classmethod
    def validate(cls) -> bool:
        """
        Valida configuracoes essenciais.

        Returns:
            True se configuracoes validas, False caso contrario
        """
        # Verificar token HuggingFace
        if not cls.HUGGINGFACE_TOKEN:
            print("[AVISO] HUGGINGFACE_TOKEN nao configurado")
            return False

        # Verificar diretorios
        if not cls.OUTPUT_DIR.exists():
            print(f"[AVISO] OUTPUT_DIR nao existe: {cls.OUTPUT_DIR}")

        return True
