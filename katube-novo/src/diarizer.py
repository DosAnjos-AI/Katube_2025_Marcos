"""
Wrapper de compatibilidade para diarizer.py legado.
Redireciona para nova arquitetura modular Stage03Diarizer.

DEPRECATED: Use Stage03Diarizer diretamente.
Este arquivo existe apenas para manter compatibilidade com código legado.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pyannote.core import Annotation

# Import da nova arquitetura
from .stages.stage_03_diarizer import Stage03Diarizer
from .core.resource_manager import ResourceManager
from .config import Config

logger = logging.getLogger(__name__)


class EnhancedDiarizer:
    """
    Wrapper de compatibilidade para EnhancedDiarizer legado.

    DEPRECATED: Este é um wrapper para manter compatibilidade.
    Use Stage03Diarizer diretamente para novos desenvolvimentos.
    """

    def __init__(self, huggingface_token: Optional[str] = None):
        """
        Inicializa wrapper de compatibilidade.

        NOTA: Parâmetros relacionados a CUDA são ignorados - nova arquitetura força CPU.
        """
        # Token HF (usa Config se não fornecido)
        self.huggingface_token = huggingface_token or Config.HUGGINGFACE_TOKEN

        # Cria ResourceManager interno
        self._resource_manager = ResourceManager()

        # Cria Stage03Diarizer
        self._stage = Stage03Diarizer(
            resource_manager=self._resource_manager,
            huggingface_token=self.huggingface_token,
            model_name=Config.PYANNOTE_MODEL,
            sample_rate=Config.SAMPLE_RATE
        )

        # Carrega recursos
        try:
            self._stage.load_resources()
        except Exception as e:
            logger.error(f"[ERRO] Falha ao carregar recursos de diarização: {e}")
            # Não levanta exceção aqui para compatibilidade com código antigo
            # que verificava self.pipeline is None

        # Propriedades de compatibilidade
        self.device = 'cpu'
        self.sample_rate = Config.SAMPLE_RATE
        self.pipeline = self._stage.pipeline  # Expõe pipeline para compatibilidade

        logger.info("[COMPAT] EnhancedDiarizer wrapper inicializado (usando Stage03Diarizer)")

    def preprocess_audio(self, audio_path: Path) -> Tuple:
        """
        Pré-processa áudio (compatibilidade).

        Args:
            audio_path: Path para arquivo de áudio

        Returns:
            Tupla (waveform, sample_rate)
        """
        return self._stage.preprocess_audio(audio_path)

    def diarize_audio(self,
                      audio_path: Path,
                      num_speakers: Optional[int] = None) -> Annotation:
        """
        Executa diarização em arquivo de áudio.

        Args:
            audio_path: Path para arquivo de áudio
            num_speakers: Número esperado de speakers (opcional)

        Returns:
            Annotation pyannote
        """
        return self._stage.diarize_audio(audio_path, num_speakers)

    def annotation_to_dataframe(self,
                                 annotation: Annotation,
                                 audio_duration: Optional[float] = None) -> pd.DataFrame:
        """
        Converte Annotation para DataFrame.

        Args:
            annotation: Annotation pyannote
            audio_duration: Duração do áudio (opcional)

        Returns:
            DataFrame com resultados
        """
        return self._stage.annotation_to_dataframe(annotation, audio_duration)

    def save_rttm(self,
                  annotation: Annotation,
                  output_path: Path,
                  audio_filename: str) -> None:
        """
        Salva resultados em formato RTTM.

        Args:
            annotation: Annotation pyannote
            output_path: Path de saída
            audio_filename: Nome do arquivo de áudio
        """
        self._stage.save_rttm(annotation, output_path, audio_filename)

    def post_process_annotation(self,
                                 annotation: Annotation,
                                 min_duration: float = 0.5) -> Annotation:
        """
        Pós-processa annotation.

        Args:
            annotation: Annotation original
            min_duration: Duração mínima de segmento

        Returns:
            Annotation processada
        """
        return self._stage.post_process_annotation(annotation, min_duration)

    def analyze_speaker_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa estatísticas de speakers.

        Args:
            df: DataFrame com diarização

        Returns:
            Dict com estatísticas
        """
        return self._stage.analyze_speaker_statistics(df)

    def diarize_batch(self,
                      audio_files: List[Path],
                      output_dir: Path,
                      save_rttm: bool = True) -> Dict[str, Any]:
        """
        Diariza múltiplos arquivos em lote.

        Args:
            audio_files: Lista de paths de áudio
            output_dir: Diretório de saída
            save_rttm: Se deve salvar arquivos RTTM

        Returns:
            Dict com resultados de processamento
        """
        return self._stage.process(
            audio_paths=audio_files,
            output_dir=output_dir,
            save_rttm=save_rttm
        )

    def _get_audio_duration(self, audio_path: Path) -> float:
        """
        Obtém duração de áudio.

        Args:
            audio_path: Path para arquivo de áudio

        Returns:
            Duração em segundos
        """
        return self._stage._get_audio_duration(audio_path)

    def _detect_overlaps_in_annotation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta overlaps na annotation.

        Args:
            df: DataFrame com diarização

        Returns:
            Dict com estatísticas de overlaps
        """
        return self._stage._detect_overlaps_in_annotation(df)

    def _load_pipeline(self):
        """
        Método legado - agora é no-op pois recursos são carregados no __init__.

        Mantido para compatibilidade.
        """
        pass

    def __del__(self):
        """Cleanup ao destruir objeto."""
        try:
            self._stage.unload_resources()
        except:
            pass
