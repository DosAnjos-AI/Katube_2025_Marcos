"""
Wrapper de compatibilidade para mos_filter.py legado.
Redireciona para nova arquitetura modular Stage02MOSFilter.

DEPRECATED: Use Stage02MOSFilter diretamente.
Este arquivo existe apenas para manter compatibilidade com código legado.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Import da nova arquitetura
from .stages.stage_02_mos_filter import Stage02MOSFilter
from .core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class MOSQualityFilter:
    """
    Wrapper de compatibilidade para MOSQualityFilter legado.

    DEPRECATED: Este é um wrapper para manter compatibilidade.
    Use Stage02MOSFilter diretamente para novos desenvolvimentos.
    """

    def __init__(self,
                 mos_threshold: float = 2.5,
                 sample_rate: int = 24000,
                 use_cuda: bool = False):
        """
        Inicializa wrapper de compatibilidade.

        NOTA: use_cuda é ignorado - nova arquitetura força CPU.
        """
        if use_cuda:
            logger.warning("[AVISO] use_cuda=True ignorado - nova arquitetura força CPU")

        # Cria ResourceManager interno
        self._resource_manager = ResourceManager()

        # Cria Stage02MOSFilter
        self._stage = Stage02MOSFilter(
            resource_manager=self._resource_manager,
            mos_threshold=mos_threshold,
            sample_rate=sample_rate
        )

        # Carrega recursos
        self._stage.load_resources()

        self.mos_threshold = mos_threshold
        self.sample_rate = sample_rate
        self.use_cuda = False  # Sempre CPU na nova arquitetura
        self.device = 'cpu'

        logger.info("[COMPAT] MOSQualityFilter wrapper inicializado (usando Stage02MOSFilter)")

    def predict_mos_score(self, audio_path: Path) -> float:
        """
        Prediz score MOS para arquivo.

        Args:
            audio_path: Path para arquivo de áudio

        Returns:
            Score MOS (1.0-5.0)
        """
        return self._stage.predict_mos_score(audio_path)

    def filter_audio_segments(self,
                               segment_paths: List[Path],
                               output_dir: Optional[Path] = None) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Filtra segmentos em 3 categorias.

        Args:
            segment_paths: Lista de paths
            output_dir: Diretório de saída (opcional)

        Returns:
            Tupla (approved, intermediate, rejected)
        """
        result = self._stage.process(segment_paths, output_dir)

        return (
            result['approved'],
            result['intermediate'],
            result['rejected']
        )

    def get_quality_report(self, segment_paths: List[Path]) -> dict:
        """
        Gera relatório de qualidade.

        Args:
            segment_paths: Lista de paths

        Returns:
            Dict com estatísticas
        """
        return self._stage.get_quality_report(segment_paths)

    def __del__(self):
        """Cleanup ao destruir objeto."""
        try:
            self._stage.unload_resources()
        except:
            pass
