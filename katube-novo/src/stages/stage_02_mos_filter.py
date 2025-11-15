"""
Stage 02: MOS Quality Filter
Filtra segmentos de áudio por qualidade usando modelo SHEET
CPU-only, lazy loading (~1GB modelo)
"""

import torch
import torchaudio
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import shutil

from ..core.base_processor import BaseProcessor
from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class Stage02MOSFilter(BaseProcessor):
    """
    Etapa 02: Filtro de qualidade MOS usando modelo SHEET.

    Input: Lista de Paths para segmentos de áudio
    Output: Dict com segmentos aprovados/intermediários/rejeitados

    Thresholds:
    - Aprovado: MOS >= 3.0
    - Intermediário: 2.5 <= MOS < 3.0
    - Rejeitado: MOS < 2.5
    """

    def __init__(self,
                 resource_manager: ResourceManager,
                 mos_threshold: float = 2.5,
                 sample_rate: int = 24000):
        """
        Inicializa filtro MOS.

        Args:
            resource_manager: Gerenciador de recursos
            mos_threshold: Threshold mínimo para aceitação (padrão: 2.5)
            sample_rate: Taxa de amostragem para processamento (padrão: 24000)
        """
        super().__init__(resource_manager)
        self.mos_threshold = mos_threshold
        self.sample_rate = sample_rate
        self.predictor = None  # Lazy loaded
        self.device = 'cpu'  # Forçado para CPU

        logger.info(f"[CONFIG] MOS Filter: threshold={mos_threshold}, sr={sample_rate}, device=cpu")

    def load_resources(self) -> None:
        """Carrega modelo SHEET sob demanda (lazy loading)."""
        if self.predictor is None:
            logger.info("[LOAD] Carregando modelo MOS (SHEET)...")

            self.predictor = self.resource_manager.load_model(
                model_key='mos_sheet',
                loader_func=self._load_sheet_model
            )

            logger.info("[OK] Modelo MOS carregado em CPU")

    def _load_sheet_model(self):
        """
        Loader function para modelo SHEET.

        Returns:
            Predictor SHEET configurado para CPU
        """
        try:
            # Carrega modelo do torch.hub
            predictor = torch.hub.load(
                "unilight/sheet:v0.1.0",
                "default",
                trust_repo=True,
                force_reload=False
            )

            # FORÇA CPU (não usar GPU)
            predictor.model.cpu()

            logger.info("[OK] SHEET model loaded on CPU")
            return predictor

        except Exception as e:
            logger.error(f"[ERRO] Falha ao carregar SHEET: {e}")
            logger.error("[INFO] Verifique: pip install torch torchaudio")
            raise RuntimeError(f"SHEET é obrigatório para MOS filter: {e}")

    @torch.no_grad()
    def predict_mos_score(self, audio_path: Path) -> float:
        """
        Prediz score MOS para arquivo de áudio.

        Args:
            audio_path: Path para arquivo de áudio

        Returns:
            Score MOS (1.0-5.0)
        """
        try:
            # Verifica se arquivo existe
            if not audio_path.exists():
                logger.error(f"[ERRO] Arquivo não encontrado: {audio_path}")
                return 1.0

            # Verifica tamanho mínimo
            file_size = audio_path.stat().st_size
            if file_size < 1024:  # < 1KB
                logger.error(f"[ERRO] Arquivo muito pequeno: {audio_path} ({file_size} bytes)")
                return 1.0

            # Usa SHEET predictor
            if self.predictor is None:
                raise RuntimeError("[ERRO] Predictor MOS não inicializado")

            score = self.predictor.predict(wav_path=str(audio_path))
            return float(score)

        except Exception as e:
            logger.error(f"[ERRO] Falha ao predizer MOS para {audio_path.name}: {e}")
            return 1.0  # Score baixo em caso de erro

    @torch.no_grad()
    def process(self,
                segment_paths: List[Path],
                output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Filtra segmentos de áudio por qualidade MOS.

        Args:
            segment_paths: Lista de paths para segmentos
            output_dir: Diretório base para salvar segmentos categorizados (opcional)

        Returns:
            Dict com:
            - 'approved': List[Path] (MOS >= 3.0)
            - 'intermediate': List[Path] (2.5 <= MOS < 3.0)
            - 'rejected': List[Path] (MOS < 2.5)
            - 'stats': Dict com estatísticas
        """
        with self.managed_processing():
            approved_segments = []
            intermediate_segments = []
            rejected_segments = []

            # Cria diretórios de saída se especificado
            approved_dir = None
            intermediate_dir = None
            rejected_dir = None

            if output_dir:
                approved_dir = output_dir / "audios_acima_3,0_MOS"
                intermediate_dir = output_dir / "audios_entre_2,5_e_3,0_MOS"
                rejected_dir = output_dir / "audios_abaixo_2,5_MOS"

                approved_dir.mkdir(parents=True, exist_ok=True)
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                rejected_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"[INFO] Filtrando {len(segment_paths)} segmentos com classificação 3-tier MOS")

            # Importa naming utils se disponível
            try:
                from ..naming_utils import extract_base_name, generate_standard_name
                has_naming_utils = True
            except ImportError:
                logger.warning("[AVISO] naming_utils não disponível, usando nomes simples")
                has_naming_utils = False

            # Processa cada segmento
            for i, segment_path in enumerate(segment_paths):
                try:
                    # Prediz MOS score
                    mos_score = self.predict_mos_score(segment_path)

                    logger.info(f"[SCORE] {segment_path.name}: MOS = {mos_score:.2f}")

                    # Classifica e salva
                    if mos_score >= 3.0:
                        # APROVADO
                        approved_segments.append(segment_path)
                        logger.info(f"[OK] Aprovado: {segment_path.name} (MOS: {mos_score:.2f})")

                        if output_dir and approved_dir:
                            self._save_segment(
                                segment_path, approved_dir, mos_score,
                                "approved", i + 1, has_naming_utils
                            )

                    elif mos_score >= 2.5:
                        # INTERMEDIÁRIO
                        intermediate_segments.append(segment_path)
                        logger.info(f"[INFO] Intermediário: {segment_path.name} (MOS: {mos_score:.2f})")

                        if output_dir and intermediate_dir:
                            self._save_segment(
                                segment_path, intermediate_dir, mos_score,
                                "intermediate", i + 1, has_naming_utils
                            )

                    else:
                        # REJEITADO
                        rejected_segments.append(segment_path)
                        logger.warning(f"[REJECT] Rejeitado: {segment_path.name} (MOS: {mos_score:.2f})")

                        if output_dir and rejected_dir:
                            self._save_segment(
                                segment_path, rejected_dir, mos_score,
                                "rejected", i + 1, has_naming_utils
                            )

                    # Log de progresso
                    if (i + 1) % 10 == 0:
                        logger.info(f"[PROGRESS] Processados {i + 1}/{len(segment_paths)} segmentos")

                except Exception as e:
                    logger.error(f"[ERRO] Falha ao processar {segment_path.name}: {e}")
                    rejected_segments.append(segment_path)

                    # Salva segmento com erro
                    if output_dir and rejected_dir:
                        try:
                            error_filename = f"mos_error_{segment_path.name}"
                            error_path = rejected_dir / error_filename
                            shutil.copy2(segment_path, error_path)
                        except Exception as save_error:
                            logger.warning(f"[AVISO] Não foi possível salvar segmento com erro: {save_error}")

            # Estatísticas finais
            total = len(segment_paths)
            stats = {
                'total': total,
                'approved': len(approved_segments),
                'intermediate': len(intermediate_segments),
                'rejected': len(rejected_segments),
                'approval_rate': len(approved_segments) / total if total > 0 else 0.0,
                'rejection_rate': len(rejected_segments) / total if total > 0 else 0.0
            }

            logger.info("[SUMMARY] Filtragem MOS 3-tier completa:")
            logger.info(f"  [OK] Aprovados (>=3.0): {stats['approved']}")
            logger.info(f"  [INFO] Intermediários (2.5-3.0): {stats['intermediate']}")
            logger.info(f"  [REJECT] Rejeitados (<2.5): {stats['rejected']}")

            return {
                'approved': approved_segments,
                'intermediate': intermediate_segments,
                'rejected': rejected_segments,
                'stats': stats
            }

    def _save_segment(self,
                      segment_path: Path,
                      target_dir: Path,
                      mos_score: float,
                      category: str,
                      index: int,
                      has_naming_utils: bool) -> None:
        """
        Salva segmento categorizado.

        Args:
            segment_path: Path do segmento original
            target_dir: Diretório destino
            mos_score: Score MOS
            category: Categoria (approved/intermediate/rejected)
            index: Índice do segmento
            has_naming_utils: Se naming_utils está disponível
        """
        try:
            mos_str = f"{mos_score:.1f}".replace('.', ',')

            if has_naming_utils:
                from ..naming_utils import extract_base_name, generate_standard_name
                base_name = extract_base_name(segment_path)
                standard_name = generate_standard_name(base_name, f"mos_{category}", index)
                filename = f"{standard_name}_{mos_str}.flac"
            else:
                # Nome simples sem naming_utils
                filename = f"{segment_path.stem}_mos_{mos_str}_{category}.flac"

            target_path = target_dir / filename
            shutil.copy2(segment_path, target_path)

            logger.debug(f"[SAVE] Segmento salvo: {filename}")

        except Exception as e:
            logger.warning(f"[AVISO] Não foi possível salvar segmento {segment_path.name}: {e}")

    def unload_resources(self) -> None:
        """Descarrega modelo SHEET da memória."""
        if self.predictor is not None:
            logger.info("[CLEANUP] Descarregando modelo MOS...")
            self.resource_manager.unload_model('mos_sheet')
            self.predictor = None
            logger.info("[OK] Modelo MOS descarregado")

    def get_quality_report(self, segment_paths: List[Path]) -> Dict[str, Any]:
        """
        Gera relatório de qualidade para segmentos.

        Args:
            segment_paths: Lista de paths de segmentos

        Returns:
            Dict com estatísticas de qualidade
        """
        scores = []

        logger.info(f"[INFO] Gerando relatório de qualidade para {len(segment_paths)} segmentos")

        for segment_path in segment_paths:
            try:
                score = self.predict_mos_score(segment_path)
                scores.append(score)
            except:
                scores.append(1.0)

        if not scores:
            return {
                'total_segments': 0,
                'average_mos': 0.0,
                'min_mos': 0.0,
                'max_mos': 0.0,
                'accepted_count': 0,
                'rejected_count': 0,
                'acceptance_rate': 0.0
            }

        scores = np.array(scores)
        accepted_count = np.sum(scores >= self.mos_threshold)

        report = {
            'total_segments': len(scores),
            'average_mos': float(np.mean(scores)),
            'min_mos': float(np.min(scores)),
            'max_mos': float(np.max(scores)),
            'accepted_count': int(accepted_count),
            'rejected_count': int(len(scores) - accepted_count),
            'acceptance_rate': float(accepted_count / len(scores))
        }

        logger.info(f"[REPORT] Média MOS: {report['average_mos']:.2f}")
        logger.info(f"[REPORT] Taxa de aceitação: {report['acceptance_rate']*100:.1f}%")

        return report
