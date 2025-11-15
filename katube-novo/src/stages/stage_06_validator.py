"""
Stage 06: Transcription Validator
Validação de transcrições usando Levenshtein distance
CPU-only, sem modelos (apenas processamento de texto)
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import textdistance

from ..core.base_processor import BaseProcessor
from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class Stage06Validator(BaseProcessor):
    """
    Etapa 06: Validação de transcrições usando similaridade Levenshtein.

    Input: Pares de transcrições (Whisper + WAV2VEC2)
    Output: Dict com scores de similaridade e aprovação/rejeição

    Validação baseada em:
    - Levenshtein distance normalizada
    - Threshold configurável
    - Classificação: aprovado/rejeitado
    """

    def __init__(self,
                 resource_manager: ResourceManager,
                 similarity_threshold: float = 0.7,
                 algorithm: str = "levenshtein"):
        """
        Inicializa validador de transcrições.

        Args:
            resource_manager: Gerenciador de recursos
            similarity_threshold: Threshold de similaridade (0.0-1.0, padrão: 0.7)
            algorithm: Algoritmo de similaridade (padrão: "levenshtein")
        """
        super().__init__(resource_manager)
        self.similarity_threshold = similarity_threshold
        self.algorithm = algorithm

        logger.info(f"[CONFIG] Validator: threshold={similarity_threshold}, algorithm={algorithm}")
        logger.info("[CONFIG] Sem modelos (processamento CPU puro)")

    def load_resources(self) -> None:
        """Não há recursos para carregar (processamento de texto puro)."""
        logger.debug("[INFO] Validator: sem recursos para carregar")
        pass

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similaridade entre dois textos.

        Args:
            text1: Primeiro texto (ex: Whisper)
            text2: Segundo texto (ex: WAV2VEC2)

        Returns:
            Score de similaridade (0.0-1.0)
        """
        if not text1 and not text2:
            return 1.0  # Ambos vazios = 100% similar

        if not text1 or not text2:
            return 0.0  # Um vazio = 0% similar

        try:
            if self.algorithm == "levenshtein":
                # Usa Levenshtein normalizado
                similarity = textdistance.levenshtein.normalized_similarity(text1, text2)
            elif self.algorithm == "jaro_winkler":
                similarity = textdistance.jaro_winkler(text1, text2)
            elif self.algorithm == "cosine":
                similarity = textdistance.cosine(text1, text2)
            else:
                # Fallback para Levenshtein
                logger.warning(f"[AVISO] Algoritmo '{self.algorithm}' desconhecido, usando Levenshtein")
                similarity = textdistance.levenshtein.normalized_similarity(text1, text2)

            return float(similarity)

        except Exception as e:
            logger.error(f"[ERRO] Falha ao calcular similaridade: {e}")
            return 0.0

    def validate_pair(self,
                      text1: str,
                      text2: str,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Valida um par de transcrições.

        Args:
            text1: Primeira transcrição (ex: Whisper)
            text2: Segunda transcrição (ex: WAV2VEC2)
            metadata: Metadados opcionais (ex: filename, segment_id)

        Returns:
            Dict com resultado de validação
        """
        similarity = self.calculate_similarity(text1, text2)
        approved = similarity >= self.similarity_threshold

        result = {
            'text1': text1,
            'text2': text2,
            'similarity': similarity,
            'threshold': self.similarity_threshold,
            'approved': approved,
            'status': 'approved' if approved else 'rejected'
        }

        if metadata:
            result['metadata'] = metadata

        return result

    def process(self,
                transcription_pairs: List[Tuple[str, str]],
                metadata_list: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Processa validação de múltiplos pares de transcrições.

        Args:
            transcription_pairs: Lista de tuplas (text1, text2)
            metadata_list: Lista opcional de metadados para cada par

        Returns:
            Dict com resultados de validação e estatísticas
        """
        with self.managed_processing():
            total_pairs = len(transcription_pairs)
            logger.info(f"[INFO] Validando {total_pairs} pares de transcrições")
            logger.info(f"[INFO] Threshold de similaridade: {self.similarity_threshold}")

            results = []
            approved_count = 0
            rejected_count = 0
            similarities = []

            for i, (text1, text2) in enumerate(transcription_pairs):
                try:
                    # Metadados opcionais
                    metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None

                    # Valida par
                    result = self.validate_pair(text1, text2, metadata)
                    results.append(result)

                    # Estatísticas
                    similarities.append(result['similarity'])
                    if result['approved']:
                        approved_count += 1
                    else:
                        rejected_count += 1

                    # Log detalhado para rejeições
                    if not result['approved']:
                        logger.warning(f"[REJECT] Par {i+1}: similaridade={result['similarity']:.3f} < {self.similarity_threshold}")
                        logger.debug(f"  Text1: {text1[:50]}...")
                        logger.debug(f"  Text2: {text2[:50]}...")

                    # Progress
                    if (i + 1) % 50 == 0:
                        logger.info(f"[PROGRESS] Validados {i + 1}/{total_pairs} pares")

                except Exception as e:
                    logger.error(f"[ERRO] Falha ao validar par {i}: {e}")
                    results.append({
                        'text1': text1 if 'text1' in locals() else "",
                        'text2': text2 if 'text2' in locals() else "",
                        'similarity': 0.0,
                        'threshold': self.similarity_threshold,
                        'approved': False,
                        'status': 'error',
                        'error': str(e)
                    })
                    rejected_count += 1

            # Estatísticas finais
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            approval_rate = approved_count / total_pairs if total_pairs > 0 else 0.0

            stats = {
                'total_pairs': total_pairs,
                'approved_count': approved_count,
                'rejected_count': rejected_count,
                'approval_rate': approval_rate,
                'avg_similarity': avg_similarity,
                'min_similarity': min(similarities) if similarities else 0.0,
                'max_similarity': max(similarities) if similarities else 0.0,
                'threshold': self.similarity_threshold
            }

            logger.info("[SUMMARY] Validação concluída:")
            logger.info(f"  [OK] Total de pares: {stats['total_pairs']}")
            logger.info(f"  [OK] Aprovados: {stats['approved_count']} ({stats['approval_rate']*100:.1f}%)")
            logger.info(f"  [REJECT] Rejeitados: {stats['rejected_count']}")
            logger.info(f"  [INFO] Similaridade média: {stats['avg_similarity']:.3f}")
            logger.info(f"  [INFO] Similaridade min/max: {stats['min_similarity']:.3f} / {stats['max_similarity']:.3f}")

            return {
                'results': results,
                'stats': stats,
                'approved': [r for r in results if r['approved']],
                'rejected': [r for r in results if not r['approved']]
            }

    def unload_resources(self) -> None:
        """Não há recursos para descarregar (processamento de texto puro)."""
        logger.debug("[INFO] Validator: sem recursos para descarregar")
        pass
