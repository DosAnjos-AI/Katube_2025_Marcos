"""
Stage 09: Dataset Generator
Geração de dataset.csv final consolidando todas as etapas
CPU-only, sem modelos (processamento de dados)
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.base_processor import BaseProcessor
from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class Stage09DatasetGenerator(BaseProcessor):
    """
    Etapa 09: Geração de dataset.csv final.

    Input: Dados consolidados de todas as etapas anteriores
    Output: Arquivo CSV com dataset TTS

    Campos do CSV:
    - audio_path: Path para arquivo de áudio final
    - transcription: Transcrição validada
    - speaker_id: ID do speaker
    - duration: Duração em segundos
    - mos_score: Score MOS de qualidade
    - similarity: Similaridade Whisper/WAV2VEC2
    - sample_rate: Taxa de amostragem
    - ... outros campos configuráveis
    """

    def __init__(self,
                 resource_manager: ResourceManager,
                 csv_delimiter: str = ',',
                 include_header: bool = True):
        """
        Inicializa gerador de dataset.

        Args:
            resource_manager: Gerenciador de recursos
            csv_delimiter: Delimitador CSV (padrão: ',')
            include_header: Incluir header no CSV (padrão: True)
        """
        super().__init__(resource_manager)
        self.csv_delimiter = csv_delimiter
        self.include_header = include_header

        logger.info(f"[CONFIG] Dataset Generator: delimiter='{csv_delimiter}', header={include_header}")

    def load_resources(self) -> None:
        """Não há recursos para carregar (processamento de dados puro)."""
        logger.debug("[INFO] Dataset Generator: sem recursos para carregar")
        pass

    def validate_entry(self, entry: Dict[str, Any]) -> bool:
        """
        Valida entrada do dataset.

        Args:
            entry: Dicionário com dados da entrada

        Returns:
            True se válido, False caso contrário
        """
        # Campos obrigatórios
        required_fields = ['audio_path', 'transcription']

        for field in required_fields:
            if field not in entry or not entry[field]:
                logger.warning(f"[AVISO] Entrada inválida: campo '{field}' ausente ou vazio")
                return False

        # Valida existência do arquivo de áudio
        audio_path = Path(entry['audio_path'])
        if not audio_path.exists():
            logger.warning(f"[AVISO] Arquivo de áudio não existe: {audio_path}")
            return False

        return True

    def create_dataset_entry(self,
                             audio_path: Path,
                             transcription: str,
                             **metadata) -> Dict[str, Any]:
        """
        Cria entrada de dataset com campos padrão.

        Args:
            audio_path: Path para arquivo de áudio
            transcription: Transcrição do áudio
            **metadata: Metadados adicionais

        Returns:
            Dict com entrada de dataset
        """
        entry = {
            'audio_path': str(audio_path),
            'transcription': transcription,
            'speaker_id': metadata.get('speaker_id', 'unknown'),
            'duration': metadata.get('duration', 0.0),
            'mos_score': metadata.get('mos_score', 0.0),
            'similarity': metadata.get('similarity', 0.0),
            'sample_rate': metadata.get('sample_rate', 48000),
            'channels': metadata.get('channels', 1),
            'file_size_mb': metadata.get('file_size_mb', 0.0),
            'created_at': metadata.get('created_at', datetime.now().isoformat()),
            'processing_status': metadata.get('processing_status', 'completed')
        }

        # Adiciona metadados extras
        for key, value in metadata.items():
            if key not in entry:
                entry[key] = value

        return entry

    def process(self,
                dataset_entries: List[Dict[str, Any]],
                output_path: Path,
                validate_entries: bool = True) -> Dict[str, Any]:
        """
        Processa geração de dataset CSV.

        Args:
            dataset_entries: Lista de entradas do dataset
            output_path: Path para arquivo CSV de saída
            validate_entries: Validar entradas antes de salvar (padrão: True)

        Returns:
            Dict com estatísticas de geração
        """
        with self.managed_processing():
            total_entries = len(dataset_entries)
            logger.info(f"[INFO] Gerando dataset com {total_entries} entradas")
            logger.info(f"[INFO] Output: {output_path}")

            valid_entries = []
            invalid_entries = []

            # Valida entradas se solicitado
            if validate_entries:
                logger.info("[VALIDATE] Validando entradas do dataset...")
                for i, entry in enumerate(dataset_entries):
                    if self.validate_entry(entry):
                        valid_entries.append(entry)
                    else:
                        invalid_entries.append(entry)
                        logger.warning(f"[INVALID] Entrada {i} inválida")

                logger.info(f"[VALIDATE] Válidas: {len(valid_entries)}, Inválidas: {len(invalid_entries)}")
            else:
                valid_entries = dataset_entries

            if not valid_entries:
                logger.error("[ERRO] Nenhuma entrada válida para salvar")
                return {
                    'success': False,
                    'error': 'Nenhuma entrada válida',
                    'total_entries': total_entries,
                    'valid_count': 0,
                    'invalid_count': len(invalid_entries)
                }

            # Cria diretório de saída
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Salva CSV
            try:
                logger.info(f"[SAVE] Salvando dataset em: {output_path}")

                # Determina campos (usa primeira entrada válida)
                fieldnames = list(valid_entries[0].keys())

                with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile,
                                            fieldnames=fieldnames,
                                            delimiter=self.csv_delimiter)

                    if self.include_header:
                        writer.writeheader()

                    writer.writerows(valid_entries)

                logger.info(f"[OK] Dataset salvo com sucesso: {len(valid_entries)} entradas")

                # Estatísticas
                stats = {
                    'success': True,
                    'output_path': str(output_path),
                    'total_entries': total_entries,
                    'valid_count': len(valid_entries),
                    'invalid_count': len(invalid_entries),
                    'fieldnames': fieldnames,
                    'file_size_bytes': output_path.stat().st_size,
                    'file_size_mb': output_path.stat().st_size / (1024 * 1024)
                }

                logger.info("[SUMMARY] Dataset gerado:")
                logger.info(f"  [OK] Arquivo: {output_path}")
                logger.info(f"  [OK] Entradas válidas: {stats['valid_count']}")
                logger.info(f"  [INFO] Entradas inválidas: {stats['invalid_count']}")
                logger.info(f"  [INFO] Campos: {len(fieldnames)}")
                logger.info(f"  [INFO] Tamanho: {stats['file_size_mb']:.2f} MB")

                return stats

            except Exception as e:
                error_msg = f"Falha ao salvar CSV: {str(e)}"
                logger.error(f"[ERRO] {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'total_entries': total_entries,
                    'valid_count': len(valid_entries),
                    'invalid_count': len(invalid_entries)
                }

    def merge_datasets(self,
                       dataset_paths: List[Path],
                       output_path: Path) -> Dict[str, Any]:
        """
        Mescla múltiplos datasets CSV em um único.

        Args:
            dataset_paths: Lista de paths para CSVs existentes
            output_path: Path para CSV mesclado

        Returns:
            Dict com estatísticas de mesclagem
        """
        logger.info(f"[MERGE] Mesclando {len(dataset_paths)} datasets")

        all_entries = []
        fieldnames_set = set()

        for dataset_path in dataset_paths:
            try:
                with open(dataset_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile, delimiter=self.csv_delimiter)
                    entries = list(reader)
                    all_entries.extend(entries)

                    # Coleta fieldnames
                    if reader.fieldnames:
                        fieldnames_set.update(reader.fieldnames)

                    logger.info(f"[MERGE] Carregado: {dataset_path.name} ({len(entries)} entradas)")

            except Exception as e:
                logger.error(f"[ERRO] Falha ao ler {dataset_path}: {e}")

        # Processa dataset mesclado
        return self.process(all_entries, output_path, validate_entries=False)

    def unload_resources(self) -> None:
        """Não há recursos para descarregar (processamento de dados puro)."""
        logger.debug("[INFO] Dataset Generator: sem recursos para descarregar")
        pass
