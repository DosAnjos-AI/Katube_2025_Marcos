"""
Katube Orchestrator - CPU-First Pipeline
Orquestrador modular do pipeline completo de processamento TTS

Substitui pipeline.py monolítico por arquitetura modular com 9 stages.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime

from .core.resource_manager import ResourceManager
from .config import Config
from .utils.logging_config import get_logger
from .utils.paths import PathManager

# Import all stages
from .stages.stage_02_mos_filter import Stage02MOSFilter
from .stages.stage_03_diarizer import Stage03Diarizer
from .stages.stage_04_stt import Stage04STT
from .stages.stage_05_normalizer import Stage05TextNormalizer
from .stages.stage_06_validator import Stage06Validator
from .stages.stage_07_denoiser import Stage07Denoiser
from .stages.stage_08_sox import Stage08SoxNormalizer
from .stages.stage_09_dataset import Stage09DatasetGenerator

logger = get_logger(__name__)


class KatubeOrchestrator:
    """
    Orquestrador principal do pipeline Katube 2025 - CPU-First.

    Executa pipeline modular com 9 etapas sequenciais e gestão inteligente de recursos:
    1. Segmentação de áudio (TODO: criar Stage01Segmenter)
    2. Filtro de qualidade MOS
    3. Diarização de speakers
    4. Transcrição STT (Whisper + WAV2VEC2)
    5. Normalização de textos
    6. Validação Levenshtein
    7. Denoising de áudio
    8. Normalização final com Sox
    9. Geração de dataset CSV

    Características:
    - CPU-first architecture
    - Lazy loading de modelos
    - Resource management centralizado
    - Error handling robusto
    - Logs padronizados sem emojis
    """

    def __init__(self,
                 output_base_dir: Optional[Path] = None,
                 huggingface_token: Optional[str] = None,
                 max_cache_mb: int = 4096,
                 enable_mos_filter: bool = True,
                 enable_denoiser: bool = False,
                 **stage_configs):
        """
        Inicializa orquestrador do pipeline.

        Args:
            output_base_dir: Diretório base para outputs
            huggingface_token: Token HuggingFace para modelos gated
            max_cache_mb: Limite de cache em MB (padrão: 4096)
            enable_mos_filter: Habilitar filtro MOS (padrão: True)
            enable_denoiser: Habilitar denoiser (padrão: False)
            **stage_configs: Configurações específicas por stage
        """
        logger.info("[INIT] Inicializando Katube Orchestrator...")

        self.output_base_dir = PathManager.ensure_path(
            output_base_dir or Config.OUTPUT_DIR
        )
        self.huggingface_token = huggingface_token or Config.HUGGINGFACE_TOKEN
        self.enable_mos_filter = enable_mos_filter
        self.enable_denoiser = enable_denoiser

        # Resource manager central - compartilhado por todas as stages
        self.resource_manager = ResourceManager()

        # Inicializar stages (sem carregar modelos ainda - lazy loading)
        self._init_stages(**stage_configs)

        logger.info("[OK] Orchestrator inicializado")
        logger.info(f"[INFO] Output dir: {self.output_base_dir}")
        logger.info(f"[INFO] MOS Filter: {enable_mos_filter}")
        logger.info(f"[INFO] Denoiser: {enable_denoiser}")

    def _init_stages(self, **configs):
        """
        Inicializa todas as stages com configurações.

        Stages são inicializadas mas modelos NÃO são carregados ainda (lazy loading).
        """
        logger.info("[INIT] Inicializando stages...")

        # Stage 02: MOS Quality Filter
        self.stage_02 = Stage02MOSFilter(
            resource_manager=self.resource_manager,
            mos_threshold=configs.get('mos_threshold', 2.5),
            sample_rate=configs.get('sample_rate', 24000)
        )
        logger.debug("[INIT] Stage 02: MOS Filter configurado")

        # Stage 03: Speaker Diarization
        self.stage_03 = Stage03Diarizer(
            resource_manager=self.resource_manager,
            huggingface_token=self.huggingface_token,
            model_name=configs.get('diarization_model', Config.PYANNOTE_MODEL),
            sample_rate=configs.get('sample_rate', 24000)
        )
        logger.debug("[INIT] Stage 03: Diarizer configurado")

        # Stage 04: STT Unified
        self.stage_04 = Stage04STT(
            resource_manager=self.resource_manager,
            whisper_model_name=configs.get('whisper_model', "freds0/distil-whisper-large-v3-ptbr"),
            wav2vec2_model_name=configs.get('wav2vec2_model', "alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-gain-normalization"),
            huggingface_token=self.huggingface_token
        )
        logger.debug("[INIT] Stage 04: STT configurado")

        # Stage 05: Text Normalizer
        self.stage_05 = Stage05TextNormalizer(
            resource_manager=self.resource_manager,
            remove_accents=configs.get('remove_accents', True),
            remove_punctuation=configs.get('remove_punctuation', True),
            lowercase=configs.get('lowercase', True)
        )
        logger.debug("[INIT] Stage 05: Text Normalizer configurado")

        # Stage 06: Validator
        self.stage_06 = Stage06Validator(
            resource_manager=self.resource_manager,
            similarity_threshold=configs.get('similarity_threshold', 0.75),
            algorithm=configs.get('similarity_algorithm', 'levenshtein')
        )
        logger.debug("[INIT] Stage 06: Validator configurado")

        # Stage 07: Denoiser
        if self.enable_denoiser:
            self.stage_07 = Stage07Denoiser(
                resource_manager=self.resource_manager,
                model_name=configs.get('denoiser_model', 'DeepFilterNet3')
            )
            logger.debug("[INIT] Stage 07: Denoiser configurado")
        else:
            self.stage_07 = None
            logger.debug("[INIT] Stage 07: Denoiser desabilitado")

        # Stage 08: Sox Normalizer
        self.stage_08 = Stage08SoxNormalizer(
            resource_manager=self.resource_manager,
            target_sample_rate=configs.get('target_sample_rate', 48000),
            target_format=configs.get('target_format', 'flac'),
            normalize_gain=configs.get('normalize_gain', True)
        )
        logger.debug("[INIT] Stage 08: Sox Normalizer configurado")

        # Stage 09: Dataset Generator
        self.stage_09 = Stage09DatasetGenerator(
            resource_manager=self.resource_manager,
            csv_delimiter=configs.get('csv_delimiter', ','),
            include_header=configs.get('include_header', True)
        )
        logger.debug("[INIT] Stage 09: Dataset Generator configurado")

        logger.info("[OK] Todas as stages configuradas")

    def process_audio_pipeline(self,
                                audio_paths: List[Path],
                                session_name: Optional[str] = None,
                                num_speakers: Optional[int] = None,
                                save_intermediates: bool = True) -> Dict[str, Any]:
        """
        Processa áudios através do pipeline completo (stages 02-09).

        NOTA: Stage 01 (Segmentação) ainda não implementada.
        Esta versão assume que audio_paths já são segmentos processados.

        Args:
            audio_paths: Lista de Paths para segmentos de áudio
            session_name: Nome da sessão de processamento
            num_speakers: Dica de número de speakers (opcional)
            save_intermediates: Salvar resultados intermediários

        Returns:
            Dict com resultados completos e metadados
        """
        start_time = time.time()

        logger.info("[INFO] ========================================")
        logger.info("[INFO] INICIANDO PIPELINE KATUBE CPU-FIRST")
        logger.info("[INFO] ========================================")
        logger.info(f"[INFO] Segmentos de entrada: {len(audio_paths)}")
        logger.info(f"[INFO] Sessão: {session_name or 'default'}")

        # Setup session directory
        session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = self.output_base_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'session_name': session_name,
            'session_dir': str(session_dir),
            'start_time': datetime.now().isoformat(),
            'stages': {},
            'success': False
        }

        try:
            # STAGE 02: MOS Quality Filter
            if self.enable_mos_filter:
                logger.info("[INFO] ----- STAGE 02: MOS Quality Filter -----")
                mos_output_dir = session_dir / '02_mos_filter' if save_intermediates else None

                mos_result = self.stage_02.process(
                    segment_paths=audio_paths,
                    output_dir=mos_output_dir
                )

                approved_segments = mos_result['approved']
                results['stages']['02_mos_filter'] = {
                    'total_input': len(audio_paths),
                    'approved': len(approved_segments),
                    'intermediate': len(mos_result.get('intermediate', [])),
                    'rejected': len(mos_result.get('rejected', [])),
                    'stats': mos_result.get('stats', {})
                }

                logger.info(f"[OK] Stage 02: {len(approved_segments)}/{len(audio_paths)} segmentos aprovados")

                # Usa apenas segmentos aprovados para próximas etapas
                audio_paths = approved_segments
            else:
                logger.info("[INFO] Stage 02: MOS Filter desabilitado")
                results['stages']['02_mos_filter'] = {'skipped': True}

            if not audio_paths:
                logger.error("[ERRO] Nenhum segmento aprovado pelo MOS filter")
                results['error'] = 'No approved segments after MOS filter'
                return results

            # STAGE 03: Speaker Diarization
            logger.info("[INFO] ----- STAGE 03: Speaker Diarization -----")
            diarization_output_dir = session_dir / '03_diarization' if save_intermediates else None

            diarization_result = self.stage_03.process(
                audio_paths=audio_paths,
                output_dir=diarization_output_dir,
                num_speakers=num_speakers,
                save_rttm=True
            )

            results['stages']['03_diarization'] = {
                'total_processed': len(audio_paths),
                'num_files': len(diarization_result)
            }

            logger.info(f"[OK] Stage 03: {len(diarization_result)} arquivos diarizados")

            # STAGE 04: STT (Whisper + WAV2VEC2) - Carregamento Sequencial
            logger.info("[INFO] ----- STAGE 04: STT (Whisper + WAV2VEC2) -----")
            stt_output_dir = session_dir / '04_stt' if save_intermediates else None

            stt_result = self.stage_04.process(
                segment_paths=audio_paths,
                output_dir=stt_output_dir,
                save_txt=True
            )

            results['stages']['04_stt'] = {
                'total_processed': len(audio_paths),
                'whisper_count': stt_result['stats']['whisper_count'],
                'wav2vec2_count': stt_result['stats']['wav2vec2_count']
            }

            logger.info(f"[OK] Stage 04: {stt_result['stats']['whisper_count']} transcrições Whisper")
            logger.info(f"[OK] Stage 04: {stt_result['stats']['wav2vec2_count']} transcrições WAV2VEC2")

            # STAGE 05: Text Normalization
            logger.info("[INFO] ----- STAGE 05: Text Normalization -----")

            # Coleta textos de Whisper e WAV2VEC2
            whisper_texts = list(stt_result['whisper_transcriptions'].values())
            wav2vec2_texts = list(stt_result['wav2vec2_transcriptions'].values())

            # Normaliza Whisper
            whisper_norm_result = self.stage_05.process(
                texts=whisper_texts,
                labels=['whisper'] * len(whisper_texts)
            )

            # Normaliza WAV2VEC2
            wav2vec2_norm_result = self.stage_05.process(
                texts=wav2vec2_texts,
                labels=['wav2vec2'] * len(wav2vec2_texts)
            )

            results['stages']['05_normalization'] = {
                'whisper_normalized': len(whisper_norm_result['normalized_texts']),
                'wav2vec2_normalized': len(wav2vec2_norm_result['normalized_texts'])
            }

            logger.info(f"[OK] Stage 05: Textos normalizados")

            # STAGE 06: Validation (Levenshtein)
            logger.info("[INFO] ----- STAGE 06: Validation (Levenshtein) -----")

            # Cria pares para validação (Whisper vs WAV2VEC2 normalizados)
            transcription_pairs = list(zip(
                whisper_norm_result['normalized_texts'],
                wav2vec2_norm_result['normalized_texts']
            ))

            validation_result = self.stage_06.process(
                transcription_pairs=transcription_pairs,
                metadata_list=[{'index': i, 'audio_path': str(audio_paths[i])}
                               for i in range(len(audio_paths))]
            )

            # Filtra segmentos aprovados
            approved_indices = [i for i, r in enumerate(validation_result['results']) if r['approved']]
            validated_audio_paths = [audio_paths[i] for i in approved_indices]

            results['stages']['06_validation'] = {
                'total_pairs': validation_result['stats']['total_pairs'],
                'approved': validation_result['stats']['approved_count'],
                'rejected': validation_result['stats']['rejected_count'],
                'approval_rate': validation_result['stats']['approval_rate'],
                'avg_similarity': validation_result['stats']['avg_similarity']
            }

            logger.info(f"[OK] Stage 06: {validation_result['stats']['approved_count']}/{validation_result['stats']['total_pairs']} pares aprovados")
            logger.info(f"[INFO] Stage 06: Similaridade média: {validation_result['stats']['avg_similarity']:.3f}")

            if not validated_audio_paths:
                logger.error("[ERRO] Nenhum par aprovado pela validação")
                results['error'] = 'No approved pairs after validation'
                results['success'] = False
                return results

            # STAGE 07: Denoising (opcional)
            if self.enable_denoiser and self.stage_07:
                logger.info("[INFO] ----- STAGE 07: Audio Denoising -----")
                denoising_output_dir = session_dir / '07_denoised'

                denoising_result = self.stage_07.process(
                    audio_paths=validated_audio_paths,
                    output_dir=denoising_output_dir,
                    suffix="_denoised"
                )

                results['stages']['07_denoising'] = {
                    'total_processed': denoising_result['total_processed'],
                    'success_count': denoising_result['success_count'],
                    'failure_count': denoising_result['failure_count']
                }

                logger.info(f"[OK] Stage 07: {denoising_result['success_count']} arquivos processados")

                # Usa áudios denoised para próxima etapa
                validated_audio_paths = [Path(r['output_path']) for r in denoising_result['successful']]
            else:
                logger.info("[INFO] Stage 07: Denoiser desabilitado (pulado)")
                results['stages']['07_denoising'] = {'skipped': True}

            # STAGE 08: Sox Normalization
            logger.info("[INFO] ----- STAGE 08: Sox Final Normalization -----")
            sox_output_dir = session_dir / '08_sox_normalized'

            sox_result = self.stage_08.process(
                audio_paths=validated_audio_paths,
                output_dir=sox_output_dir,
                suffix="_final"
            )

            results['stages']['08_sox'] = {
                'total_processed': sox_result['total_processed'],
                'success_count': sox_result['success_count'],
                'failure_count': sox_result['failure_count']
            }

            logger.info(f"[OK] Stage 08: {sox_result['success_count']} arquivos normalizados")

            # STAGE 09: Dataset Generation
            logger.info("[INFO] ----- STAGE 09: Dataset Generation -----")

            # Prepara entradas do dataset
            dataset_entries = []
            for i in approved_indices:
                if i < len(sox_result['successful']):
                    sox_file = sox_result['successful'][i]
                    entry = self.stage_09.create_dataset_entry(
                        audio_path=Path(sox_file['output_path']),
                        transcription=whisper_texts[i],  # Usa texto original Whisper
                        speaker_id='unknown',  # TODO: extrair de diarization
                        duration=0.0,  # TODO: calcular duração
                        mos_score=0.0,  # TODO: pegar score real
                        similarity=validation_result['results'][i]['similarity'],
                        sample_rate=sox_file['sample_rate']
                    )
                    dataset_entries.append(entry)

            dataset_csv_path = session_dir / 'dataset.csv'
            dataset_result = self.stage_09.process(
                dataset_entries=dataset_entries,
                output_path=dataset_csv_path,
                validate_entries=True
            )

            results['stages']['09_dataset'] = {
                'csv_path': str(dataset_csv_path),
                'total_entries': dataset_result['total_entries'],
                'valid_count': dataset_result['valid_count'],
                'invalid_count': dataset_result['invalid_count']
            }

            logger.info(f"[OK] Stage 09: Dataset gerado: {dataset_csv_path}")
            logger.info(f"[INFO] Stage 09: {dataset_result['valid_count']} entradas válidas")

            # Finalização
            end_time = time.time()
            elapsed = end_time - start_time

            results['success'] = True
            results['end_time'] = datetime.now().isoformat()
            results['elapsed_seconds'] = elapsed
            results['final_dataset_path'] = str(dataset_csv_path)
            results['final_dataset_entries'] = dataset_result['valid_count']

            logger.info("[INFO] ========================================")
            logger.info("[OK] PIPELINE CONCLUÍDO COM SUCESSO")
            logger.info(f"[INFO] Tempo total: {elapsed:.1f}s ({elapsed/60:.1f}min)")
            logger.info(f"[INFO] Dataset final: {dataset_result['valid_count']} entradas")
            logger.info(f"[INFO] Salvo em: {dataset_csv_path}")
            logger.info("[INFO] ========================================")

        except Exception as e:
            logger.error(f"[ERRO] Pipeline falhou: {e}")
            logger.exception("[ERRO] Stack trace:")
            results['success'] = False
            results['error'] = str(e)
            raise

        finally:
            # Limpeza de recursos
            logger.info("[CLEANUP] Liberando recursos...")
            self.resource_manager.unload_all()
            logger.info("[OK] Recursos liberados")

        return results

    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de uso de recursos.

        Returns:
            Dict com informações de modelos carregados e memória
        """
        return {
            'loaded_models': self.resource_manager.get_loaded_models(),
            'memory_stats': self.resource_manager.get_memory_stats()
        }

    def cleanup(self):
        """Cleanup explícito de recursos."""
        logger.info("[CLEANUP] Limpeza manual solicitada")
        self.resource_manager.unload_all()
        logger.info("[OK] Cleanup concluído")
