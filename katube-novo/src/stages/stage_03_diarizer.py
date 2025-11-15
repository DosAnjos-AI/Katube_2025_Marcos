"""
Stage 03: Speaker Diarization
Diarização de speakers usando pyannote.audio pipeline
CPU-only, lazy loading (~2GB modelo)
"""

import torch
import torchaudio
import pandas as pd
import numpy as np
import soundfile as sf
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

from ..core.base_processor import BaseProcessor
from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class Stage03Diarizer(BaseProcessor):
    """
    Etapa 03: Diarização de speakers usando pyannote.audio.

    Input: Path(s) para arquivo(s) de áudio
    Output: Dict com resultados de diarização (speakers, timestamps, estatísticas)

    Features:
    - Detecção automática de número de speakers
    - Pós-processamento (merge, limpeza)
    - Análise de estatísticas por speaker
    - Detecção de overlaps
    - Exportação RTTM
    """

    def __init__(self,
                 resource_manager: ResourceManager,
                 huggingface_token: Optional[str] = None,
                 model_name: str = "pyannote/speaker-diarization-3.1",
                 sample_rate: int = 24000,
                 min_segment_duration: float = 0.5):
        """
        Inicializa diarizador.

        Args:
            resource_manager: Gerenciador de recursos
            huggingface_token: Token HuggingFace para modelos gated
            model_name: Nome do modelo pyannote (padrão: speaker-diarization-3.1)
            sample_rate: Taxa de amostragem (padrão: 24000)
            min_segment_duration: Duração mínima de segmento (padrão: 0.5s)
        """
        super().__init__(resource_manager)
        self.huggingface_token = huggingface_token
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.min_segment_duration = min_segment_duration
        self.pipeline = None  # Lazy loaded
        self.device = 'cpu'  # Forçado para CPU

        logger.info(f"[CONFIG] Diarizer: model={model_name}, sr={sample_rate}, device=cpu")

    def load_resources(self) -> None:
        """Carrega pipeline pyannote sob demanda (lazy loading)."""
        if self.pipeline is None:
            logger.info(f"[LOAD] Carregando pipeline pyannote: {self.model_name}")

            self.pipeline = self.resource_manager.load_model(
                model_key=f'pyannote_diarization_{self.model_name}',
                loader_func=self._load_pyannote_pipeline
            )

            logger.info("[OK] Pipeline de diarização carregado em CPU")

    def _load_pyannote_pipeline(self):
        """
        Loader function para pyannote pipeline.

        Returns:
            Pipeline pyannote configurado para CPU
        """
        try:
            # Carrega pipeline do HuggingFace
            pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=self.huggingface_token
            )

            if pipeline is None:
                raise ValueError("Pipeline retornou None - verifique permissões de acesso ao modelo")

            # FORÇA CPU (não usar GPU)
            pipeline.to(torch.device('cpu'))

            logger.info(f"[OK] Pipeline {self.model_name} carregado em CPU")
            return pipeline

        except Exception as e:
            error_msg = f"[ERRO] Falha ao carregar pipeline de diarização: {e}"

            # Mensagem especial para modelos gated
            if "gated" in str(e).lower() or "unauthorized" in str(e).lower() or "401" in str(e):
                error_msg += "\n\n[INFO] SOLUCAO: Visite e aceite os termos:"
                error_msg += "\n  - https://hf.co/pyannote/speaker-diarization-3.1"
                error_msg += "\n  - https://hf.co/pyannote/segmentation-3.0"
                error_msg += "\n  - https://hf.co/pyannote/embedding"
                error_msg += "\nEm seguida, reinicie o servidor."

            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @torch.no_grad()
    def preprocess_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """
        Pré-processa áudio para diarização.

        Args:
            audio_path: Path para arquivo de áudio

        Returns:
            Tupla (waveform, sample_rate)
        """
        try:
            # Carrega áudio
            waveform, sample_rate = torchaudio.load(str(audio_path))

            # Converte para mono se estéreo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample se necessário
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
                sample_rate = self.sample_rate

            # Normaliza
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val

            return waveform, sample_rate

        except Exception as e:
            logger.error(f"[ERRO] Falha ao pré-processar áudio {audio_path}: {e}")
            raise

    @torch.no_grad()
    def diarize_audio(self,
                      audio_path: Path,
                      num_speakers: Optional[int] = None) -> Annotation:
        """
        Executa diarização em arquivo de áudio.

        Args:
            audio_path: Path para arquivo de áudio
            num_speakers: Número esperado de speakers (opcional)

        Returns:
            Annotation pyannote com resultados da diarização
        """
        if self.pipeline is None:
            raise RuntimeError("[ERRO] Pipeline de diarização não carregado")

        logger.info(f"[PROCESS] Diarizando: {audio_path.name}")

        # Pré-processa áudio
        waveform, sample_rate = self.preprocess_audio(audio_path)

        # Prepara input para pipeline
        audio_input = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        # Define número de speakers se fornecido
        if num_speakers is not None:
            try:
                # pyannote.audio 3.x: instantiate com número de clusters
                self.pipeline.instantiate({"clustering": {"num_clusters": num_speakers}})
            except Exception as e:
                logger.warning(f"[AVISO] Não foi possível definir num_speakers={num_speakers}: {e}")

        # Executa diarização
        try:
            diarization = self.pipeline(audio_input)
            num_detected = len(diarization.labels())
            logger.info(f"[OK] Diarização concluída: {num_detected} speakers detectados")
            return diarization

        except Exception as e:
            logger.error(f"[ERRO] Diarização falhou para {audio_path}: {e}")
            raise

    def annotation_to_dataframe(self,
                                 annotation: Annotation,
                                 audio_duration: Optional[float] = None) -> pd.DataFrame:
        """
        Converte Annotation pyannote para DataFrame pandas.

        Args:
            annotation: Annotation pyannote
            audio_duration: Duração total do áudio (opcional)

        Returns:
            DataFrame com colunas: START, END, DURATION, SPEAKER, CONFIDENCE
        """
        segments_data = []

        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segments_data.append({
                'START': segment.start,
                'END': segment.end,
                'DURATION': segment.duration,
                'SPEAKER': speaker,
                'CONFIDENCE': 1.0  # pyannote não fornece confidence nesta versão
            })

        df = pd.DataFrame(segments_data)

        if not df.empty:
            # Ordena por tempo de início
            df = df.sort_values('START').reset_index(drop=True)

            # Adiciona métricas relativas se duração fornecida
            if audio_duration is not None and audio_duration > 0:
                df['RELATIVE_START'] = df['START'] / audio_duration
                df['RELATIVE_END'] = df['END'] / audio_duration

        return df

    def post_process_annotation(self,
                                 annotation: Annotation,
                                 min_duration: Optional[float] = None) -> Annotation:
        """
        Pós-processa resultados de diarização.

        Args:
            annotation: Annotation original
            min_duration: Duração mínima de segmento (usa self.min_segment_duration se None)

        Returns:
            Annotation processada
        """
        if min_duration is None:
            min_duration = self.min_segment_duration

        # Remove segmentos muito curtos
        cleaned = annotation.support(min_duration)

        # Merge de segmentos próximos do mesmo speaker
        processed = Annotation()

        for speaker in cleaned.labels():
            speaker_timeline = cleaned.label_timeline(speaker)
            # Merge segmentos separados por menos de 0.5s
            merged_timeline = speaker_timeline.support(0.5)

            for segment in merged_timeline:
                processed[segment] = speaker

        return processed

    def analyze_speaker_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa estatísticas de speakers.

        Args:
            df: DataFrame com resultados de diarização

        Returns:
            Dict com estatísticas gerais e por speaker
        """
        if df.empty:
            return {'error': 'Sem dados de diarização disponíveis'}

        stats = {}

        # Estatísticas gerais
        total_duration = df['DURATION'].sum()
        stats['total_speech_duration'] = float(total_duration)
        stats['num_segments'] = int(len(df))
        stats['num_speakers'] = int(df['SPEAKER'].nunique())

        # Estatísticas por speaker
        speaker_stats = []
        for speaker in df['SPEAKER'].unique():
            speaker_df = df[df['SPEAKER'] == speaker]
            speaker_info = {
                'speaker': str(speaker),
                'total_duration': float(speaker_df['DURATION'].sum()),
                'num_segments': int(len(speaker_df)),
                'avg_segment_duration': float(speaker_df['DURATION'].mean()),
                'speaking_percentage': float((speaker_df['DURATION'].sum() / total_duration) * 100)
            }
            speaker_stats.append(speaker_info)

        # Ordena por tempo de fala
        speaker_stats = sorted(speaker_stats, key=lambda x: x['total_duration'], reverse=True)
        stats['speakers'] = speaker_stats

        # Análise de overlaps
        overlaps = self._detect_overlaps_in_annotation(df)
        stats['overlaps'] = overlaps

        return stats

    def _detect_overlaps_in_annotation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta overlaps na annotation.

        Args:
            df: DataFrame com diarização

        Returns:
            Dict com estatísticas de overlaps
        """
        if df.empty:
            return {'num_overlaps': 0, 'total_overlap_duration': 0.0}

        overlaps = []

        # Ordena por tempo de início
        df_sorted = df.sort_values('START')

        for i in range(len(df_sorted) - 1):
            current = df_sorted.iloc[i]
            next_segment = df_sorted.iloc[i + 1]

            # Verifica se segmentos se sobrepõem
            if (current['END'] > next_segment['START'] and
                    current['SPEAKER'] != next_segment['SPEAKER']):
                overlap_start = next_segment['START']
                overlap_end = min(current['END'], next_segment['END'])
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > 0:
                    overlaps.append({
                        'start': float(overlap_start),
                        'end': float(overlap_end),
                        'duration': float(overlap_duration),
                        'speakers': [str(current['SPEAKER']), str(next_segment['SPEAKER'])]
                    })

        total_overlap_duration = sum(o['duration'] for o in overlaps)

        return {
            'num_overlaps': len(overlaps),
            'total_overlap_duration': float(total_overlap_duration),
            'overlap_details': overlaps
        }

    def save_rttm(self,
                  annotation: Annotation,
                  output_path: Path,
                  audio_filename: str) -> None:
        """
        Salva resultados em formato RTTM.

        Args:
            annotation: Annotation pyannote
            output_path: Path de saída para arquivo RTTM
            audio_filename: Nome do arquivo de áudio original
        """
        try:
            with open(output_path, 'w') as f:
                annotation.write_rttm(f)
            logger.info(f"[SAVE] RTTM salvo: {output_path}")
        except Exception as e:
            logger.error(f"[ERRO] Falha ao salvar RTTM: {e}")
            raise

    def _get_audio_duration(self, audio_path: Path) -> float:
        """
        Obtém duração de áudio em segundos.

        Args:
            audio_path: Path para arquivo de áudio

        Returns:
            Duração em segundos
        """
        try:
            with sf.SoundFile(audio_path) as f:
                return len(f) / f.samplerate
        except:
            # Fallback usando torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            return waveform.shape[1] / sample_rate

    @torch.no_grad()
    def process(self,
                audio_paths: List[Path],
                output_dir: Optional[Path] = None,
                num_speakers: Optional[int] = None,
                save_rttm: bool = True) -> Dict[str, Any]:
        """
        Processa diarização em lote.

        Args:
            audio_paths: Lista de paths para arquivos de áudio
            output_dir: Diretório de saída (opcional)
            num_speakers: Número esperado de speakers (opcional)
            save_rttm: Se deve salvar arquivos RTTM

        Returns:
            Dict com resultados de processamento
        """
        with self.managed_processing():
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)

            results = {}
            total_files = len(audio_paths)

            logger.info(f"[INFO] Processando {total_files} arquivo(s) para diarização")

            for idx, audio_path in enumerate(audio_paths, 1):
                try:
                    logger.info(f"[PROGRESS] Arquivo {idx}/{total_files}: {audio_path.name}")

                    # Verifica se RTTM já existe
                    if output_dir and save_rttm:
                        rttm_path = output_dir / f"{audio_path.stem}.rttm"
                        if rttm_path.exists():
                            logger.info(f"[SKIP] RTTM já existe para {audio_path.name}")
                            continue

                    # Executa diarização
                    annotation = self.diarize_audio(audio_path, num_speakers)

                    # Converte para DataFrame
                    audio_duration = self._get_audio_duration(audio_path)
                    df = self.annotation_to_dataframe(annotation, audio_duration)

                    # Pós-processa
                    processed_annotation = self.post_process_annotation(annotation)

                    # Salva RTTM se solicitado
                    rttm_path_str = None
                    if output_dir and save_rttm:
                        rttm_path = output_dir / f"{audio_path.stem}.rttm"
                        self.save_rttm(processed_annotation, rttm_path, audio_path.name)
                        rttm_path_str = str(rttm_path)

                    # Analisa estatísticas
                    stats = self.analyze_speaker_statistics(df)

                    results[str(audio_path)] = {
                        'annotation': processed_annotation,
                        'dataframe': df,
                        'statistics': stats,
                        'rttm_path': rttm_path_str
                    }

                    logger.info(f"[OK] Diarização concluída: {audio_path.name}")

                except Exception as e:
                    logger.error(f"[ERRO] Falha ao processar {audio_path}: {e}")
                    results[str(audio_path)] = {'error': str(e)}

            logger.info(f"[SUMMARY] Diarização em lote concluída: {len(results)}/{total_files} arquivos processados")

            return results

    def unload_resources(self) -> None:
        """Descarrega pipeline pyannote da memória."""
        if self.pipeline is not None:
            logger.info("[CLEANUP] Descarregando pipeline de diarização...")
            self.resource_manager.unload_model(f'pyannote_diarization_{self.model_name}')
            self.pipeline = None
            logger.info("[OK] Pipeline de diarização descarregado")
