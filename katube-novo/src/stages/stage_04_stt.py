"""
Stage 04: Speech-to-Text Unified
Transcrição STT usando Whisper e WAV2VEC2 SEQUENCIALMENTE
CPU-only, lazy loading sequencial para economizar RAM (~6GB total)

CRÍTICO: Modelos NUNCA em memória simultaneamente!
"""

import torch
import librosa
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from ..core.base_processor import BaseProcessor
from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class Stage04STT(BaseProcessor):
    """
    Etapa 04: Transcrição STT unificada com carregamento sequencial.

    Input: Lista de Paths para segmentos de áudio
    Output: Dict com transcrições de Whisper e WAV2VEC2

    Estratégia de Memória:
    1. Carregar Whisper → processar tudo → descarregar
    2. Carregar WAV2VEC2 → processar tudo → descarregar
    3. NUNCA ter ambos modelos em RAM juntos

    Economia: ~3GB RAM (vs carregar ambos simultaneamente)
    """

    def __init__(self,
                 resource_manager: ResourceManager,
                 whisper_model_name: str = "freds0/distil-whisper-large-v3-ptbr",
                 wav2vec2_model_name: str = "alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-gain-normalization",
                 huggingface_token: Optional[str] = None,
                 target_sample_rate: int = 16000):
        """
        Inicializa transcriber STT unificado.

        Args:
            resource_manager: Gerenciador de recursos
            whisper_model_name: Nome do modelo Whisper (HuggingFace)
            wav2vec2_model_name: Nome do modelo WAV2VEC2 (HuggingFace)
            huggingface_token: Token HuggingFace (para modelos gated)
            target_sample_rate: Taxa de amostragem alvo (padrão: 16000)
        """
        super().__init__(resource_manager)
        self.whisper_model_name = whisper_model_name
        self.wav2vec2_model_name = wav2vec2_model_name
        self.huggingface_token = huggingface_token
        self.target_sample_rate = target_sample_rate
        self.device = 'cpu'  # Forçado para CPU

        # Modelos lazy loaded sequencialmente
        self.whisper_processor = None
        self.whisper_model = None
        self.wav2vec2_processor = None
        self.wav2vec2_model = None

        logger.info(f"[CONFIG] STT Unificado: whisper={whisper_model_name[:50]}...")
        logger.info(f"[CONFIG] STT Unificado: wav2vec2={wav2vec2_model_name[:50]}...")
        logger.info("[CONFIG] Estratégia: Carregamento sequencial (economia de RAM)")

    def load_resources(self) -> None:
        """
        NÃO carrega modelos aqui.

        Carregamento sequencial é feito no process() para economizar RAM.
        Este método existe apenas para compatibilidade com BaseProcessor.
        """
        logger.info("[INFO] STT: Modelos serão carregados sequencialmente durante process()")
        pass

    def load_whisper(self) -> None:
        """
        Carrega APENAS Whisper (~3GB).

        REGRA: Descarregar WAV2VEC2 antes de chamar este método!
        """
        if self.whisper_model is not None:
            logger.info("[CACHE] Whisper já carregado")
            return

        logger.info("[LOAD] Carregando Whisper modelo...")
        logger.info(f"[INFO] Modelo: {self.whisper_model_name}")

        try:
            # Carrega processor (leve, não usa ResourceManager)
            self.whisper_processor = WhisperProcessor.from_pretrained(
                self.whisper_model_name,
                token=self.huggingface_token
            )

            # Carrega modelo (pesado, usa ResourceManager)
            def _loader():
                model = WhisperForConditionalGeneration.from_pretrained(
                    self.whisper_model_name,
                    torch_dtype=torch.float32,  # CPU usa float32
                    token=self.huggingface_token
                )
                model.to(self.device)
                return model

            self.whisper_model = self.resource_manager.load_model(
                model_key='whisper_stt_model',
                loader_func=_loader
            )

            logger.info("[OK] Whisper carregado em CPU")

        except Exception as e:
            logger.error(f"[ERRO] Falha ao carregar Whisper: {e}")
            raise RuntimeError(f"Falha ao carregar Whisper STT: {e}")

    def unload_whisper(self) -> None:
        """
        Descarrega Whisper e libera memória.

        CRÍTICO: Chamado antes de carregar WAV2VEC2!
        """
        if self.whisper_model is not None:
            logger.info("[CLEANUP] Descarregando Whisper...")
            self.resource_manager.unload_model('whisper_stt_model')
            self.whisper_model = None
            self.whisper_processor = None
            logger.info("[OK] Whisper descarregado, memória liberada")

    def load_wav2vec2(self) -> None:
        """
        Carrega APENAS WAV2VEC2 (~3GB).

        REGRA: Descarregar Whisper antes de chamar este método!
        """
        if self.wav2vec2_model is not None:
            logger.info("[CACHE] WAV2VEC2 já carregado")
            return

        logger.info("[LOAD] Carregando WAV2VEC2 modelo...")
        logger.info(f"[INFO] Modelo: {self.wav2vec2_model_name}")
        logger.info("[INFO] Especializado para Português Brasileiro")

        try:
            # Carrega processor (leve, não usa ResourceManager)
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
                self.wav2vec2_model_name
            )

            # Carrega modelo (pesado, usa ResourceManager)
            def _loader():
                model = Wav2Vec2ForCTC.from_pretrained(
                    self.wav2vec2_model_name
                )
                model.to(self.device)
                return model

            self.wav2vec2_model = self.resource_manager.load_model(
                model_key='wav2vec2_stt_model',
                loader_func=_loader
            )

            logger.info("[OK] WAV2VEC2 carregado em CPU")

        except Exception as e:
            logger.error(f"[ERRO] Falha ao carregar WAV2VEC2: {e}")
            raise RuntimeError(f"Falha ao carregar WAV2VEC2 STT: {e}")

    def unload_wav2vec2(self) -> None:
        """
        Descarrega WAV2VEC2 e libera memória.
        """
        if self.wav2vec2_model is not None:
            logger.info("[CLEANUP] Descarregando WAV2VEC2...")
            self.resource_manager.unload_model('wav2vec2_stt_model')
            self.wav2vec2_model = None
            self.wav2vec2_processor = None
            logger.info("[OK] WAV2VEC2 descarregado, memória liberada")

    def _preprocess_audio(self, audio_path: Path) -> np.ndarray:
        """
        Pré-processa áudio para STT.

        Args:
            audio_path: Path para arquivo de áudio

        Returns:
            Array numpy com áudio preprocessado
        """
        try:
            # Carrega áudio
            audio, sr = librosa.load(audio_path, sr=self.target_sample_rate)

            # Converte para mono se estéreo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Normaliza
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            return audio

        except Exception as e:
            logger.error(f"[ERRO] Falha ao pré-processar {audio_path}: {e}")
            raise

    @torch.no_grad()
    def _transcribe_whisper(self, audio_path: Path) -> str:
        """
        Transcreve áudio com Whisper.

        Args:
            audio_path: Path para arquivo de áudio

        Returns:
            Transcrição em texto
        """
        try:
            if self.whisper_model is None:
                raise RuntimeError("[ERRO] Whisper não está carregado")

            # Pré-processa áudio
            audio = self._preprocess_audio(audio_path)

            # Processa com Whisper processor
            input_features = self.whisper_processor(
                audio,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt"
            ).input_features.to(self.device)

            # Gera transcrição
            predicted_ids = self.whisper_model.generate(input_features)
            transcription = self.whisper_processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            logger.debug(f"[WHISPER] {audio_path.name}: {transcription[:50]}...")
            return transcription.strip()

        except Exception as e:
            logger.error(f"[ERRO] Whisper falhou para {audio_path}: {e}")
            return ""

    @torch.no_grad()
    def _transcribe_wav2vec2(self, audio_path: Path) -> str:
        """
        Transcreve áudio com WAV2VEC2.

        Args:
            audio_path: Path para arquivo de áudio

        Returns:
            Transcrição em texto
        """
        try:
            if self.wav2vec2_model is None:
                raise RuntimeError("[ERRO] WAV2VEC2 não está carregado")

            # Pré-processa áudio
            audio = self._preprocess_audio(audio_path)

            # Processa com WAV2VEC2 processor
            input_values = self.wav2vec2_processor(
                audio,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt"
            ).input_values.to(self.device)

            # Gera transcrição
            logits = self.wav2vec2_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.wav2vec2_processor.batch_decode(predicted_ids)[0]

            logger.debug(f"[WAV2VEC2] {audio_path.name}: {transcription[:50]}...")
            return transcription.strip()

        except Exception as e:
            logger.error(f"[ERRO] WAV2VEC2 falhou para {audio_path}: {e}")
            return ""

    @torch.no_grad()
    def process(self,
                segment_paths: List[Path],
                output_dir: Optional[Path] = None,
                save_txt: bool = True) -> Dict[str, Any]:
        """
        Processa transcrição STT com ambos modelos SEQUENCIALMENTE.

        Fluxo:
        1. Carregar Whisper → transcrever todos → descarregar
        2. Carregar WAV2VEC2 → transcrever todos → descarregar
        3. Salvar resultados se output_dir fornecido

        Args:
            segment_paths: Lista de paths de áudio
            output_dir: Diretório de saída (opcional)
            save_txt: Se deve salvar arquivos .txt com transcrições

        Returns:
            Dict com resultados de ambos modelos
        """
        total_segments = len(segment_paths)
        logger.info(f"[INFO] Processando {total_segments} segmentos com STT sequencial")

        # Prepara diretórios de saída
        whisper_dir = None
        wav2vec2_dir = None
        if output_dir and save_txt:
            stt_dir = output_dir / 'stt_results'
            whisper_dir = stt_dir / 'STT-whisper'
            wav2vec2_dir = stt_dir / 'STT-wav2vec2'
            whisper_dir.mkdir(parents=True, exist_ok=True)
            wav2vec2_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'whisper_transcriptions': {},
            'wav2vec2_transcriptions': {},
            'whisper_results': [],
            'wav2vec2_results': []
        }

        # FASE 1: Whisper
        logger.info("[PHASE-1] Iniciando transcrição com Whisper")
        self.load_whisper()

        for i, segment_path in enumerate(segment_paths, 1):
            logger.info(f"[WHISPER] Segmento {i}/{total_segments}: {segment_path.name}")

            try:
                transcription = self._transcribe_whisper(segment_path)
                results['whisper_transcriptions'][str(segment_path)] = transcription

                # Salva arquivo .txt se solicitado
                if whisper_dir and save_txt:
                    # Importa naming utils se disponível
                    try:
                        from ..naming_utils import extract_base_name, generate_standard_name
                        base_name = extract_base_name(segment_path)
                        standard_name = generate_standard_name(base_name, "stt_whisper", i)
                        txt_filename = f"{standard_name}.txt"
                    except ImportError:
                        txt_filename = f"{segment_path.stem}_whisper.txt"

                    txt_path = whisper_dir / txt_filename
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(transcription)

                    results['whisper_results'].append({
                        'segment': segment_path.name,
                        'transcription': transcription,
                        'file': str(txt_path)
                    })

                logger.info(f"[OK] Whisper concluído para {segment_path.name}")

            except Exception as e:
                logger.error(f"[ERRO] Whisper falhou para {segment_path.name}: {e}")
                results['whisper_transcriptions'][str(segment_path)] = ""

        self.unload_whisper()
        logger.info(f"[PHASE-1] Whisper concluído: {len(results['whisper_transcriptions'])} transcrições")

        # FASE 2: WAV2VEC2
        logger.info("[PHASE-2] Iniciando transcrição com WAV2VEC2")
        self.load_wav2vec2()

        for i, segment_path in enumerate(segment_paths, 1):
            logger.info(f"[WAV2VEC2] Segmento {i}/{total_segments}: {segment_path.name}")

            try:
                transcription = self._transcribe_wav2vec2(segment_path)
                results['wav2vec2_transcriptions'][str(segment_path)] = transcription

                # Salva arquivo .txt se solicitado
                if wav2vec2_dir and save_txt:
                    # Importa naming utils se disponível
                    try:
                        from ..naming_utils import extract_base_name, generate_standard_name
                        base_name = extract_base_name(segment_path)
                        standard_name = generate_standard_name(base_name, "stt_wav2vec2", i)
                        txt_filename = f"{standard_name}.txt"
                    except ImportError:
                        txt_filename = f"{segment_path.stem}_wav2vec2.txt"

                    txt_path = wav2vec2_dir / txt_filename
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(transcription)

                    results['wav2vec2_results'].append({
                        'segment': segment_path.name,
                        'transcription': transcription,
                        'file': str(txt_path)
                    })

                logger.info(f"[OK] WAV2VEC2 concluído para {segment_path.name}")

            except Exception as e:
                logger.error(f"[ERRO] WAV2VEC2 falhou para {segment_path.name}: {e}")
                results['wav2vec2_transcriptions'][str(segment_path)] = ""

        self.unload_wav2vec2()
        logger.info(f"[PHASE-2] WAV2VEC2 concluído: {len(results['wav2vec2_transcriptions'])} transcrições")

        # Estatísticas finais
        results['stats'] = {
            'total_segments': total_segments,
            'whisper_count': len(results['whisper_results']),
            'wav2vec2_count': len(results['wav2vec2_results']),
            'whisper_dir': str(whisper_dir) if whisper_dir else None,
            'wav2vec2_dir': str(wav2vec2_dir) if wav2vec2_dir else None
        }

        logger.info("[SUMMARY] STT sequencial concluído:")
        logger.info(f"  [OK] Whisper: {results['stats']['whisper_count']} transcrições")
        logger.info(f"  [OK] WAV2VEC2: {results['stats']['wav2vec2_count']} transcrições")

        return results

    def unload_resources(self) -> None:
        """
        Garante que todos os modelos estão descarregados.
        """
        logger.info("[CLEANUP] Descarregando todos os modelos STT...")
        self.unload_whisper()
        self.unload_wav2vec2()
        logger.info("[OK] Todos os modelos STT descarregados")
