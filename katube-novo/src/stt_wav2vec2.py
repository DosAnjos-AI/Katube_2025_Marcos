"""
Wrapper de compatibilidade para stt_wav2vec2.py legado.
Redireciona para nova arquitetura modular Stage04STT.

DEPRECATED: Use Stage04STT diretamente.
Este arquivo existe apenas para manter compatibilidade com código legado.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

# Import da nova arquitetura
from .stages.stage_04_stt import Stage04STT
from .core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class WAV2VEC2STTTranscriber:
    """
    Wrapper de compatibilidade para WAV2VEC2STTTranscriber legado.

    DEPRECATED: Este é um wrapper para manter compatibilidade.
    Use Stage04STT diretamente para novos desenvolvimentos.

    NOTA: Este wrapper carrega/descarrega WAV2VEC2 para cada chamada,
    o que é menos eficiente que usar Stage04STT diretamente.
    """

    def __init__(self,
                 wav2vec2_model_name: str = "alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-gain-normalization",
                 device: str = "cpu"):
        """
        Inicializa wrapper de compatibilidade.

        NOTA: Parâmetro 'device' é ignorado - nova arquitetura força CPU.
        """
        if device != "cpu":
            logger.warning("[AVISO] device='cuda' ignorado - nova arquitetura força CPU")

        # Cria ResourceManager interno
        self._resource_manager = ResourceManager()

        # Cria Stage04STT (sem carregar modelos ainda)
        self._stage = Stage04STT(
            resource_manager=self._resource_manager,
            wav2vec2_model_name=wav2vec2_model_name
        )

        self.device = 'cpu'
        self.wav2vec2_model_name = wav2vec2_model_name

        # Propriedades de compatibilidade
        self.wav2vec2_processor = None  # Será setado ao carregar
        self.wav2vec2_model = None  # Será setado ao carregar

        logger.info("[COMPAT] WAV2VEC2STTTranscriber wrapper inicializado")
        logger.warning("[RECOMENDACAO] Use Stage04STT diretamente para melhor eficiência")

    def _preprocess_audio(self, audio_path: Path, target_sr: int = 16000):
        """
        Pré-processa áudio (compatibilidade).

        Args:
            audio_path: Path para arquivo de áudio
            target_sr: Taxa de amostragem alvo

        Returns:
            Array numpy com áudio preprocessado
        """
        return self._stage._preprocess_audio(audio_path)

    def transcribe_audio(self, audio_path: Path) -> str:
        """
        Transcreve áudio usando WAV2VEC2.

        Args:
            audio_path: Path para arquivo de áudio

        Returns:
            Transcrição em texto
        """
        # Carrega WAV2VEC2 se necessário
        if self._stage.wav2vec2_model is None:
            self._stage.load_wav2vec2()
            # Expõe propriedades para compatibilidade
            self.wav2vec2_processor = self._stage.wav2vec2_processor
            self.wav2vec2_model = self._stage.wav2vec2_model

        return self._stage._transcribe_wav2vec2(audio_path)

    def transcribe_segments(self,
                            segment_paths: List[Path],
                            output_dir: Path) -> Dict[str, Any]:
        """
        Transcreve múltiplos segmentos usando WAV2VEC2.

        Args:
            segment_paths: Lista de paths de áudio
            output_dir: Diretório de saída

        Returns:
            Dict com resultados de transcrição
        """
        logger.info(f"[COMPAT] Transcrevendo {len(segment_paths)} segmentos com WAV2VEC2")

        # Carrega WAV2VEC2
        self._stage.load_wav2vec2()
        self.wav2vec2_processor = self._stage.wav2vec2_processor
        self.wav2vec2_model = self._stage.wav2vec2_model

        # Prepara diretório de saída
        stt_dir = output_dir / 'stt_results'
        wav2vec2_dir = stt_dir / 'STT-wav2vec2'
        wav2vec2_dir.mkdir(parents=True, exist_ok=True)

        wav2vec2_results = []

        for i, segment_path in enumerate(segment_paths, 1):
            logger.info(f"[WAV2VEC2] Processando segmento {i}/{len(segment_paths)}: {segment_path.name}")

            try:
                # Transcreve
                transcription = self._stage._transcribe_wav2vec2(segment_path)

                # Salva arquivo .txt
                try:
                    from .naming_utils import extract_base_name, generate_standard_name
                    base_name = extract_base_name(segment_path)
                    standard_name = generate_standard_name(base_name, "stt_wav2vec2", i)
                    txt_filename = f"{standard_name}.txt"
                except ImportError:
                    txt_filename = f"{segment_path.stem}_wav2vec2.txt"

                wav2vec2_file = wav2vec2_dir / txt_filename
                with open(wav2vec2_file, 'w', encoding='utf-8') as f:
                    f.write(transcription)

                wav2vec2_results.append({
                    "segment": segment_path.name,
                    "transcription": transcription,
                    "file": str(wav2vec2_file)
                })

                logger.info(f"[OK] Transcrito {segment_path.name}")

            except Exception as e:
                logger.error(f"[ERRO] Falha ao processar {segment_path.name}: {e}")

        # Descarrega WAV2VEC2
        self._stage.unload_wav2vec2()
        self.wav2vec2_processor = None
        self.wav2vec2_model = None

        logger.info(f"[SUMMARY] WAV2VEC2 concluído: {len(wav2vec2_results)} transcrições")

        return {
            "wav2vec2_results": wav2vec2_results,
            "wav2vec2_dir": str(wav2vec2_dir),
            "total_segments": len(segment_paths),
            "wav2vec2_count": len(wav2vec2_results)
        }

    def _load_models(self):
        """
        Método legado - agora é no-op pois recursos são carregados sob demanda.

        Mantido para compatibilidade.
        """
        pass

    def __del__(self):
        """Cleanup ao destruir objeto."""
        try:
            self._stage.unload_wav2vec2()
        except:
            pass
