"""
Wrapper de compatibilidade para stt_whisper.py legado.
Redireciona para nova arquitetura modular Stage04STT.

DEPRECATED: Use Stage04STT diretamente.
Este arquivo existe apenas para manter compatibilidade com código legado.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import da nova arquitetura
from .stages.stage_04_stt import Stage04STT
from .core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class WhisperSTTTranscriber:
    """
    Wrapper de compatibilidade para WhisperSTTTranscriber legado.

    DEPRECATED: Este é um wrapper para manter compatibilidade.
    Use Stage04STT diretamente para novos desenvolvimentos.

    NOTA: Este wrapper carrega/descarrega Whisper para cada chamada,
    o que é menos eficiente que usar Stage04STT diretamente.
    """

    def __init__(self,
                 whisper_model_name: str = "freds0/distil-whisper-large-v3-ptbr",
                 device: str = "cpu",
                 huggingface_token: Optional[str] = None):
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
            whisper_model_name=whisper_model_name,
            huggingface_token=huggingface_token
        )

        self.device = 'cpu'
        self.whisper_model_name = whisper_model_name
        self.huggingface_token = huggingface_token

        # Propriedades de compatibilidade
        self.whisper_processor = None  # Será setado ao carregar
        self.whisper_model = None  # Será setado ao carregar

        logger.info("[COMPAT] WhisperSTTTranscriber wrapper inicializado")
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
        Transcreve áudio usando Whisper.

        Args:
            audio_path: Path para arquivo de áudio

        Returns:
            Transcrição em texto
        """
        # Carrega Whisper se necessário
        if self._stage.whisper_model is None:
            self._stage.load_whisper()
            # Expõe propriedades para compatibilidade
            self.whisper_processor = self._stage.whisper_processor
            self.whisper_model = self._stage.whisper_model

        return self._stage._transcribe_whisper(audio_path)

    def transcribe_segments(self,
                            segment_paths: List[Path],
                            output_dir: Path) -> Dict[str, Any]:
        """
        Transcreve múltiplos segmentos usando Whisper.

        Args:
            segment_paths: Lista de paths de áudio
            output_dir: Diretório de saída

        Returns:
            Dict com resultados de transcrição
        """
        logger.info(f"[COMPAT] Transcrevendo {len(segment_paths)} segmentos com Whisper")

        # Carrega Whisper
        self._stage.load_whisper()
        self.whisper_processor = self._stage.whisper_processor
        self.whisper_model = self._stage.whisper_model

        # Prepara diretório de saída
        stt_dir = output_dir / 'stt_results'
        whisper_dir = stt_dir / 'STT-whisper'
        whisper_dir.mkdir(parents=True, exist_ok=True)

        whisper_results = []

        for i, segment_path in enumerate(segment_paths, 1):
            logger.info(f"[WHISPER] Processando segmento {i}/{len(segment_paths)}: {segment_path.name}")

            try:
                # Transcreve
                transcription = self._stage._transcribe_whisper(segment_path)

                # Salva arquivo .txt
                try:
                    from .naming_utils import extract_base_name, generate_standard_name
                    base_name = extract_base_name(segment_path)
                    standard_name = generate_standard_name(base_name, "stt_whisper", i)
                    txt_filename = f"{standard_name}.txt"
                except ImportError:
                    txt_filename = f"{segment_path.stem}_whisper.txt"

                whisper_file = whisper_dir / txt_filename
                with open(whisper_file, 'w', encoding='utf-8') as f:
                    f.write(transcription)

                whisper_results.append({
                    "segment": segment_path.name,
                    "transcription": transcription,
                    "file": str(whisper_file)
                })

                logger.info(f"[OK] Transcrito {segment_path.name}")

            except Exception as e:
                logger.error(f"[ERRO] Falha ao processar {segment_path.name}: {e}")

        # Descarrega Whisper
        self._stage.unload_whisper()
        self.whisper_processor = None
        self.whisper_model = None

        logger.info(f"[SUMMARY] Whisper concluído: {len(whisper_results)} transcrições")

        return {
            "whisper_results": whisper_results,
            "whisper_dir": str(whisper_dir),
            "total_segments": len(segment_paths),
            "whisper_count": len(whisper_results)
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
            self._stage.unload_whisper()
        except:
            pass
