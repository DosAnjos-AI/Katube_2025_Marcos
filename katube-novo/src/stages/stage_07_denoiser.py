"""
Stage 07: Audio Denoiser
Denoising de áudio usando DeepFilterNet
CPU-only, lazy loading (~1GB modelo)
"""

import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..core.base_processor import BaseProcessor
from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class Stage07Denoiser(BaseProcessor):
    """
    Etapa 07: Denoising de áudio usando DeepFilterNet.

    Input: Lista de Paths para arquivos de áudio
    Output: Dict com paths de áudios processados

    Modelo: DeepFilterNet3 (~1GB)
    Sample Rate: 48kHz (fixo)
    """

    def __init__(self,
                 resource_manager: ResourceManager,
                 model_name: str = "DeepFilterNet3",
                 df_sample_rate: int = 48000):
        """
        Inicializa denoiser.

        Args:
            resource_manager: Gerenciador de recursos
            model_name: Nome do modelo DeepFilterNet (padrão: DeepFilterNet3)
            df_sample_rate: Sample rate do DeepFilterNet (padrão: 48000)
        """
        super().__init__(resource_manager)
        self.model_name = model_name
        self.df_sample_rate = df_sample_rate
        self.device = 'cpu'  # Forçado para CPU

        # Lazy loaded
        self.df_state = None
        self.df_model = None

        logger.info(f"[CONFIG] Denoiser: model={model_name}, sr={df_sample_rate}, device=cpu")

    def load_resources(self) -> None:
        """Carrega modelo DeepFilterNet sob demanda (lazy loading)."""
        if self.df_model is None:
            logger.info(f"[LOAD] Carregando DeepFilterNet: {self.model_name}")

            try:
                # Import DeepFilterNet
                from df.enhance import init_df

                # Carrega modelo via ResourceManager
                def _loader():
                    state, model, loaded_model_name = init_df(self.model_name)
                    logger.info(f"[OK] DeepFilterNet carregado: {loaded_model_name}")
                    return (state, model, loaded_model_name)

                result = self.resource_manager.load_model(
                    model_key=f'deepfilternet_{self.model_name}',
                    loader_func=_loader
                )

                self.df_state, self.df_model, loaded_name = result
                logger.info("[OK] DeepFilterNet carregado em CPU")

            except ImportError as e:
                logger.error(f"[ERRO] DeepFilterNet não instalado: {e}")
                logger.error("[INFO] Instale: pip install deepfilternet")
                raise RuntimeError(f"DeepFilterNet é necessário para denoising: {e}")
            except Exception as e:
                logger.error(f"[ERRO] Falha ao carregar DeepFilterNet: {e}")
                raise

    @torch.no_grad()
    def denoise_audio(self,
                      input_path: Path,
                      output_path: Path) -> Dict[str, Any]:
        """
        Aplica denoising a um arquivo de áudio.

        Args:
            input_path: Path do arquivo de entrada
            output_path: Path do arquivo de saída

        Returns:
            Dict com resultado do processamento
        """
        try:
            if self.df_model is None:
                raise RuntimeError("[ERRO] DeepFilterNet não carregado")

            from df.enhance import load_audio, enhance, save_audio

            # Carrega áudio (DeepFilterNet sempre usa 48kHz)
            audio, _ = load_audio(str(input_path), sr=self.df_sample_rate)

            # Aplica enhancement (denoising)
            enhanced = enhance(self.df_state, self.df_model, audio)

            # Salva áudio processado
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_audio(str(output_path), enhanced, self.df_sample_rate)

            logger.debug(f"[OK] Denoised: {input_path.name} → {output_path.name}")

            return {
                'success': True,
                'input_path': str(input_path),
                'output_path': str(output_path),
                'sample_rate': self.df_sample_rate
            }

        except Exception as e:
            logger.error(f"[ERRO] Falha ao processar {input_path.name}: {e}")
            return {
                'success': False,
                'input_path': str(input_path),
                'error': str(e)
            }

    @torch.no_grad()
    def process(self,
                audio_paths: List[Path],
                output_dir: Path,
                suffix: str = "_denoised") -> Dict[str, Any]:
        """
        Processa denoising em lote.

        Args:
            audio_paths: Lista de paths de áudio
            output_dir: Diretório de saída
            suffix: Sufixo para arquivos de saída (padrão: "_denoised")

        Returns:
            Dict com resultados de processamento
        """
        with self.managed_processing():
            output_dir.mkdir(parents=True, exist_ok=True)

            total_files = len(audio_paths)
            logger.info(f"[INFO] Processando {total_files} arquivos para denoising")

            results = {
                'successful': [],
                'failed': [],
                'total_processed': total_files,
                'success_count': 0,
                'failure_count': 0
            }

            for i, input_path in enumerate(audio_paths, 1):
                try:
                    # Gera output path
                    output_filename = f"{input_path.stem}{suffix}.flac"
                    output_path = output_dir / output_filename

                    logger.info(f"[PROCESS] {i}/{total_files}: {input_path.name}")

                    # Aplica denoising
                    result = self.denoise_audio(input_path, output_path)

                    if result['success']:
                        results['successful'].append(result)
                        results['success_count'] += 1
                        logger.info(f"[OK] Denoised: {output_filename}")
                    else:
                        results['failed'].append(result)
                        results['failure_count'] += 1
                        logger.error(f"[ERRO] Falhou: {input_path.name}")

                    # Progress
                    if i % 10 == 0:
                        logger.info(f"[PROGRESS] Processados {i}/{total_files} arquivos")

                except Exception as e:
                    logger.error(f"[ERRO] Exceção ao processar {input_path.name}: {e}")
                    results['failed'].append({
                        'success': False,
                        'input_path': str(input_path),
                        'error': str(e)
                    })
                    results['failure_count'] += 1

            logger.info("[SUMMARY] Denoising concluído:")
            logger.info(f"  [OK] Sucesso: {results['success_count']}")
            logger.info(f"  [ERRO] Falhas: {results['failure_count']}")
            logger.info(f"  [INFO] Diretório de saída: {output_dir}")

            return results

    def unload_resources(self) -> None:
        """Descarrega modelo DeepFilterNet da memória."""
        if self.df_model is not None:
            logger.info("[CLEANUP] Descarregando DeepFilterNet...")
            self.resource_manager.unload_model(f'deepfilternet_{self.model_name}')
            self.df_state = None
            self.df_model = None
            logger.info("[OK] DeepFilterNet descarregado")
