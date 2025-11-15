"""
Stage 08: Sox Audio Normalizer
Normalização final de áudio usando Sox
CPU-only, sem modelos (subprocess para Sox CLI)
"""

import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

from ..core.base_processor import BaseProcessor
from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class Stage08SoxNormalizer(BaseProcessor):
    """
    Etapa 08: Normalização final de áudio usando Sox.

    Input: Lista de Paths para arquivos de áudio
    Output: Dict com paths de áudios normalizados

    Operações:
    - Conversão de sample rate
    - Conversão para mono
    - Normalização de ganho
    - Conversão de formato
    """

    def __init__(self,
                 resource_manager: ResourceManager,
                 target_sample_rate: int = 48000,
                 target_format: str = "flac",
                 target_channels: int = 1,
                 normalize_gain: bool = True):
        """
        Inicializa normalizador Sox.

        Args:
            resource_manager: Gerenciador de recursos
            target_sample_rate: Sample rate alvo (padrão: 48000)
            target_format: Formato de saída (padrão: flac)
            target_channels: Número de canais (padrão: 1=mono)
            normalize_gain: Normalizar ganho (padrão: True)
        """
        super().__init__(resource_manager)
        self.target_sample_rate = target_sample_rate
        self.target_format = target_format
        self.target_channels = target_channels
        self.normalize_gain = normalize_gain
        self.sox_executable = None

        logger.info(f"[CONFIG] Sox Normalizer: sr={target_sample_rate}, format={target_format}, "
                    f"channels={target_channels}, norm_gain={normalize_gain}")

    def load_resources(self) -> None:
        """Verifica disponibilidade do Sox."""
        if self.sox_executable is None:
            logger.info("[CHECK] Verificando instalação do Sox...")
            self.sox_executable = self._find_sox()
            logger.info(f"[OK] Sox encontrado: {self.sox_executable}")

    def _find_sox(self) -> str:
        """
        Encontra executável Sox no sistema.

        Returns:
            Path para executável Sox

        Raises:
            RuntimeError: Se Sox não for encontrado
        """
        # Caminhos possíveis para Sox
        sox_paths = [
            'sox',  # PATH
            r'C:\Program Files\Chris Bagwell\SoX\sox.exe',
            r'C:\Program Files (x86)\Chris Bagwell\SoX\sox.exe',
            # WinGet
            os.path.join(os.getenv('LOCALAPPDATA', ''), 'Microsoft', 'WinGet', 'Packages',
                         'ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe',
                         'sox-14.4.2', 'sox.exe'),
            os.path.expanduser(r'~\AppData\Local\Microsoft\WinGet\Packages\ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe\sox-14.4.2\sox.exe'),
            # Linux common paths
            '/usr/bin/sox',
            '/usr/local/bin/sox'
        ]

        for sox_path in sox_paths:
            try:
                result = subprocess.run(
                    [sox_path, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"[FOUND] Sox em: {sox_path}")
                    return sox_path
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                continue

        # Sox não encontrado
        error_msg = "Sox não encontrado. Instale Sox: https://sox.sourceforge.net/"
        logger.error(f"[ERRO] {error_msg}")
        raise RuntimeError(error_msg)

    def normalize_audio(self,
                        input_path: Path,
                        output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Normaliza um arquivo de áudio usando Sox.

        Args:
            input_path: Path do arquivo de entrada
            output_path: Path do arquivo de saída (opcional)

        Returns:
            Dict com resultado da normalização
        """
        try:
            if self.sox_executable is None:
                raise RuntimeError("[ERRO] Sox não inicializado")

            # Gera output path se não fornecido
            if output_path is None:
                output_path = input_path.parent / f"{input_path.stem}_sox.{self.target_format}"

            logger.debug(f"[PROCESS] Normalizando: {input_path.name}")

            # Constrói comando Sox
            cmd = [
                self.sox_executable,
                str(input_path),
                '-r', str(self.target_sample_rate),  # Sample rate
                '-c', str(self.target_channels),     # Channels
            ]

            # Adiciona normalização de ganho
            if self.normalize_gain:
                cmd.append('--norm')

            # Output path
            cmd.append(str(output_path))

            # Executa Sox
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos timeout
            )

            if result.returncode == 0:
                # Sucesso
                input_size = input_path.stat().st_size / (1024 * 1024)  # MB
                output_size = output_path.stat().st_size / (1024 * 1024)  # MB

                return {
                    'success': True,
                    'input_path': str(input_path),
                    'output_path': str(output_path),
                    'input_size_mb': input_size,
                    'output_size_mb': output_size,
                    'sample_rate': self.target_sample_rate,
                    'channels': self.target_channels,
                    'format': self.target_format
                }
            else:
                # Erro
                error_msg = result.stderr.strip() if result.stderr else "Erro desconhecido"
                logger.error(f"[ERRO] Sox falhou: {error_msg}")
                return {
                    'success': False,
                    'input_path': str(input_path),
                    'error': error_msg
                }

        except subprocess.TimeoutExpired:
            error_msg = "Timeout de normalização Sox (5 minutos)"
            logger.error(f"[ERRO] {error_msg}")
            return {
                'success': False,
                'input_path': str(input_path),
                'error': error_msg
            }
        except Exception as e:
            error_msg = f"Erro ao normalizar: {str(e)}"
            logger.error(f"[ERRO] {error_msg}")
            return {
                'success': False,
                'input_path': str(input_path),
                'error': error_msg
            }

    def process(self,
                audio_paths: List[Path],
                output_dir: Path,
                suffix: str = "_sox") -> Dict[str, Any]:
        """
        Processa normalização Sox em lote.

        Args:
            audio_paths: Lista de paths de áudio
            output_dir: Diretório de saída
            suffix: Sufixo para arquivos de saída (padrão: "_sox")

        Returns:
            Dict com resultados de processamento
        """
        with self.managed_processing():
            output_dir.mkdir(parents=True, exist_ok=True)

            total_files = len(audio_paths)
            logger.info(f"[INFO] Normalizando {total_files} arquivos com Sox")

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
                    output_filename = f"{input_path.stem}{suffix}.{self.target_format}"
                    output_path = output_dir / output_filename

                    logger.info(f"[PROCESS] {i}/{total_files}: {input_path.name}")

                    # Normaliza
                    result = self.normalize_audio(input_path, output_path)

                    if result['success']:
                        results['successful'].append(result)
                        results['success_count'] += 1
                        logger.info(f"[OK] Normalizado: {output_filename}")
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

            logger.info("[SUMMARY] Normalização Sox concluída:")
            logger.info(f"  [OK] Sucesso: {results['success_count']}")
            logger.info(f"  [ERRO] Falhas: {results['failure_count']}")
            logger.info(f"  [INFO] Diretório de saída: {output_dir}")

            return results

    def unload_resources(self) -> None:
        """Não há recursos para descarregar (Sox é subprocess)."""
        logger.debug("[INFO] Sox Normalizer: sem recursos para descarregar")
        pass
