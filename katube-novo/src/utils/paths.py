"""
Path Manager for Katube
Centralized path handling and validation
"""

import os
from pathlib import Path
from typing import Optional, Union


class PathManager:
    """
    Gerenciador centralizado de paths.

    Fornece métodos utilitários para:
    - Validação de paths
    - Criação de diretórios
    - Normalização de caminhos
    - Resolução de paths relativos
    """

    @staticmethod
    def ensure_path(path: Union[str, Path], create: bool = True) -> Path:
        """
        Garante que path é Path object e existe.

        Args:
            path: Path como string ou Path
            create: Criar diretório se não existir (padrão: True)

        Returns:
            Path object

        Raises:
            ValueError: Se path for None ou vazio
        """
        if path is None:
            raise ValueError("Path cannot be None")

        path = Path(path)

        if create and not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path

    @staticmethod
    def ensure_parent(file_path: Union[str, Path]) -> Path:
        """
        Garante que diretório pai do arquivo existe.

        Args:
            file_path: Path do arquivo

        Returns:
            Path object do arquivo
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path

    @staticmethod
    def validate_file_exists(file_path: Union[str, Path]) -> Path:
        """
        Valida que arquivo existe.

        Args:
            file_path: Path do arquivo

        Returns:
            Path object

        Raises:
            FileNotFoundError: Se arquivo não existir
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path não é um arquivo: {file_path}")

        return file_path

    @staticmethod
    def validate_dir_exists(dir_path: Union[str, Path]) -> Path:
        """
        Valida que diretório existe.

        Args:
            dir_path: Path do diretório

        Returns:
            Path object

        Raises:
            FileNotFoundError: Se diretório não existir
        """
        dir_path = Path(dir_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {dir_path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path não é um diretório: {dir_path}")

        return dir_path

    @staticmethod
    def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> Path:
        """
        Obtém path relativo.

        Args:
            path: Path absoluto ou relativo
            base: Path base para relativização

        Returns:
            Path relativo
        """
        path = Path(path).resolve()
        base = Path(base).resolve()

        try:
            return path.relative_to(base)
        except ValueError:
            # Se não for possível relativizar, retorna path absoluto
            return path

    @staticmethod
    def normalize_path(path: Union[str, Path]) -> Path:
        """
        Normaliza path (resolve symlinks, ~, etc).

        Args:
            path: Path para normalizar

        Returns:
            Path normalizado
        """
        return Path(path).expanduser().resolve()

    @staticmethod
    def find_files(
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False
    ) -> list[Path]:
        """
        Encontra arquivos em diretório.

        Args:
            directory: Diretório para busca
            pattern: Pattern glob (padrão: "*")
            recursive: Busca recursiva (padrão: False)

        Returns:
            Lista de Paths encontrados
        """
        directory = Path(directory)

        if not directory.exists():
            return []

        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))
