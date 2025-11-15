"""
Gerenciamento de caminhos cross-platform.
Garante compatibilidade Windows/Linux usando pathlib.
"""
from pathlib import Path
from typing import Union
import os


class PathManager:
    """
    Gerenciador de caminhos cross-platform usando pathlib.

    Usa exclusivamente pathlib.Path para compatibilidade Windows/Linux.
    Previne vulnerabilidades de Path Traversal.

    Exemplo:
        >>> pm = PathManager()
        >>> root = pm.get_project_root()
        >>> isinstance(root, Path)
        True
    """

    @staticmethod
    def ensure_path(path: Union[str, Path]) -> Path:
        """
        Converte para Path e cria diretorio se necessario.

        Args:
            path: String ou Path object

        Returns:
            Path object, com diretorio criado se nao existir

        Exemplo:
            >>> pm = PathManager()
            >>> import tempfile
            >>> test_dir = tempfile.mkdtemp()
            >>> p = pm.ensure_path(Path(test_dir) / "subdir" / "test")
            >>> p.parent.exists()
            True
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def get_project_root() -> Path:
        """
        Retorna raiz do projeto de forma confiavel.

        Assume que este arquivo esta em utils/ na raiz do projeto.

        Returns:
            Path para raiz do projeto (katube-novo/)

        Exemplo:
            >>> pm = PathManager()
            >>> root = pm.get_project_root()
            >>> root.name
            'katube-novo'
        """
        # Este arquivo esta em utils/paths.py, entao parent.parent e a raiz
        return Path(__file__).parent.parent

    @staticmethod
    def resolve_relative(relative_path: str) -> Path:
        """
        Resolve path relativo a partir da raiz do projeto.

        Args:
            relative_path: Path relativo (ex: "audios/data", "logs/app.log")

        Returns:
            Path absoluto resolvido

        Exemplo:
            >>> pm = PathManager()
            >>> p = pm.resolve_relative("audios/test")
            >>> p.is_absolute()
            True
        """
        return PathManager.get_project_root() / relative_path

    @staticmethod
    def safe_join(*parts: str) -> Path:
        """
        Join seguro de paths (previne Path Traversal).

        Remove componentes ".." maliciosos que podem ser usados
        para navegar fora do diretorio pretendido.

        Args:
            *parts: Componentes do path a serem unidos

        Returns:
            Path combinado de forma segura

        Exemplo:
            >>> pm = PathManager()
            >>> p = pm.safe_join("audios", "test", "file.wav")
            >>> str(p)
            'audios/test/file.wav'
            >>> p2 = pm.safe_join("audios", "../../../etc/passwd")
            >>> "../" in str(p2)
            False
        """
        # Remove ".." maliciosos de cada parte
        clean_parts = [p.replace("..", "").replace("./", "") for p in parts]
        # Remove partes vazias
        clean_parts = [p for p in clean_parts if p]
        return Path(*clean_parts) if clean_parts else Path(".")

    @staticmethod
    def get_file_size_mb(file_path: Union[str, Path]) -> float:
        """
        Retorna tamanho do arquivo em MB.

        Args:
            file_path: Caminho do arquivo

        Returns:
            Tamanho em MB (ou 0.0 se arquivo nao existe)

        Exemplo:
            >>> pm = PathManager()
            >>> size = pm.get_file_size_mb(__file__)
            >>> size > 0
            True
        """
        p = Path(file_path)
        if p.exists() and p.is_file():
            return p.stat().st_size / (1024 ** 2)
        return 0.0

    @staticmethod
    def list_files(directory: Union[str, Path], pattern: str = "*") -> list[Path]:
        """
        Lista arquivos em um diretorio com pattern opcional.

        Args:
            directory: Diretorio a listar
            pattern: Pattern glob (padrao: "*" = todos arquivos)

        Returns:
            Lista de Path objects

        Exemplo:
            >>> pm = PathManager()
            >>> files = pm.list_files(pm.get_project_root() / "utils", "*.py")
            >>> len(files) > 0
            True
        """
        p = Path(directory)
        if p.exists() and p.is_dir():
            return sorted(p.glob(pattern))
        return []

    @staticmethod
    def is_safe_path(path: Union[str, Path], base_dir: Union[str, Path]) -> bool:
        """
        Verifica se path esta dentro de base_dir (previne Path Traversal).

        Args:
            path: Path a verificar
            base_dir: Diretorio base permitido

        Returns:
            True se path esta dentro de base_dir, False caso contrario

        Exemplo:
            >>> pm = PathManager()
            >>> base = Path("/home/user/project")
            >>> pm.is_safe_path("/home/user/project/data/file.txt", base)
            True
            >>> pm.is_safe_path("/etc/passwd", base)
            False
        """
        try:
            p = Path(path).resolve()
            base = Path(base_dir).resolve()
            return base in p.parents or p == base
        except (ValueError, OSError):
            return False
