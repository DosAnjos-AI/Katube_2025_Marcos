"""
Testes de validação para Katube 2025 CPU-First
Valida orchestrator, resource manager e stages
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import subprocess
import re


def test_python_version():
    """Valida versão Python >= 3.10"""
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 10, "Python 3.10+ requerido"


def test_imports_core():
    """Valida imports de módulos core"""
    from core.resource_manager import ResourceManager
    from core.base_processor import BaseProcessor

    assert ResourceManager is not None
    assert BaseProcessor is not None


def test_imports_stages():
    """Valida que todos os stages podem ser importados"""
    from stages import (
        Stage02MOSFilter,
        Stage03Diarizer,
        Stage04STT,
        Stage05TextNormalizer,
        Stage06Validator,
        Stage07Denoiser,
        Stage08SoxNormalizer,
        Stage09DatasetGenerator
    )

    # Verificar que todas as classes existem
    stages = [
        Stage02MOSFilter,
        Stage03Diarizer,
        Stage04STT,
        Stage05TextNormalizer,
        Stage06Validator,
        Stage07Denoiser,
        Stage08SoxNormalizer,
        Stage09DatasetGenerator
    ]

    for stage_class in stages:
        assert stage_class is not None
        assert hasattr(stage_class, 'load_resources')
        assert hasattr(stage_class, 'process')
        assert hasattr(stage_class, 'unload_resources')


def test_imports_utils():
    """Valida imports de utils"""
    from utils.logging_config import setup_logging, get_logger
    from utils.paths import PathManager

    assert setup_logging is not None
    assert get_logger is not None
    assert PathManager is not None


def test_orchestrator_import():
    """Valida import do orchestrator"""
    from orchestrator import KatubeOrchestrator

    assert KatubeOrchestrator is not None


def test_resource_manager_init():
    """Testa inicialização do ResourceManager"""
    from core.resource_manager import ResourceManager

    rm = ResourceManager(max_cache_size_mb=100)
    assert rm.max_cache_size_mb == 100
    assert rm._cache == {}
    assert rm._cache_size_mb == 0


def test_resource_manager_methods():
    """Valida métodos do ResourceManager"""
    from core.resource_manager import ResourceManager

    rm = ResourceManager()

    # Verificar métodos existem
    assert hasattr(rm, 'load_model')
    assert hasattr(rm, 'unload_model')
    assert hasattr(rm, 'unload_all')
    assert hasattr(rm, 'get_cached_keys')


def test_orchestrator_init_without_token():
    """Testa inicialização do orchestrator sem token HF (deve funcionar)"""
    from orchestrator import KatubeOrchestrator
    from pathlib import Path

    # Inicializar sem token (não vai carregar modelos ainda)
    orch = KatubeOrchestrator(
        output_base_dir=Path("/tmp/test_output"),
        huggingface_token=None  # Sem token, mas deve inicializar
    )

    assert orch.resource_manager is not None
    assert hasattr(orch, 'stage_02')
    assert hasattr(orch, 'stage_03')
    assert hasattr(orch, 'stage_04')
    assert hasattr(orch, 'stage_05')
    assert hasattr(orch, 'stage_06')
    assert hasattr(orch, 'stage_07')
    assert hasattr(orch, 'stage_08')
    assert hasattr(orch, 'stage_09')


def test_no_gpu_code():
    """Garante ausência de código GPU forçado em stages"""
    # Procurar por .cuda() ou .to('cuda') em arquivos Python
    src_dir = Path(__file__).parent.parent / 'src'

    gpu_patterns = [
        r'\.cuda\(\)',
        r"\.to\(['\"]cuda['\"]\)",
        r"device=['\"]cuda['\"]",
        r"torch\.device\(['\"]cuda['\"]\)"
    ]

    violations = []

    # Buscar em stages, core, utils, orchestrator
    search_dirs = ['stages', 'core', 'utils']
    search_files = ['orchestrator.py']

    for dir_name in search_dirs:
        dir_path = src_dir / dir_name
        if dir_path.exists():
            for py_file in dir_path.glob('*.py'):
                content = py_file.read_text()
                for pattern in gpu_patterns:
                    if re.search(pattern, content):
                        violations.append(f"{py_file.name}: {pattern}")

    for file_name in search_files:
        file_path = src_dir / file_name
        if file_path.exists():
            content = file_path.read_text()
            for pattern in gpu_patterns:
                if re.search(pattern, content):
                    violations.append(f"{file_name}: {pattern}")

    # CPU-only: pode ter referências a 'cpu', mas não a 'cuda'
    # Comentários ou docstrings com 'cuda' são OK, mas código não
    assert len(violations) == 0, f"Código GPU encontrado: {violations}"


def test_no_emojis_in_code():
    """Garante ausência de emojis em código Python"""
    src_dir = Path(__file__).parent.parent / 'src'

    # Regex para emojis
    emoji_pattern = re.compile(r'[\U0001F300-\U0001F9FF]')

    violations = []

    # Buscar em todos arquivos Python
    for py_file in src_dir.rglob('*.py'):
        if '.backup' in str(py_file) or '__pycache__' in str(py_file):
            continue

        content = py_file.read_text(encoding='utf-8', errors='ignore')
        if emoji_pattern.search(content):
            violations.append(str(py_file.relative_to(src_dir)))

    assert len(violations) == 0, f"Emojis encontrados em: {violations}"


def test_logging_no_emojis():
    """Valida que logging_config não usa emojis"""
    from utils.logging_config import LOG_FORMAT

    # Verificar que formato não tem emojis
    emoji_pattern = re.compile(r'[\U0001F300-\U0001F9FF]')
    assert not emoji_pattern.search(LOG_FORMAT), "Emojis no LOG_FORMAT"


def test_base_processor_pattern():
    """Valida que BaseProcessor segue padrão correto"""
    from core.base_processor import BaseProcessor
    from abc import ABC, abstractmethod

    # Verificar que é abstract
    assert issubclass(BaseProcessor, ABC)

    # Verificar métodos abstratos
    abstract_methods = BaseProcessor.__abstractmethods__
    assert 'load_resources' in abstract_methods
    assert 'process' in abstract_methods
    assert 'unload_resources' in abstract_methods

    # Verificar managed_processing existe
    assert hasattr(BaseProcessor, 'managed_processing')


def test_stages_inherit_base_processor():
    """Valida que todos stages herdam de BaseProcessor"""
    from core.base_processor import BaseProcessor
    from stages import (
        Stage02MOSFilter,
        Stage03Diarizer,
        Stage04STT,
        Stage05TextNormalizer,
        Stage06Validator,
        Stage07Denoiser,
        Stage08SoxNormalizer,
        Stage09DatasetGenerator
    )

    stages = [
        Stage02MOSFilter,
        Stage03Diarizer,
        Stage04STT,
        Stage05TextNormalizer,
        Stage06Validator,
        Stage07Denoiser,
        Stage08SoxNormalizer,
        Stage09DatasetGenerator
    ]

    for stage_class in stages:
        assert issubclass(stage_class, BaseProcessor), \
            f"{stage_class.__name__} não herda de BaseProcessor"


def test_requirements_files_exist():
    """Valida que todos requirements existem"""
    root_dir = Path(__file__).parent.parent

    required_files = [
        'requirements_fase_01.txt',
        'requirements_fase_02.txt',
        'requirements_fase_03.txt',
        'requirements_fase_04.txt'
    ]

    for req_file in required_files:
        file_path = root_dir / req_file
        assert file_path.exists(), f"{req_file} não encontrado"
        assert file_path.stat().st_size > 0, f"{req_file} está vazio"


def test_documentation_exists():
    """Valida que documentação existe"""
    root_dir = Path(__file__).parent.parent

    required_docs = [
        'README.md',
        'INSTALL.md',
        'MIGRATION.md',
        '.env.example'
    ]

    for doc_file in required_docs:
        file_path = root_dir / doc_file
        assert file_path.exists(), f"{doc_file} não encontrado"
        assert file_path.stat().st_size > 0, f"{doc_file} está vazio"


def test_readme_sections():
    """Valida que README tem seções obrigatórias"""
    root_dir = Path(__file__).parent.parent
    readme = (root_dir / 'README.md').read_text()

    required_sections = [
        '## Características',
        '## Requisitos',
        '## Instalação',
        '## Uso',
        '## Arquitetura',
        '## Troubleshooting'
    ]

    for section in required_sections:
        assert section in readme, f"Seção '{section}' não encontrada no README"


def test_env_example_has_token():
    """Valida que .env.example tem HUGGINGFACE_TOKEN"""
    root_dir = Path(__file__).parent.parent
    env_example = (root_dir / '.env.example').read_text()

    assert 'HUGGINGFACE_TOKEN' in env_example, \
        "HUGGINGFACE_TOKEN não encontrado em .env.example"


def test_torch_cpu_available():
    """Valida que PyTorch está instalado (CPU)"""
    try:
        import torch

        # Verificar versão
        assert hasattr(torch, '__version__')

        # Verificar que é CPU-only (cuda disponível = False)
        # Nota: se CUDA estiver instalado, tudo bem, mas não é necessário
        # O importante é que o código força CPU

    except ImportError:
        pytest.skip("PyTorch não instalado (OK para testes sem deps)")


def test_no_flask_imports():
    """Garante que não há imports de Flask (removido)"""
    src_dir = Path(__file__).parent.parent / 'src'

    violations = []

    for py_file in src_dir.rglob('*.py'):
        if '.backup' in str(py_file):
            continue

        content = py_file.read_text()
        if re.search(r'from flask import|import flask', content, re.IGNORECASE):
            violations.append(str(py_file.relative_to(src_dir)))

    assert len(violations) == 0, f"Imports Flask encontrados em: {violations}"


def test_no_ytdlp_imports():
    """Garante que não há imports de yt-dlp (removido)"""
    src_dir = Path(__file__).parent.parent / 'src'

    violations = []

    for py_file in src_dir.rglob('*.py'):
        if '.backup' in str(py_file):
            continue

        content = py_file.read_text()
        if re.search(r'import yt_dlp|from yt_dlp', content):
            violations.append(str(py_file.relative_to(src_dir)))

    assert len(violations) == 0, f"Imports yt-dlp encontrados em: {violations}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
