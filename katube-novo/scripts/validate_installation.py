#!/usr/bin/env python3
"""
Script de validação de instalação Katube 2025 CPU-First
Verifica dependências, modelos, e configuração
"""

import sys
import subprocess
from pathlib import Path
from typing import Tuple, List

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def print_header(title: str):
    """Imprime header formatado"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_check(name: str, status: bool, message: str = ""):
    """Imprime resultado de check"""
    symbol = "[OK]" if status else "[ERRO]"
    print(f"{symbol:8} {name:40} {message}")


def check_python_version() -> Tuple[bool, str]:
    """Valida versão Python >= 3.10"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor} (requer 3.10+)"


def check_dependencies() -> Tuple[bool, List[str]]:
    """Verifica dependências críticas instaladas"""
    critical_deps = [
        'torch',
        'torchaudio',
        'librosa',
        'soundfile',
        'pyannote.audio',
        'transformers',
        'textdistance',
        'pandas'
    ]

    missing = []
    for dep in critical_deps:
        try:
            __import__(dep.replace('.', '_') if '.' in dep else dep)
        except ImportError:
            missing.append(dep)

    if missing:
        return False, missing
    else:
        return True, []


def check_sox() -> Tuple[bool, str]:
    """Valida instalação SoX"""
    try:
        result = subprocess.run(
            ['sox', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            return True, version
        else:
            return False, "Sox instalado mas com erro"
    except FileNotFoundError:
        return False, "Sox não encontrado no PATH"
    except subprocess.TimeoutExpired:
        return False, "Sox timeout"
    except Exception as e:
        return False, f"Erro: {str(e)}"


def check_huggingface_token() -> Tuple[bool, str]:
    """Verifica token HuggingFace configurado"""
    env_file = Path('.env')

    if not env_file.exists():
        return False, ".env não encontrado"

    try:
        content = env_file.read_text()

        # Verificar se tem HUGGINGFACE_TOKEN definido
        if 'HUGGINGFACE_TOKEN=' not in content:
            return False, "HUGGINGFACE_TOKEN não definido"

        # Verificar se não é o placeholder
        if 'seu_token_aqui' in content:
            return False, "Token ainda é placeholder (edite .env)"

        # Extrair token
        for line in content.split('\n'):
            if line.startswith('HUGGINGFACE_TOKEN='):
                token = line.split('=', 1)[1].strip()
                if token and len(token) > 10:
                    return True, f"Token configurado ({token[:10]}...)"
                else:
                    return False, "Token vazio ou muito curto"

        return False, "Token não encontrado"

    except Exception as e:
        return False, f"Erro ao ler .env: {e}"


def check_imports_core() -> Tuple[bool, str]:
    """Testa imports dos módulos core"""
    try:
        from core.resource_manager import ResourceManager
        from core.base_processor import BaseProcessor

        return True, "Core modules OK"
    except ImportError as e:
        return False, f"Import falhou: {e}"


def check_imports_stages() -> Tuple[bool, str]:
    """Testa imports de todos os stages"""
    try:
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

        return True, "8 stages importados"
    except ImportError as e:
        return False, f"Import falhou: {e}"


def check_imports_orchestrator() -> Tuple[bool, str]:
    """Testa import do orchestrator"""
    try:
        from orchestrator import KatubeOrchestrator

        return True, "Orchestrator OK"
    except ImportError as e:
        return False, f"Import falhou: {e}"


def check_torch_cpu() -> Tuple[bool, str]:
    """Verifica que PyTorch está em modo CPU"""
    try:
        import torch

        version = torch.__version__
        is_cuda = torch.cuda.is_available()

        if is_cuda:
            return True, f"PyTorch {version} (CUDA disponível mas não necessário)"
        else:
            return True, f"PyTorch {version} (CPU-only - correto)"

    except ImportError:
        return False, "PyTorch não instalado"


def check_models_loadable() -> Tuple[bool, str]:
    """Testa se ResourceManager consegue inicializar"""
    try:
        from core.resource_manager import ResourceManager

        rm = ResourceManager(max_cache_size_mb=100)

        # Apenas testar inicialização, não carregar modelos (requer HF token)
        return True, "ResourceManager inicializado"

    except Exception as e:
        return False, f"Erro: {e}"


def check_directory_structure() -> Tuple[bool, str]:
    """Valida estrutura de diretórios"""
    required_dirs = [
        'src',
        'src/core',
        'src/stages',
        'src/utils'
    ]

    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            missing.append(dir_path)

    if missing:
        return False, f"Faltando: {', '.join(missing)}"
    else:
        return True, "Estrutura correta"


def check_requirements_files() -> Tuple[bool, str]:
    """Verifica que requirements existem"""
    required_files = [
        'requirements_fase_01.txt',
        'requirements_fase_02.txt',
        'requirements_fase_03.txt',
        'requirements_fase_04.txt'
    ]

    missing = []
    for req_file in required_files:
        if not Path(req_file).exists():
            missing.append(req_file)

    if missing:
        return False, f"Faltando: {', '.join(missing)}"
    else:
        return True, "4 requirements OK"


def check_documentation() -> Tuple[bool, str]:
    """Verifica que documentação existe"""
    required_docs = [
        'README.md',
        'INSTALL.md',
        'MIGRATION.md',
        '.env.example'
    ]

    missing = []
    for doc_file in required_docs:
        if not Path(doc_file).exists():
            missing.append(doc_file)

    if missing:
        return False, f"Faltando: {', '.join(missing)}"
    else:
        return True, "Documentação completa"


def main():
    """Executa todas as validações"""
    print_header("Katube 2025 - Validação de Instalação")
    print("Este script verifica se a instalação está correta e completa.")

    checks = [
        ("Versão Python", check_python_version),
        ("Estrutura de diretórios", check_directory_structure),
        ("Requirements files", check_requirements_files),
        ("Documentação", check_documentation),
        ("Dependências Python", check_dependencies),
        ("SoX instalado", check_sox),
        ("Token HuggingFace", check_huggingface_token),
        ("Imports Core", check_imports_core),
        ("Imports Stages", check_imports_stages),
        ("Imports Orchestrator", check_imports_orchestrator),
        ("PyTorch CPU", check_torch_cpu),
        ("ResourceManager", check_models_loadable),
    ]

    print_header("Executando Checks")

    results = []
    for name, check_func in checks:
        try:
            status, message = check_func()
            print_check(name, status, message)
            results.append(status)
        except Exception as e:
            print_check(name, False, f"Exceção: {e}")
            results.append(False)

    # Sumário
    print_header("Sumário")
    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\nTotal de checks: {total}")
    print(f"[OK]   Passou: {passed}")
    print(f"[ERRO] Falhou: {failed}")

    if failed == 0:
        print("\n" + "=" * 70)
        print("  INSTALAÇÃO COMPLETA E FUNCIONAL!")
        print("=" * 70)
        print("\nPróximos passos:")
        print("1. Configurar .env com seu token HuggingFace (se ainda não fez)")
        print("2. Aceitar termos dos modelos:")
        print("   - https://hf.co/pyannote/speaker-diarization-3.1")
        print("   - https://hf.co/pyannote/segmentation-3.0")
        print("3. Ler README.md para instruções de uso")
        print("4. Testar com: python -c 'from orchestrator import KatubeOrchestrator'")
        return 0
    else:
        print("\n" + "=" * 70)
        print("  INSTALAÇÃO INCOMPLETA")
        print("=" * 70)
        print("\nCorreções necessárias:")

        if not results[checks.index(("Dependências Python", check_dependencies))]:
            print("- Instalar dependências faltando (ver requirements_fase_*.txt)")

        if not results[checks.index(("SoX instalado", check_sox))]:
            print("- Instalar SoX:")
            print("  Windows: winget install --id ChrisBagwell.SoX")
            print("  Linux: sudo apt-get install sox")
            print("  macOS: brew install sox")

        if not results[checks.index(("Token HuggingFace", check_huggingface_token))]:
            print("- Configurar token HuggingFace no arquivo .env")
            print("  1. Copiar .env.example para .env")
            print("  2. Obter token: https://huggingface.co/settings/tokens")
            print("  3. Editar .env e adicionar token")

        print("\nConsulte INSTALL.md para instruções detalhadas.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
