# Guia de Instalação - Katube 2025

Guia passo a passo detalhado para instalação do Katube 2025 em Windows e Linux.

## Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Instalação Python](#instalação-python)
3. [Instalação SoX](#instalação-sox)
4. [Configuração HuggingFace](#configuração-huggingface)
5. [Instalação do Projeto](#instalação-do-projeto)
6. [Verificação](#verificação)
7. [Troubleshooting](#troubleshooting)

---

## Pré-requisitos

### Hardware Mínimo

- **CPU**: Quad-core 2.5GHz+ (recomendado: 8+ cores)
- **RAM**: 8GB mínimo (recomendado: 16GB)
- **Disco**: 20GB livres (10GB para modelos + 10GB workspace)
- **GPU**: Não necessária (CPU-only)

### Software Base

- **Sistema Operacional**:
  - Windows 10/11 (64-bit)
  - Linux (Ubuntu 20.04+, Debian 11+, ou similar)
  - macOS 11+ (não testado oficialmente)
- **Python**: 3.10 ou 3.11 (3.12+ pode ter incompatibilidades)
- **Git**: Para clonar repositório

---

## Instalação Python

### Windows

#### Opção 1: Microsoft Store (Recomendado)

1. Abrir Microsoft Store
2. Buscar "Python 3.11"
3. Clicar em "Obter" e instalar
4. Verificar instalação:
   ```powershell
   python --version
   # Deve retornar: Python 3.11.x
   ```

#### Opção 2: python.org

1. Acessar: https://www.python.org/downloads/windows/
2. Baixar "Python 3.11.x - Windows installer (64-bit)"
3. Executar instalador:
   - ✓ Marcar "Add Python 3.11 to PATH"
   - ✓ Marcar "Install pip"
   - Clicar "Install Now"
4. Verificar:
   ```powershell
   python --version
   pip --version
   ```

#### Opção 3: WinGet

```powershell
winget install Python.Python.3.11
```

### Linux (Ubuntu/Debian)

```bash
# Atualizar sistema
sudo apt update
sudo apt upgrade -y

# Instalar Python 3.11 e ferramentas
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Verificar instalação
python3.11 --version
pip3 --version
```

### macOS

```bash
# Homebrew (instalar se não tiver: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)")
brew install python@3.11

# Verificar
python3.11 --version
```

---

## Instalação SoX

SoX (Sound eXchange) é necessário para Stage 08 (normalização de áudio).

### Windows

#### Opção 1: WinGet (Recomendado)

```powershell
winget install --id ChrisBagwell.SoX --source winget
```

#### Opção 2: Download Manual

1. Acessar: https://sourceforge.net/projects/sox/files/sox/
2. Baixar "sox-14.4.2-win32.zip"
3. Extrair para `C:\Program Files\sox`
4. Adicionar ao PATH:
   - Abrir "Variáveis de Ambiente"
   - Editar "Path" em "Variáveis do Sistema"
   - Adicionar: `C:\Program Files\sox`
   - Reiniciar terminal

#### Verificar Instalação

```powershell
sox --version
# Deve retornar: sox: SoX v14.4.2
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y sox libsox-fmt-all
sox --version
```

### macOS

```bash
brew install sox
sox --version
```

---

## Configuração HuggingFace

Os modelos de diarização (pyannote) requerem autenticação HuggingFace.

### 1. Criar Conta HuggingFace

1. Acessar: https://huggingface.co/join
2. Preencher formulário e criar conta
3. Confirmar e-mail

### 2. Obter Token de Acesso

1. Login em: https://huggingface.co/
2. Acessar: https://huggingface.co/settings/tokens
3. Clicar "New token"
   - Name: `katube_access`
   - Role: `read`
4. Copiar token (formato: `hf_xxxxxxxxxxxxx`)
5. **IMPORTANTE**: Salvar token em local seguro

### 3. Aceitar Termos dos Modelos

**CRÍTICO**: Você DEVE aceitar os termos de uso dos modelos pyannote.

1. Acessar: https://huggingface.co/pyannote/speaker-diarization-3.1
   - Clicar "Agree and access repository"

2. Acessar: https://huggingface.co/pyannote/segmentation-3.0
   - Clicar "Agree and access repository"

**Sem aceitar termos, o pipeline falhará com erro de autenticação!**

---

## Instalação do Projeto

### 1. Clonar Repositório

```bash
# Via HTTPS
git clone https://github.com/DosAnjos-AI/Katube_2025_Marcos.git
cd Katube_2025_Marcos/katube-novo

# Via SSH (se configurado)
git clone git@github.com:DosAnjos-AI/Katube_2025_Marcos.git
cd Katube_2025_Marcos/katube-novo
```

### 2. Criar Ambiente Virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Se erro de execução de scripts:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Verificar ativação** (prompt deve mostrar `(venv)` no início):
```bash
(venv) $ which python  # Linux/Mac
(venv) PS> where.exe python  # Windows
```

### 3. Atualizar pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Instalar Dependências (ORDEM CRÍTICA!)

**IMPORTANTE**: Instalar na ordem correta para evitar conflitos.

#### Fase 01: PyTorch CPU

```bash
pip install -r requirements_fase_01.txt
```

**Validar**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
# Esperado: PyTorch 2.8.0 | CUDA: False
```

#### Fase 02: Audio Libraries

```bash
pip install -r requirements_fase_02.txt
```

**Validar**:
```bash
python -c "import librosa, soundfile; print('Audio libs OK')"
```

#### Fase 03: Pyannote e Transformers

```bash
pip install -r requirements_fase_03.txt
```

**Validar**:
```bash
python -c "from pyannote.audio import Pipeline; from transformers import AutoModel; print('ML libs OK')"
```

#### Fase 04: Utils

```bash
pip install -r requirements_fase_04.txt
```

**Validar**:
```bash
python -c "import textdistance, pandas; print('Utils OK')"
```

### 5. Configurar Variáveis de Ambiente

```bash
# Copiar template
cp .env.example .env

# Editar .env
# Windows:
notepad .env

# Linux/Mac:
nano .env  # ou vim .env
```

**Editar .env e preencher**:
```env
HUGGINGFACE_TOKEN=hf_seu_token_aqui  # Token do passo 3.2
MOS_THRESHOLD=2.5
SIMILARITY_THRESHOLD=0.75
MAX_CACHE_SIZE_MB=4096
ENABLE_MOS_FILTER=true
ENABLE_DENOISER=false
```

---

## Verificação

### Script de Verificação Completa

Criar arquivo `verify_install.py`:

```python
#!/usr/bin/env python
"""Verifica instalação do Katube 2025"""

import sys
from pathlib import Path

def check_python():
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        print(f"❌ Python {version.major}.{version.minor} detectado. Requer 3.10+")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_imports():
    tests = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("pyannote.audio", "Pyannote.audio"),
        ("transformers", "Transformers"),
        ("textdistance", "TextDistance"),
        ("pandas", "Pandas")
    ]

    all_ok = True
    for module, name in tests:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} não encontrado")
            all_ok = False
    return all_ok

def check_sox():
    import subprocess
    try:
        result = subprocess.run(
            ['sox', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✓ SoX instalado")
            return True
    except FileNotFoundError:
        print("❌ SoX não encontrado")
    except Exception as e:
        print(f"❌ Erro ao verificar SoX: {e}")
    return False

def check_env():
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ Arquivo .env não encontrado")
        return False

    content = env_file.read_text()
    if 'seu_token_aqui' in content or 'HUGGINGFACE_TOKEN=' not in content:
        print("⚠ Arquivo .env não configurado (token HuggingFace)")
        return False

    print("✓ Arquivo .env configurado")
    return True

def check_torch_cpu():
    import torch
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print("⚠ CUDA disponível (esperado CPU-only)")
    else:
        print("✓ PyTorch CPU-only (correto)")
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("Katube 2025 - Verificação de Instalação")
    print("=" * 60)

    checks = [
        ("Python", check_python),
        ("Imports", check_imports),
        ("SoX", check_sox),
        ("Configuração", check_env),
        ("PyTorch CPU", check_torch_cpu)
    ]

    results = []
    for name, func in checks:
        print(f"\n[{name}]")
        results.append(func())

    print("\n" + "=" * 60)
    if all(results):
        print("✓ Instalação completa e funcional!")
    else:
        print("❌ Instalação incompleta. Veja erros acima.")
        sys.exit(1)
```

Executar:
```bash
python verify_install.py
```

---

## Troubleshooting

### Erro: "Python não reconhecido"

**Windows**:
1. Reinstalar Python marcando "Add to PATH"
2. Ou adicionar manualmente:
   - `C:\Users\SeuUsuario\AppData\Local\Programs\Python\Python311`
   - `C:\Users\SeuUsuario\AppData\Local\Programs\Python\Python311\Scripts`

**Linux**:
```bash
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
```

### Erro: "pip install" falha com SSL

```bash
# Windows
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pip setuptools

# Linux
pip install --upgrade certifi
```

### Erro: "Microsoft Visual C++ required" (Windows)

Instalar Build Tools:
1. Baixar: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Instalar "Desktop development with C++"

### Erro: "No module named '_lzma'" (Linux)

```bash
sudo apt install liblzma-dev
python3.11 -m pip install --upgrade --force-reinstall lzma
```

### Erro: Conflito de versões PyTorch

```bash
# Limpar e reinstalar
pip uninstall -y torch torchaudio transformers
pip cache purge
pip install -r requirements_fase_01.txt
```

### Ambiente virtual não ativa

**PowerShell (Windows)**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux**: Verificar se criou com `python3.11 -m venv` (não `virtualenv`)

### SoX não encontrado após instalação

**Windows**:
1. Reiniciar terminal/PowerShell
2. Verificar PATH: `$env:Path -split ';'` (PowerShell)
3. Adicionar manualmente se necessário

**Linux**:
```bash
which sox
# Se vazio:
sudo updatedb
locate sox
# Adicionar ao PATH se necessário
```

### Erro: "huggingface_hub.utils._errors.RepositoryNotFoundError"

1. Verificar token no `.env`
2. Aceitar termos dos modelos (passo 4.3)
3. Testar token:
   ```bash
   python -c "from huggingface_hub import HfApi; HfApi().whoami(token='hf_seu_token')"
   ```

---

## Próximos Passos

Após instalação bem-sucedida:

1. Ler [README.md](README.md) para uso básico
2. Ler [MIGRATION.md](MIGRATION.md) se vindo de versão antiga
3. Testar com exemplo:
   ```python
   from pathlib import Path
   from src.orchestrator import KatubeOrchestrator

   orchestrator = KatubeOrchestrator(
       output_base_dir=Path("./output"),
       huggingface_token="hf_seu_token"
   )
   print("✓ Katube pronto para uso!")
   ```

---

## Suporte

- **Issues**: https://github.com/DosAnjos-AI/Katube_2025_Marcos/issues
- **Documentação**: [README.md](README.md)

---

**Última atualização**: 2025-11-15
**Versão do guia**: 2.0.0
