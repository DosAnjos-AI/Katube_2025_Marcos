# MIGRATION GUIDE - Katube 2025 CPU Refactoring

## Versao: 2.0.0 - CPU-First Architecture

Data de inicio: 2025-11-15
Branch: claude/katube-cpu-refactor-01CZQsLdk1xydii4ZdwmA17o

---

## Estrutura Antiga vs Nova

### ANTES (Versao 1.x):
```
katube-novo/
├── src/
│   ├── pipeline.py (2138 linhas - monolitico)
│   ├── audio_segmenter.py (529 linhas)
│   ├── mos_filter.py (535 linhas)
│   ├── text_normalizer.py (806 linhas)
│   ├── diarizer.py (347 linhas)
│   ├── stt_whisper.py
│   ├── stt_wav2vec2.py
│   ├── stt_transcriber.py
│   ├── denoiser.py
│   ├── sox_normalizer.py
│   ├── marcos_validator.py
│   ├── naming_utils.py
│   ├── config.py
│   └── [outros modulos...]
├── app.py (Flask - NAO USADO)
├── test_diarizer.py (paths hardcoded)
└── requirements.txt (versoes conflitantes)
```

### DEPOIS (Versao 2.0):
```
katube-novo/
├── core/
│   ├── __init__.py
│   ├── base_processor.py       # Classe abstrata base
│   ├── resource_manager.py     # Lazy loading + cache LRU
│   └── config.py               # Configuracoes centralizadas
├── stages/
│   ├── __init__.py
│   ├── stage_01_segmenter.py   # Segmentacao (VAD)
│   ├── stage_02_mos_filter.py  # Filtro qualidade (SHEET)
│   ├── stage_03_diarizer.py    # Diarizacao (pyannote)
│   ├── stage_04_stt.py         # STT unificado (Whisper + WAV2VEC2 sequencial)
│   ├── stage_05_normalizer.py  # Normalizacao de texto
│   ├── stage_06_validator.py   # Validacao Levenshtein
│   ├── stage_07_denoiser.py    # Denoising (DeepFilterNet)
│   ├── stage_08_sox.py         # Normalizacao Sox
│   └── stage_09_dataset.py     # Geracao CSV final
├── utils/
│   ├── __init__.py
│   ├── paths.py                # PathManager cross-platform
│   ├── logging_config.py       # Logs padronizados
│   └── naming.py               # Utilitarios de nomenclatura
├── orchestrator.py             # Novo pipeline orquestrador (substitui pipeline.py)
├── batch_processor.py          # Processamento em lote
├── run_pipeline.py             # Interface CLI
├── requirements_fase_01.txt    # PyTorch CPU
├── requirements_fase_02.txt    # Bibliotecas de audio
├── requirements_fase_03.txt    # Pyannote
├── requirements_fase_04.txt    # Utilitarios
├── .backup/                    # Codigo original preservado
├── README.md
├── CHANGELOG.md
└── .env.example
```

---

## Objetivos da Refatoracao

### 1. CPU-FIRST (Zero GPU/CUDA)
- **ANTES**: 21 referencias a GPU/CUDA em 7 arquivos
  - Parametro `use_cuda` em 4 modulos
  - `.to(device)` em multiplos locais
  - `torch.cuda.is_available()` checks
- **DEPOIS**: 100% CPU
  - `device='cpu'` fixo em todos modelos
  - Zero referencias a CUDA
  - Otimizado para inferencia CPU

### 2. LAZY LOADING (Economia de ~60% RAM)
- **ANTES**: Todos modelos carregados no `__init__`
  - Pyannote (~2GB) + SHEET (~1GB) + Whisper (~3GB) + WAV2VEC2 (~2GB)
  - Total: 9-11GB RAM simultaneos
  - Zero garbage collection
- **DEPOIS**: Carregamento sob demanda
  - ResourceManager com cache LRU
  - Modelos carregados apenas quando necessarios
  - `@torch.no_grad()` em todas inferencias
  - `gc.collect()` apos descarregar modelos
  - Whisper e WAV2VEC2 NUNCA simultaneos em RAM

### 3. MODULARIZACAO
- **ANTES**: Pipeline monolitico (2138 linhas)
  - Dificil manutencao e debug
  - Logica misturada (orquestracao + processamento + metadados)
- **DEPOIS**: 9 stages independentes
  - Cada stage herda de `BaseProcessor`
  - Interface padronizada: `load_resources()`, `process()`, `unload_resources()`
  - Facil teste e manutencao
  - Orquestrador simples e limpo

### 4. CROSS-PLATFORM
- **ANTES**: Paths hardcoded, problemas Windows/Linux
- **DEPOIS**: `pathlib.Path` em 100% do codigo
  - `PathManager` para operacoes cross-platform
  - Paths relativos a raiz do projeto

### 5. LIMPEZA
- **ANTES**:
  - 71 emojis em logs (encoding quebrado)
  - Imports quebrados (`from src.`)
  - Arquivos obsoletos (app.py, test_diarizer.py)
- **DEPOIS**:
  - Logs profissionais com tags ASCII: [OK], [ERRO], [AVISO], [INFO]
  - Imports corretos
  - Codigo morto removido

### 6. DOCUMENTACAO
- **ANTES**: README minimo, sem guia instalacao
- **DEPOIS**:
  - README completo com arquitetura
  - INSTALL.md passo-a-passo
  - CHANGELOG detalhado
  - Type hints e docstrings em 100% do codigo

---

## Breaking Changes

### Imports
```python
# ANTES:
from src.marcos_validation.text_normalizer import process_stt_results
from pipeline import AudioProcessingPipeline

# DEPOIS:
from stages.stage_05_normalizer import Stage05Normalizer
from orchestrator import KatubeOrchestrator
```

### Inicializacao do Pipeline
```python
# ANTES:
pipeline = AudioProcessingPipeline(use_cuda=False)
pipeline.process_video(...)

# DEPOIS:
from core import ResourceManager
from orchestrator import KatubeOrchestrator

resource_manager = ResourceManager(max_cache_size_mb=4096)
orchestrator = KatubeOrchestrator(resource_manager)
orchestrator.run_pipeline(...)
```

### Configuracao
```python
# ANTES:
from config import Config
config = Config()

# DEPOIS:
from core.config import Config
config = Config()  # Mesma interface, mas sem opcoes GPU
```

---

## Mudancas Planejadas por Etapa

1. **ETAPA 01**: Setup inicial - estrutura base
2. **ETAPA 02**: Remover codigo obsoleto
3. **ETAPA 03**: Remover GPU/CUDA (21 referencias)
4. **ETAPA 04**: Remover emojis (71 ocorrencias)
5. **ETAPA 05**: Criar core (ResourceManager + BaseProcessor)
6. **ETAPA 06**: Criar utils (paths + logging)
7. **ETAPA 07**: Modularizar Stage 01 (Segmentador)
8. **ETAPA 08**: Modularizar Stage 02 (MOS Filter)
9. **ETAPA 09**: Modularizar Stage 03 (Diarizacao)
10. **ETAPA 10**: Modularizar Stage 04 (STT Unificado)
11. **ETAPA 11**: Modularizar Stages 05-09
12. **ETAPA 12**: Criar Orchestrator
13. **ETAPA 13**: Requirements finais + docs
14. **ETAPA 14**: Testes e validacao
15. **ETAPA 15**: Merge e finalizacao

---

## Performance Esperado

### RAM Usage:
- **ANTES**: 9-11GB (pico com todos modelos carregados)
- **DEPOIS**: 3-4GB (pico com lazy loading)
- **ECONOMIA**: ~60%

### Tempo de Processamento:
- **ANTES**: Baseline
- **DEPOIS**: +5-10% (overhead de carga/descarga), mas MUITO mais estavel

### Compatibilidade:
- **ANTES**: Problemas com paths Windows, precisa GPU NVIDIA
- **DEPOIS**: Windows/Linux/Mac, CPU-only (sem dependencia GPU)

---

## Regras Fundamentais Aplicadas

### REGRA 1 - PRESERVAR FUNCIONAMENTO:
- Codigo atual FUNCIONA - nao quebre
- Mudancas incrementais e validadas
- Compatibilidade de interface quando possivel

### REGRA 2 - SEM EMOJIS EM CODIGO:
- Comentarios: portugues sem emojis
- Logs: tags ASCII [OK], [ERRO], [AVISO], [INFO]
- Documentacao usuario: emojis permitidos

### REGRA 3 - EFICIENCIA:
- Lazy loading obrigatorio para modelos pesados
- `torch.no_grad()` em todas inferencias
- `gc.collect()` apos descarregar modelos
- Cache inteligente (LRU)

### REGRA 4 - PADROES:
- Type hints obrigatorios
- Docstrings completas
- `pathlib` (nunca `os.path`)
- Logging estruturado

---

## Status da Migracao

- [X] ETAPA 01: Setup inicial - CONCLUIDA
- [X] ETAPA 02: Limpeza - CONCLUIDA
  - app.py movido para .backup/ (Flask nao usado)
  - Nenhum import quebrado encontrado (codigo ja correto)
  - Zero referencias a 'from src.' no codigo ativo
- [ ] ETAPA 03: Remover GPU/CUDA
- [ ] ETAPA 04: Remover emojis
- [ ] ETAPA 05: Core modules
- [ ] ETAPA 06: Utils
- [ ] ETAPA 07-11: Stages modularizadas
- [ ] ETAPA 12: Orchestrator
- [ ] ETAPA 13: Documentacao
- [ ] ETAPA 14: Testes
- [ ] ETAPA 15: Finalizacao

---

## Backup

Codigo original preservado em `.backup/` para referencia e rollback se necessario.

## Contato

Equipe CEIA-Alcateia-AI
Projeto: Katube 2025
