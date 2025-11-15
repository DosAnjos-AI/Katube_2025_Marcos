# 🚀 START HERE - Katube 2025 CPU-First

**Bem-vindo ao Katube 2025!** Pipeline automatizado para criação de datasets TTS de alta qualidade, otimizado para processamento CPU.

---

## ⚡ INÍCIO RÁPIDO - 4 Passos

### 1. Instalar Dependências (ORDEM OBRIGATÓRIA)

```bash
# IMPORTANTE: Instalar na ordem correta!
pip install -r requirements_fase_01.txt  # PyTorch CPU
pip install -r requirements_fase_02.txt  # Audio libs
pip install -r requirements_fase_03.txt  # Pyannote e Transformers
pip install -r requirements_fase_04.txt  # Utils
```

**Por que 4 fases?**
- Fase 01: PyTorch CPU-only (sem CUDA, economiza ~2GB download)
- Fase 02: Bibliotecas de áudio (librosa, soundfile, etc)
- Fase 03: Modelos ML (pyannote, transformers)
- Fase 04: Utilitários (DeepFilterNet, textdistance, pandas)

### 2. Instalar SoX (Sistema)

**Windows:**
```powershell
winget install --id ChrisBagwell.SoX
```

**Linux:**
```bash
sudo apt-get install sox libsox-fmt-all
```

**macOS:**
```bash
brew install sox
```

### 3. Configurar Token HuggingFace

```bash
# Copiar template
cp .env.example .env

# Editar .env e adicionar seu token
# .env:
HUGGINGFACE_TOKEN=hf_seu_token_aqui
```

**Como obter o token:**
1. Acesse: https://huggingface.co/settings/tokens
2. Crie um novo token (tipo: read)
3. **IMPORTANTE**: Aceite os termos dos modelos:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

### 4. Validar Instalação

```bash
python scripts/validate_installation.py
```

**Output esperado:**
```
[OK]   Versão Python
[OK]   Dependências Python
[OK]   SoX instalado
[OK]   Token HuggingFace
[OK]   Imports Core
[OK]   Imports Stages
...
✓ INSTALAÇÃO COMPLETA E FUNCIONAL!
```

---

## 🎯 Como Usar

### Opção 1: Python API (Recomendado)

```python
from pathlib import Path
from src.orchestrator import KatubeOrchestrator

# Configurar orchestrator
orchestrator = KatubeOrchestrator(
    output_base_dir=Path("./output"),
    huggingface_token="hf_seu_token_aqui",
    max_cache_mb=4096,
    enable_mos_filter=True,
    enable_denoiser=False
)

# Processar áudios
audio_paths = [
    Path("audio1.wav"),
    Path("audio2.wav")
]

result = orchestrator.process_audio_pipeline(
    audio_paths=audio_paths,
    session_name="minha_sessao",
    num_speakers=2,
    save_intermediates=True
)

# Verificar resultados
print(f"Dataset: {result['dataset']['output_path']}")
print(f"Válidos: {result['dataset']['valid_count']}")
```

### Opção 2: CLI (TODO - Em desenvolvimento)

```bash
# Quando run_pipeline.py estiver pronto:
python run_pipeline.py \
  --audio /caminho/para/audios \
  --session nome_sessao \
  --speakers 2 \
  --mos-threshold 3.0
```

---

## 🏗️ Arquitetura - 9 Stages

O Katube processa áudios em 9 etapas independentes:

| # | Stage | Modelo | RAM | Descrição |
|---|-------|--------|-----|-----------|
| 01 | **Segmentação** | - | CPU | Corte inteligente (TODO) |
| 02 | **Filtro MOS** | SHEET (~1GB) | ~1.5GB | Qualidade de áudio |
| 03 | **Diarização** | pyannote (~2GB) | ~2.5GB | Identificação speakers |
| 04 | **STT** | Whisper + WAV2VEC2 | ~3GB* | Transcrição dual |
| 05 | **Normalização** | - | CPU | Padronização texto |
| 06 | **Validação** | - | CPU | Levenshtein similarity |
| 07 | **Denoising** | DeepFilterNet (~1GB) | ~1.5GB | Remoção ruído (opcional) |
| 08 | **Sox** | - | CPU | Normalização final |
| 09 | **Dataset** | - | CPU | Geração CSV |

**\*Stage 04**: Modelos carregados **sequencialmente** (não simultâneos) para economizar RAM.

---

## 📁 Estrutura de Arquivos

### Código

```
katube-novo/
├── src/
│   ├── core/                   # Módulos centrais
│   │   ├── base_processor.py   # Classe base abstrata
│   │   └── resource_manager.py # Gestão de memória
│   ├── stages/                 # 9 etapas do pipeline
│   │   ├── stage_02_mos_filter.py
│   │   ├── stage_03_diarizer.py
│   │   ├── stage_04_stt.py
│   │   ├── stage_05_normalizer.py
│   │   ├── stage_06_validator.py
│   │   ├── stage_07_denoiser.py
│   │   ├── stage_08_sox.py
│   │   └── stage_09_dataset.py
│   ├── utils/                  # Utilitários
│   │   ├── logging_config.py
│   │   └── paths.py
│   └── orchestrator.py         # Orquestrador principal
├── tests/                      # Testes unitários
├── scripts/                    # Scripts de validação
├── requirements_fase_*.txt     # Dependências (4 fases)
├── .env.example                # Template configuração
├── README.md                   # Documentação completa
├── INSTALL.md                  # Guia de instalação
├── MIGRATION.md                # Migração v1→v2
└── CHANGELOG.md                # Histórico de mudanças
```

### Output (após processamento)

```
output/sessao_nome/
├── stage_02_mos/
│   ├── approved/               # MOS ≥ 3.0
│   ├── medium/                 # MOS 2.5-3.0
│   └── rejected/               # MOS < 2.5
├── stage_03_diarization/
│   ├── speaker_01/
│   ├── speaker_02/
│   └── timeline.rttm
├── stage_04_stt/
│   ├── whisper_results/
│   ├── wav2vec2_results/
│   └── combined_results.json
├── stage_06_validation/
│   ├── approved/               # Similarity ≥ 0.75
│   └── rejected/               # Similarity < 0.75
├── stage_07_denoised/          # Se enable_denoiser=True
├── stage_08_normalized/        # Áudios finais
└── dataset.csv                 # Dataset consolidado (Stage 09)
```

---

## 🎓 Documentação Completa

Para mais detalhes, consulte:

- **[README.md](README.md)**: Documentação completa (11KB)
  - Instalação detalhada
  - Arquitetura de 9 stages
  - Otimizações e troubleshooting
  - Comparativo old vs new

- **[INSTALL.md](INSTALL.md)**: Guia de instalação (12KB)
  - Python/SoX por plataforma
  - Configuração HuggingFace
  - Script de verificação
  - Troubleshooting (10 problemas)

- **[MIGRATION.md](MIGRATION.md)**: Migração v1→v2 (16KB)
  - Breaking changes
  - Comparativo detalhado
  - Migration checklist
  - FAQ

- **[CHANGELOG.md](CHANGELOG.md)**: Histórico (12KB)
  - Versão 2.0.0 completa
  - Performance benchmarks
  - Roadmap futuro

---

## 🔧 Configuração Avançada

### Ajustar Thresholds

Edite `.env`:

```env
# Filtro MOS (qualidade de áudio)
MOS_THRESHOLD=2.5           # Padrão: 2.5
MOS_HIGH_THRESHOLD=3.0      # Alto: 3.0
MOS_MEDIUM_THRESHOLD=2.5    # Médio: 2.5

# Validação de transcrição
SIMILARITY_THRESHOLD=0.75   # Padrão: 0.75 (75% similaridade)

# Recursos
MAX_CACHE_SIZE_MB=4096      # Cache de modelos (padrão: 4GB)

# Stages opcionais
ENABLE_MOS_FILTER=true      # Ativar filtro MOS
ENABLE_DENOISER=false       # Ativar denoising (lento)
```

### Configurar por Stage

```python
orchestrator = KatubeOrchestrator(
    output_base_dir=Path("./output"),
    huggingface_token="hf_...",

    # Config Stage 02 (MOS)
    stage_02_config={
        'threshold_high': 3.0,
        'threshold_medium': 2.5
    },

    # Config Stage 06 (Validação)
    stage_06_config={
        'similarity_threshold': 0.80  # Mais rigoroso
    },

    # Config Stage 08 (Sox)
    stage_08_config={
        'target_sample_rate': 48000,
        'target_channels': 1,
        'normalize_gain': True
    }
)
```

---

## 💡 Dicas e Boas Práticas

### Performance

1. **RAM limitada?**
   - Reduza `MAX_CACHE_SIZE_MB` no `.env`
   - Processe menos áudios por vez
   - Desabilite denoiser (`ENABLE_DENOISER=false`)

2. **Processamento lento?**
   - Normal! CPU-only é ~2x mais lento que GPU
   - Processe em lote durante a noite
   - Use `save_intermediates=False` para economizar disco

3. **Quality vs Speed?**
   - Alta qualidade: `MOS_THRESHOLD=3.0`, `SIMILARITY_THRESHOLD=0.80`
   - Balanceado: `MOS_THRESHOLD=2.5`, `SIMILARITY_THRESHOLD=0.75` (padrão)
   - Permissivo: `MOS_THRESHOLD=2.0`, `SIMILARITY_THRESHOLD=0.70`

### Troubleshooting Rápido

**Erro: "Pipeline returned None"**
- Verifique token HuggingFace no `.env`
- Aceite termos dos modelos pyannote

**Erro: "Sox not found"**
- Instale SoX no sistema (ver Passo 2)

**Alto uso de RAM**
- Reduza `MAX_CACHE_SIZE_MB`
- Verifique que Stage 04 está usando carregamento sequencial

**Import errors**
- Reinstale requirements NA ORDEM: fase_01 → fase_02 → fase_03 → fase_04

---

## 🆘 Suporte

- **Issues**: https://github.com/DosAnjos-AI/Katube_2025_Marcos/issues
- **Validação**: `python scripts/validate_installation.py`
- **Testes**: `pytest tests/ -v` (requer pytest instalado)

---

## 📊 Performance Esperada

### Recursos (CPU-only)

- **RAM Inicialização**: ~100MB (lazy loading)
- **RAM Processamento**: ~3.5GB (média)
- **RAM Pico** (Stage 04 STT): ~3GB (sequencial)
- **Disco**: ~10GB (modelos) + workspace

### Tempo de Processamento (exemplo: 60min de áudio)

- **Download**: ~5min (depende da conexão)
- **Stage 02 (MOS)**: ~12min
- **Stage 03 (Diarização)**: ~35min
- **Stage 04 (STT)**: ~45min (Whisper) + ~40min (WAV2VEC2)
- **Stage 05-09**: ~10min (total)
- **Total**: ~2h30min (CPU i7, 16GB RAM)

**Nota**: GPU seria ~2x mais rápido, mas requer hardware caro e instalação complexa.

---

## ✅ Checklist Pré-Produção

Antes de processar datasets grandes:

- [ ] Validação passou: `python scripts/validate_installation.py`
- [ ] Token HuggingFace configurado em `.env`
- [ ] SoX instalado e funcionando
- [ ] Testado com 2-3 áudios pequenos primeiro
- [ ] Quality metrics ajustados (MOS, similarity)
- [ ] Espaço em disco suficiente (~10GB modelos + workspace)
- [ ] RAM disponível (~8GB mínimo, 16GB recomendado)

---

**Última atualização**: 2025-11-15
**Versão**: 2.0.0-cpu
**Status**: Pronto para produção

**Boa sorte com seus datasets! 🚀**
