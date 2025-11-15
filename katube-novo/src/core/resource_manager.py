"""
Resource Manager - Gerenciamento centralizado de recursos
Lazy loading de modelos pesados (~1-3GB)
"""

import logging
from typing import Dict, Any, Callable, Optional
import gc
import torch

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Gerenciador centralizado de recursos do pipeline.

    Responsabilidades:
    - Lazy loading de modelos pesados
    - Cache de modelos carregados
    - Limpeza de memória entre etapas
    - Forçar device='cpu' para todos os modelos
    """

    def __init__(self):
        self._loaded_models: Dict[str, Any] = {}
        self._memory_stats: Dict[str, int] = {}
        logger.info("[INIT] ResourceManager inicializado (CPU-only)")

    def load_model(self,
                   model_key: str,
                   loader_func: Callable[[], Any],
                   force_reload: bool = False) -> Any:
        """
        Carrega modelo sob demanda (lazy loading).

        Args:
            model_key: Identificador único do modelo
            loader_func: Função que carrega o modelo
            force_reload: Forçar recarga mesmo se já estiver em cache

        Returns:
            Modelo carregado
        """
        if model_key in self._loaded_models and not force_reload:
            logger.info(f"[CACHE] Modelo '{model_key}' já carregado")
            return self._loaded_models[model_key]

        logger.info(f"[LOAD] Carregando modelo '{model_key}'...")

        try:
            model = loader_func()
            self._loaded_models[model_key] = model

            # Log memory usage (estimate)
            if hasattr(model, 'parameters'):
                param_count = sum(p.numel() for p in model.parameters())
                self._memory_stats[model_key] = param_count
                logger.info(f"[OK] Modelo '{model_key}' carregado ({param_count:,} params)")
            else:
                logger.info(f"[OK] Modelo '{model_key}' carregado")

            return model

        except Exception as e:
            logger.error(f"[ERRO] Falha ao carregar '{model_key}': {e}")
            raise

    def unload_model(self, model_key: str) -> None:
        """
        Descarrega modelo da memória.

        Args:
            model_key: Identificador do modelo
        """
        if model_key in self._loaded_models:
            logger.info(f"[UNLOAD] Descarregando '{model_key}'...")
            del self._loaded_models[model_key]

            if model_key in self._memory_stats:
                del self._memory_stats[model_key]

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"[OK] Modelo '{model_key}' descarregado")
        else:
            logger.warning(f"[AVISO] Modelo '{model_key}' não estava carregado")

    def unload_all(self) -> None:
        """Descarrega todos os modelos."""
        logger.info("[CLEANUP] Descarregando todos os modelos...")

        for model_key in list(self._loaded_models.keys()):
            self.unload_model(model_key)

        logger.info("[OK] Todos os modelos descarregados")

    def get_loaded_models(self) -> list:
        """Retorna lista de modelos atualmente carregados."""
        return list(self._loaded_models.keys())

    def is_loaded(self, model_key: str) -> bool:
        """Verifica se modelo está carregado."""
        return model_key in self._loaded_models

    def get_memory_stats(self) -> Dict[str, int]:
        """Retorna estatísticas de memória."""
        return self._memory_stats.copy()
