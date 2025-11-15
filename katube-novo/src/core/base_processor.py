"""
Base Processor - Classe base para todos os estágios do pipeline
"""

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict

from .resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Classe base abstrata para processadores do pipeline.

    Padrão de uso:
    1. __init__: Configurar parâmetros
    2. load_resources(): Carregar modelos (lazy)
    3. process(): Executar processamento
    4. unload_resources(): Limpar memória
    """

    def __init__(self, resource_manager: ResourceManager):
        """
        Inicializa processador.

        Args:
            resource_manager: Gerenciador de recursos compartilhado
        """
        self.resource_manager = resource_manager
        self._is_initialized = False
        logger.debug(f"[INIT] {self.__class__.__name__} criado")

    @abstractmethod
    def load_resources(self) -> None:
        """
        Carrega recursos necessários (modelos, etc.).
        Implementado por cada stage específico.
        """
        pass

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Executa processamento principal.
        Implementado por cada stage específico.

        Returns:
            Resultado do processamento (depende do stage)
        """
        pass

    @abstractmethod
    def unload_resources(self) -> None:
        """
        Descarrega recursos da memória.
        Implementado por cada stage específico.
        """
        pass

    @contextmanager
    def managed_processing(self):
        """
        Context manager para processamento seguro.

        Garante:
        - Recursos carregados antes do processamento
        - Logs de início/fim
        - Exception handling

        Uso:
        ```python
        def process(self, data):
            with self.managed_processing():
                # seu código aqui
                return result
        ```
        """
        stage_name = self.__class__.__name__
        logger.info(f"[START] {stage_name} iniciado")

        try:
            # Carrega recursos se ainda não foram carregados
            if not self._is_initialized:
                self.load_resources()
                self._is_initialized = True

            yield

            logger.info(f"[OK] {stage_name} concluído")

        except Exception as e:
            logger.error(f"[ERRO] {stage_name} falhou: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        self.load_resources()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_resources()
        return False

    def get_stage_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o stage.

        Returns:
            Dicionário com metadados do stage
        """
        return {
            'name': self.__class__.__name__,
            'initialized': self._is_initialized,
        }
