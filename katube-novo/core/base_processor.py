"""
Classe base abstrata para processadores de pipeline.
Garante interface padronizada e integracao com ResourceManager.
"""
from abc import ABC, abstractmethod
from typing import Any
from contextlib import contextmanager
import logging

from core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Classe base para todos processadores de etapas do pipeline.

    Subclasses devem implementar:
    - process(): Logica principal de processamento
    - load_resources(): Carregamento de modelos/recursos
    - unload_resources(): Liberacao de recursos

    Exemplo de subclasse:
        >>> class MyProcessor(BaseProcessor):
        ...     def process(self, data):
        ...         return f"processed: {data}"
        ...     def load_resources(self):
        ...         pass
        ...     def unload_resources(self):
        ...         pass
        >>> rm = ResourceManager()
        >>> proc = MyProcessor(rm)
        >>> proc.process("test")
        'processed: test'
    """

    def __init__(self, resource_manager: ResourceManager):
        """
        Inicializa processador.

        Args:
            resource_manager: Gerenciador de recursos compartilhado
        """
        self.resource_manager = resource_manager
        self._resources_loaded = False
        logger.debug(f"[DEBUG] {self.__class__.__name__} inicializado")

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Metodo principal de processamento (IMPLEMENTAR EM SUBCLASSES).

        Args:
            input_data: Dados de entrada (tipo especifico por stage)

        Returns:
            Dados processados (tipo especifico por stage)

        Raises:
            NotImplementedError: Se nao implementado em subclasse
        """
        pass

    @abstractmethod
    def load_resources(self) -> None:
        """
        Carrega recursos necessarios (IMPLEMENTAR EM SUBCLASSES).

        Deve usar self.resource_manager.load_model() para carregar
        modelos com lazy loading e cache.

        Exemplo:
            def load_resources(self):
                self.model = self.resource_manager.load_model(
                    'my_model',
                    lambda: load_my_model()
                )
        """
        pass

    @abstractmethod
    def unload_resources(self) -> None:
        """
        Libera recursos apos processamento (IMPLEMENTAR EM SUBCLASSES).

        Deve usar self.resource_manager.unload_model() para descarregar
        modelos e liberar memoria.

        Exemplo:
            def unload_resources(self):
                self.resource_manager.unload_model('my_model')
                self._resources_loaded = False
        """
        pass

    @contextmanager
    def managed_processing(self):
        """
        Context manager para processamento com gestao automatica de recursos.

        Carrega recursos ao entrar no contexto e permite reuso.
        Nao descarrega automaticamente para permitir cache entre chamadas.

        Uso:
            >>> class TestProcessor(BaseProcessor):
            ...     def process(self, data):
            ...         return data
            ...     def load_resources(self):
            ...         self._resources_loaded = True
            ...     def unload_resources(self):
            ...         self._resources_loaded = False
            >>> rm = ResourceManager()
            >>> proc = TestProcessor(rm)
            >>> with proc.managed_processing():
            ...     result = proc.process("test")
            >>> result
            'test'
        """
        try:
            if not self._resources_loaded:
                logger.info(f"[CONFIG] Carregando recursos de {self.__class__.__name__}...")
                self.load_resources()
                self._resources_loaded = True
                logger.info(f"[OK] Recursos de {self.__class__.__name__} carregados")
            else:
                logger.debug(f"[DEBUG] Recursos de {self.__class__.__name__} ja carregados (reuso)")
            yield
        finally:
            # Nao descarrega automaticamente para permitir reuso e cache
            pass

    def is_ready(self) -> bool:
        """
        Verifica se o processador esta pronto (recursos carregados).

        Returns:
            True se recursos estao carregados, False caso contrario

        Exemplo:
            >>> class TestProcessor(BaseProcessor):
            ...     def process(self, data):
            ...         return data
            ...     def load_resources(self):
            ...         pass
            ...     def unload_resources(self):
            ...         pass
            >>> rm = ResourceManager()
            >>> proc = TestProcessor(rm)
            >>> proc.is_ready()
            False
            >>> proc._resources_loaded = True
            >>> proc.is_ready()
            True
        """
        return self._resources_loaded

    def reset(self) -> None:
        """
        Reseta o processador, descarregando recursos.

        Util para forcar recarregamento de modelos ou limpar estado.

        Exemplo:
            >>> class TestProcessor(BaseProcessor):
            ...     def process(self, data):
            ...         return data
            ...     def load_resources(self):
            ...         self._resources_loaded = True
            ...     def unload_resources(self):
            ...         self._resources_loaded = False
            >>> rm = ResourceManager()
            >>> proc = TestProcessor(rm)
            >>> proc._resources_loaded = True
            >>> proc.reset()
            >>> proc._resources_loaded
            False
        """
        if self._resources_loaded:
            logger.info(f"[INFO] Resetando {self.__class__.__name__}...")
            self.unload_resources()
            logger.info(f"[OK] {self.__class__.__name__} resetado")
