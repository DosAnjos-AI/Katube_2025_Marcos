"""
Stage 05: Text Normalizer
Normalização de textos STT para validação e dataset
CPU-only, sem modelos (apenas processamento de texto)
"""

import re
import unicodedata
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..core.base_processor import BaseProcessor
from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class Stage05TextNormalizer(BaseProcessor):
    """
    Etapa 05: Normalização de textos para comparação e dataset.

    Input: Lista de textos (transcrições STT)
    Output: Dict com textos normalizados

    Operações:
    - Lowercasing
    - Remoção de acentos
    - Remoção de pontuação
    - Normalização de espaços
    - Conversão de números (opcional)
    """

    # Mapeamento de caracteres especiais para português
    CHARS_MAP = {
        'ï': 'i', 'ù': 'u', 'ö': 'o', 'î': 'i', 'ñ': 'n',
        'ë': 'e', 'ì': 'i', 'ò': 'o', 'ů': 'u', 'ẽ': 'e',
        'ü': 'u', 'è': 'e', 'æ': 'a', 'å': 'a', 'ø': 'o',
        'þ': 't', 'ð': 'd', 'ß': 's', 'ł': 'l', 'đ': 'd',
        'ć': 'c', 'č': 'c', 'š': 's', 'ž': 'z', 'ý': 'y'
    }

    def __init__(self,
                 resource_manager: ResourceManager,
                 remove_accents: bool = True,
                 remove_punctuation: bool = True,
                 lowercase: bool = True,
                 normalize_spaces: bool = True,
                 convert_numbers: bool = False):
        """
        Inicializa normalizador de texto.

        Args:
            resource_manager: Gerenciador de recursos
            remove_accents: Remover acentos (padrão: True)
            remove_punctuation: Remover pontuação (padrão: True)
            lowercase: Converter para minúsculas (padrão: True)
            normalize_spaces: Normalizar espaços (padrão: True)
            convert_numbers: Converter números para extenso (padrão: False)
        """
        super().__init__(resource_manager)
        self.remove_accents = remove_accents
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.normalize_spaces = normalize_spaces
        self.convert_numbers = convert_numbers

        # Compila regex patterns para performance
        self._compile_patterns()

        logger.info("[CONFIG] Text Normalizer: sem modelos (processamento CPU puro)")
        logger.info(f"[CONFIG] Opções: accents={remove_accents}, punct={remove_punctuation}, "
                    f"lower={lowercase}, spaces={normalize_spaces}, numbers={convert_numbers}")

    def _compile_patterns(self):
        """Compila regex patterns para melhor performance."""
        # Pontuação para remover
        self.punctuation_pattern = re.compile(r'[.,;:!?¿¡\"\'(){}\[\]<>/\\|@#$%^&*_+=~`]')

        # Múltiplos espaços
        self.spaces_pattern = re.compile(r'\s+')

        # Números (para conversão opcional)
        self.number_pattern = re.compile(r'\b\d+\b')

    def load_resources(self) -> None:
        """Não há recursos para carregar (processamento de texto puro)."""
        logger.debug("[INFO] Text Normalizer: sem recursos para carregar")
        pass

    def _apply_char_mapping(self, text: str) -> str:
        """
        Aplica mapeamento de caracteres especiais.

        Args:
            text: Texto de entrada

        Returns:
            Texto com caracteres mapeados
        """
        for special_char, normal_char in self.CHARS_MAP.items():
            text = text.replace(special_char, normal_char)
        return text

    def _remove_accents(self, text: str) -> str:
        """
        Remove acentos do texto.

        Args:
            text: Texto de entrada

        Returns:
            Texto sem acentos
        """
        # Normaliza para NFD (decomposição)
        nfd = unicodedata.normalize('NFD', text)
        # Remove marcas diacríticas
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')

    def _remove_punctuation(self, text: str) -> str:
        """
        Remove pontuação do texto.

        Args:
            text: Texto de entrada

        Returns:
            Texto sem pontuação
        """
        return self.punctuation_pattern.sub('', text)

    def _normalize_spaces(self, text: str) -> str:
        """
        Normaliza espaços (múltiplos espaços → único espaço).

        Args:
            text: Texto de entrada

        Returns:
            Texto com espaços normalizados
        """
        return self.spaces_pattern.sub(' ', text).strip()

    def _convert_numbers_to_words(self, text: str) -> str:
        """
        Converte números para extenso (simplificado).

        Args:
            text: Texto de entrada

        Returns:
            Texto com números convertidos

        NOTA: Implementação simplificada. Para conversão completa,
        use biblioteca externa como 'num2words'.
        """
        def replace_number(match):
            num = int(match.group())
            if 0 <= num <= 20:
                words = ["zero", "um", "dois", "três", "quatro", "cinco",
                         "seis", "sete", "oito", "nove", "dez", "onze",
                         "doze", "treze", "quatorze", "quinze", "dezesseis",
                         "dezessete", "dezoito", "dezenove", "vinte"]
                return words[num]
            else:
                # Para números maiores, manter como número
                return str(num)

        return self.number_pattern.sub(replace_number, text)

    def normalize_text(self, text: str) -> str:
        """
        Normaliza um único texto.

        Args:
            text: Texto de entrada

        Returns:
            Texto normalizado
        """
        if not text:
            return ""

        # Aplica mapeamento de caracteres especiais
        text = self._apply_char_mapping(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove acentos
        if self.remove_accents:
            text = self._remove_accents(text)

        # Remove pontuação
        if self.remove_punctuation:
            text = self._remove_punctuation(text)

        # Converte números (opcional)
        if self.convert_numbers:
            text = self._convert_numbers_to_words(text)

        # Normaliza espaços
        if self.normalize_spaces:
            text = self._normalize_spaces(text)

        return text

    def process(self,
                texts: List[str],
                labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Processa lista de textos para normalização.

        Args:
            texts: Lista de textos para normalizar
            labels: Labels opcionais para identificação (ex: ["whisper", "wav2vec2"])

        Returns:
            Dict com textos normalizados e estatísticas
        """
        with self.managed_processing():
            total_texts = len(texts)
            logger.info(f"[INFO] Normalizando {total_texts} textos")

            normalized_texts = []

            for i, text in enumerate(texts):
                try:
                    normalized = self.normalize_text(text)
                    normalized_texts.append(normalized)

                    if (i + 1) % 100 == 0:
                        logger.info(f"[PROGRESS] Normalizados {i + 1}/{total_texts} textos")

                except Exception as e:
                    logger.error(f"[ERRO] Falha ao normalizar texto {i}: {e}")
                    normalized_texts.append("")  # Adiciona string vazia em caso de erro

            # Estatísticas
            stats = {
                'total_processed': total_texts,
                'success_count': sum(1 for t in normalized_texts if t),
                'empty_count': sum(1 for t in normalized_texts if not t),
                'avg_length_original': sum(len(t) for t in texts) / total_texts if total_texts > 0 else 0,
                'avg_length_normalized': sum(len(t) for t in normalized_texts) / total_texts if total_texts > 0 else 0
            }

            logger.info(f"[SUMMARY] Normalização concluída:")
            logger.info(f"  [OK] Processados: {stats['total_processed']}")
            logger.info(f"  [OK] Sucesso: {stats['success_count']}")
            logger.info(f"  [INFO] Vazios: {stats['empty_count']}")
            logger.info(f"  [INFO] Tamanho médio original: {stats['avg_length_original']:.1f} chars")
            logger.info(f"  [INFO] Tamanho médio normalizado: {stats['avg_length_normalized']:.1f} chars")

            result = {
                'normalized_texts': normalized_texts,
                'stats': stats
            }

            # Adiciona labels se fornecidos
            if labels:
                result['labels'] = labels

            return result

    def unload_resources(self) -> None:
        """Não há recursos para descarregar (processamento de texto puro)."""
        logger.debug("[INFO] Text Normalizer: sem recursos para descarregar")
        pass
