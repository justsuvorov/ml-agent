"""Клиенты внешнего LLM-сервиса.

Контракт модели — как в сервисе: ``AIModel.response(query: str) -> str``.
Реализованы :class:`QwenModel` (OpenAI-совместимый ``/v1/completions``) и :class:`VskAIModel`
(``/v1/chat/completions``, reasoning-модели vLLM), а также :class:`OfflineModel` для отладки.
Можно подключить и свой класс с методом ``response`` через ``[llm] class = module:ClassName``.

Агент работает с LLM через :class:`LLMClient`: склеивает системный промпт и запрос в один
``query``, вызывает ``response`` и достаёт из текста JSON. Все запросы/ответы пишутся
в ``llm_log.jsonl`` директории запуска.

Настройки берутся из ``config.py`` проекта (имена как в сервисе)::

    llm_model_type = 'vsk'            # vsk | qwen | offline | 'module:Class'
    vsk_api_url = 'https://<host>/v1/chat/completions'
    vsk_api_key = '...'
    vsk_model_name = '...'
    vsk_max_tokens = 100000
    vsk_thinking_token_budget = 1000
    qwen_api_url = 'https://<host>/v1/completions'
    qwen_model_name = '...'
    qwen_max_tokens = 32000
    llm_timeout = 600

Секция ``[llm]`` задания может переопределить ``model_type`` (например ``offline`` для отладки).
"""
import json
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import urllib3
from loguru import logger


class LLMError(RuntimeError):
    pass


# ── Модели (интерфейс сервиса) ────────────────────────────────────────────────

class AIModel(ABC):
    @abstractmethod
    def response(self, query: str) -> str:
        pass


class RetryingHttpModel(AIModel):
    """Общая логика ретраев QwenModel / VskAIModel.

    - 503 / 504 / перегрузка / ошибки соединения / таймаут — до ``retries`` попыток с паузой ``retry_delay``;
    - пустой ответ (ValueError) — до ``empty_response_retries`` попыток с паузой ``empty_response_delay``.
    """
    name = 'LLM'
    retries = 3
    retry_delay = 5
    empty_response_retries = 3
    empty_response_delay = 1

    def __init__(self, api_url: str, model_name: str, api_key: str = '', timeout: int = 600,
                 verify_ssl: bool = False, max_tokens: int = 100000):
        self._api_url = api_url
        self._model_name = model_name
        self._api_key = api_key or ''
        self._timeout = timeout
        self._verify = verify_ssl
        self._max_tokens = max_tokens
        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def response(self, query: str) -> str:
        for attempt in range(1, self.retries + 1):
            try:
                return self._call_api(query)
            except ValueError as exc:
                if self._is_empty_response(exc) and attempt < self.empty_response_retries:
                    logger.warning(f'{self.name} не вернул текст, попытка {attempt}/{self.empty_response_retries}, '
                                   f'повтор через {self.empty_response_delay} сек')
                    time.sleep(self.empty_response_delay)
                    continue
                raise RuntimeError(f'Ошибка {self.name} API: {exc}') from exc

            except requests.Timeout as exc:
                if attempt < self.retries:
                    logger.warning(f'{self.name} таймаут, попытка {attempt}/{self.retries}, '
                                   f'повтор через {self.retry_delay} сек')
                    time.sleep(self.retry_delay)
                    continue
                raise RuntimeError(f'Ошибка {self.name} API: таймаут после {self.retries} попыток') from exc

            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code in (503, 504) and attempt < self.retries:
                    logger.warning(f'{self.name} {exc.response.status_code}, попытка {attempt}/{self.retries}, '
                                   f'повтор через {self.retry_delay} сек')
                    time.sleep(self.retry_delay)
                    continue
                raise RuntimeError(f'Ошибка {self.name} API: {exc}') from exc

            except Exception as exc:
                if self._is_overload(exc) and attempt < self.retries:
                    logger.warning(f'{self.name} перегружен, попытка {attempt}/{self.retries}, '
                                   f'повтор через {self.retry_delay} сек. Ошибка: {exc}')
                    time.sleep(self.retry_delay)
                    continue
                raise RuntimeError(f'Ошибка {self.name} API: {exc}') from exc

        raise RuntimeError(f'Сервис {self.name} недоступен. Попробуйте позже.')

    def _post(self, payload: dict) -> dict:
        headers = {'Authorization': f'Bearer {self._api_key}'} if self._api_key else {}
        resp = requests.post(self._api_url, json=payload, headers=headers,
                             timeout=self._timeout, verify=self._verify)
        resp.raise_for_status()
        return resp.json()

    @abstractmethod
    def _call_api(self, query: str) -> str:
        ...

    @staticmethod
    def _is_overload(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(s in text for s in ('503', '504', 'unavailable', 'overloaded', 'connection', 'timeout'))

    @staticmethod
    def _is_empty_response(exc: Exception) -> bool:
        text = str(exc).lower()
        return 'не вернул текст' in text or 'no response' in text


class QwenModel(RetryingHttpModel):
    """Qwen через OpenAI-совместимый completions API (``prompt`` -> ``choices[0].text``)."""
    name = 'Qwen'

    def _call_api(self, query: str) -> str:
        data = self._post({'model': self._model_name, 'prompt': query, 'max_tokens': self._max_tokens})
        text = data['choices'][0]['text']
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if not text:
            raise ValueError('Qwen не вернул текст')
        return text


class VskAIModel(RetryingHttpModel):
    """VSK AI через OpenAI-совместимый chat API (vLLM с reasoning-parser)."""
    name = 'VSK AI'
    retries = 5

    def __init__(self, *args, thinking_token_budget: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self._thinking_token_budget = thinking_token_budget

    def _call_api(self, query: str) -> str:
        return self._extract_text(self._post({
            'model': self._model_name,
            'messages': [{'role': 'user', 'content': query}],
            'thinking_token_budget': self._thinking_token_budget,
            'max_tokens': self._max_tokens,
        }))

    def _extract_text(self, data: dict) -> str:
        """Текст ответа из chat/completions.

        vLLM с reasoning-parser кладёт рассуждение в ``message.reasoning_content``, а ``content``
        бывает ``None`` (``finish_reason=length`` или парсер не отделил ответ). Агенту нужен JSON,
        поэтому в этом случае ищем JSON-объект в конце reasoning.
        """
        choices = data.get('choices') or []
        if not choices:
            raise ValueError(f'VSK AI не вернул текст: в ответе нет choices, ключи={sorted(data)}')
        choice = choices[0]
        message = choice.get('message') or {}
        finish = choice.get('finish_reason')
        usage = data.get('usage') or {}

        content = message.get('content')
        if isinstance(content, list):  # формат «частей» OpenAI
            content = ''.join(p.get('text', '') for p in content if isinstance(p, dict))
        text = re.sub(r'<think>.*?</think>', '', content or '', flags=re.DOTALL).strip()

        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        logger.info(f"VSK AI ответ: finish_reason={finish}, prompt_tokens={usage.get('prompt_tokens')}, "
                    f"completion_tokens={usage.get('completion_tokens')}, content={len(text)} симв., "
                    f"reasoning={len(reasoning)} симв.")
        if text:
            if finish == 'length':
                logger.warning(f'VSK AI: ответ обрезан по max_tokens={self._max_tokens}')
            return text

        tail = _last_json_object(reasoning)
        if tail:
            logger.warning('VSK AI: content пуст, JSON ответа найден в reasoning_content — берём его')
            return tail

        hint = ''
        if finish == 'length':
            hint = (f' — модель исчерпала max_tokens={self._max_tokens}, не дойдя до ответа; '
                    'уменьшите thinking_token_budget или увеличьте max_tokens')
        elif finish == 'content_filter':
            hint = ' — ответ заблокирован фильтром содержимого на стороне VSK AI'
        raise ValueError(f'VSK AI не вернул текст (finish_reason={finish}, reasoning={len(reasoning)} симв.){hint}')


class OfflineModel(AIModel):
    """Заглушка без внешнего сервиса — ответы формирует :class:`LLMClient` по контексту запроса."""

    def response(self, query: str) -> str:
        return '{}'


# ── Разбор ответа ─────────────────────────────────────────────────────────────

def _last_json_object(text: str) -> Optional[str]:
    end = text.rfind('}')
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            depth += {'}': 1, '{': -1}.get(text[i], 0)
            if depth == 0:
                candidate = text[i:end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    break
        end = text.rfind('}', 0, end)
    return None


def parse_json_answer(text: str) -> dict:
    """Достаёт JSON-объект из текста ответа модели (в т.ч. из ```json ...``` блока)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    fenced = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = _last_json_object(text)
        if candidate:
            return json.loads(candidate)
        raise LLMError(f'Ответ LLM не содержит JSON: {text[:500]}')


# ── Клиент агента ─────────────────────────────────────────────────────────────

class LLMClient:
    """Обёртка над AIModel: промпт -> JSON, с логированием и повтором при невалидном JSON."""

    def __init__(self, model: AIModel, log_path: Optional[Path] = None, json_retries: int = 2):
        self.model = model
        self.log_path = log_path
        self.json_retries = json_retries

    @property
    def offline(self) -> bool:
        return isinstance(self.model, OfflineModel)

    def ask_json(self, kind: str, system: str, prompt: str, context: dict = None) -> dict:
        """Отправляет запрос и возвращает JSON-ответ.

        :param kind: тип запроса (``fix_config`` / ``report``) — для логов и offline-режима.
        :param context: структурированные данные запроса; используются в offline-режиме.
        """
        if self.offline:
            answer = offline_answer(kind, context or {})
            self._log(kind, prompt, json.dumps(answer, ensure_ascii=False), answer, None)
            return answer

        query = f'{system}\n\n{prompt}'
        last_error = None
        for attempt in range(1, self.json_retries + 2):
            text, answer, error = None, None, None
            try:
                text = self.model.response(query)
                answer = parse_json_answer(text)
                return answer
            except LLMError as exc:  # модель ответила, но не JSON — просим ещё раз
                error = last_error = str(exc)
                query = (f'{system}\n\n{prompt}\n\nВАЖНО: предыдущий ответ не был валидным JSON. '
                         f'Верни только один JSON-объект по схеме.')
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                self._log(kind, query, text, answer, error)
        raise LLMError(last_error)

    def _log(self, kind, query, text, answer, error):
        if self.log_path is None:
            return
        record = {'time': datetime.now().isoformat(timespec='seconds'), 'kind': kind,
                  'model': type(self.model).__name__, 'query': query, 'raw_answer': text,
                  'answer': answer, 'error': error}
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


def offline_answer(kind: str, context: dict) -> dict:
    """Детерминированные ответы без LLM: правок не предлагает, отчёт строит по формальному сравнению."""
    if kind == 'fix_config':
        return {'diagnosis': 'offline-режим: LLM не подключена, автоматических правок нет',
                'action': 'skip', 'patches': []}
    if kind != 'report':
        return {}
    threshold = context.get('min_improvement') or 0
    key = context.get('business_key')
    business = [r for r in context.get('business') or [] if key and r.get(key) is not None]
    groups = []
    if business:  # лучшая комбинация в каждой группе по бизнес-метрике
        for group in dict.fromkeys(r['group'] for r in business):
            best = max((r for r in business if r['group'] == group), key=lambda r: r[key])
            better = best[key] > threshold
            groups.append({'group': group, 'best_combination': best['combination'] if better else 'все общие',
                           'verdict': 'better' if better else 'neutral',
                           'comment': f"{key}: {best[key]} ({best['combination']})"})
    else:  # без бизнес-метрики — по статистической метрике отдельных моделей
        for row in context.get('comparison', []):
            if row.get('group') == '__ALL__':  # общая модель — эталон, не сравнивается
                continue
            delta = row.get('improvement')
            verdict = 'unknown' if delta is None else 'better' if delta > threshold else                 'worse' if delta < -threshold else 'neutral'
            groups.append({'group': f"{row.get('block')}: {row.get('group')}", 'best_combination': '-',
                           'verdict': verdict,
                           'comment': f"{row.get('model')} {row.get('metric')}: "
                                      f"{row.get('new_value')} против {row.get('reference_value')}"})
    better = [g['group'] for g in groups if g['verdict'] == 'better']
    recommendation = ('Отдельные модели дают прирост для: ' + ', '.join(map(str, better))) if better else         'Разделение не даёт прироста относительно общих моделей — оставить общие.'
    return {'summary': 'Отчёт сформирован в offline-режиме (без LLM) по формальному сравнению метрик.',
            'groups': groups, 'recommendation': recommendation,
            'risks': ['Вывод сделан без интерпретации LLM.'], 'next_steps': []}


def _setting(config, name, default=None):
    value = getattr(config, name, default)
    if hasattr(value, 'get_secret_value'):  # pydantic SecretStr
        value = value.get_secret_value()
    return default if value is None else value


def build_llm_client(config, overrides: dict = None, log_path: Path = None) -> LLMClient:
    """Создаёт клиента LLM по настройкам ``config.py`` (+ переопределения из секции ``[llm]`` задания)."""
    overrides = overrides or {}
    model_type = str(overrides.get('model_type') or _setting(config, 'llm_model_type', 'offline'))
    timeout = int(_setting(config, 'llm_timeout', 600))
    verify_ssl = bool(_setting(config, 'llm_verify_ssl', False))

    if model_type == 'offline':
        model = OfflineModel()
    elif model_type in ('vsk', 'qwen'):
        url = _setting(config, f'{model_type}_api_url')
        if not url:
            raise ValueError(f'В config.py не задан {model_type}_api_url')
        common = dict(api_url=url, model_name=_setting(config, f'{model_type}_model_name'),
                      api_key=_setting(config, f'{model_type}_api_key', ''), timeout=timeout,
                      verify_ssl=verify_ssl, max_tokens=int(_setting(config, f'{model_type}_max_tokens', 100000)))
        if model_type == 'vsk':
            model = VskAIModel(**common, thinking_token_budget=int(_setting(config, 'vsk_thinking_token_budget', 1000)))
        else:
            model = QwenModel(**common)
    else:  # свой класс с методом response(query) -> str и конструктором без аргументов
        from ml_agent.utils import import_object
        model = import_object(model_type)()
    logger.info(f'LLM||модель: {type(model).__name__}')
    return LLMClient(model, log_path=log_path, json_retries=int(_setting(config, 'llm_json_retries', 2)))
