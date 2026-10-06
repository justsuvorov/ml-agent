"""Безопасное редактирование JSON-конфигов моделей mldataworker.

Агент никогда не правит исходный конфиг: работает с копией в директории эксперимента,
а каждое изменение фиксирует в ``changes`` (попадает в отчёт).

Поддерживаемые операции (их же может вернуть LLM):

* ``{"op": "remove_feature", "model": "frequency" | "*", "feature": "X"}`` — удалить фичу/колонку
  из модели со всеми ссылками (features, treatment_dict, relative_features, intersections,
  cat_features_catboost, column_weight);
* ``{"op": "set", "path": "models_configs/frequency/params_catboost/depth", "value": 6}``;
* ``{"op": "delete", "path": "models_configs/frequency/data_filter_condition"}``.

Путь — сегменты через ``/``; элемент списка адресуется индексом или значением поля ``name``.
"""
import copy
import json
from pathlib import Path
from typing import Any, List

PROTECTED_PATHS = ('data_config/local_name_source', 'data_config/source', 'data_config/separation')
MAX_PATCHES_PER_ATTEMPT = 20


class PatchError(ValueError):
    pass


class ModelsConfigEditor:
    def __init__(self, data: dict):
        self.data = copy.deepcopy(data)
        self.changes: List[str] = []

    @classmethod
    def from_file(cls, path) -> 'ModelsConfigEditor':
        with open(path, encoding='utf-8') as f:
            return cls(json.load(f))

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        return path

    # ---------- чтение ----------
    @property
    def models(self) -> List[dict]:
        return self.data.get('models_configs', [])

    def model_names(self) -> List[str]:
        return [m.get('name') for m in self.models]

    def feature_names(self, model_name: str = None) -> List[str]:
        names = []
        for model in self._select_models(model_name or '*'):
            names += [f['name'] for f in model.get('features') or []]
            names += [r['name'] for r in model.get('relative_features') or []]
        return list(dict.fromkeys(names))

    def referenced_columns(self) -> set:
        """Все колонки датасета, на которые ссылается конфиг (для проверки до запуска)."""
        cols = set()
        for model in self.models:
            relative = {r['name'] for r in model.get('relative_features') or []}
            cols |= {f['name'] for f in model.get('features') or []} - relative
            for r in model.get('relative_features') or []:
                cols |= {r['numerator'], r['denominator']}
            for key in ('column_target', 'column_exposure', 'column_weight'):
                if model.get(key):
                    cols.add(model[key])
        return cols

    def summary(self) -> dict:
        """Компактное описание конфига для промпта LLM."""
        return {'data_config': {k: v for k, v in self.data.get('data_config', {}).items()
                                if k in ('source', 'separation', 'extra_columns')},
                'models': [{'name': m.get('name'), 'objective': m.get('objective'),
                            'wrapper': m.get('wrapper'), 'column_target': m.get('column_target'),
                            'column_exposure': m.get('column_exposure'),
                            'features': [f['name'] for f in m.get('features') or []],
                            'relative_features': m.get('relative_features'),
                            'intersections': m.get('intersections'),
                            'params_catboost': m.get('params_catboost'),
                            'params_glm': m.get('params_glm'),
                            'data_filter_condition': m.get('data_filter_condition')}
                           for m in self.models]}

    # ---------- изменения ----------
    def apply(self, patches: List[dict]) -> List[str]:
        if len(patches) > MAX_PATCHES_PER_ATTEMPT:
            raise PatchError(f'Слишком много правок за одну попытку: {len(patches)}')
        applied = []
        for patch in patches:
            op = patch.get('op')
            if op == 'remove_feature':
                applied.append(self.remove_feature(patch['feature'], patch.get('model', '*')))
            elif op == 'set':
                applied.append(self.set(patch['path'], patch.get('value')))
            elif op == 'delete':
                applied.append(self.delete(patch['path']))
            else:
                raise PatchError(f'Неподдерживаемая операция: {patch}')
        return applied

    def remove_feature(self, feature: str, model_name: str = '*') -> str:
        touched = []
        for model in self._select_models(model_name):
            if self._remove_from_model(model, feature):
                touched.append(model['name'])
        if not touched:
            raise PatchError(f'Фича/колонка {feature} не найдена в моделях {model_name}')
        return self._record(f'remove_feature {feature} из моделей {touched}')

    def set(self, path: str, value: Any) -> str:
        parent, key = self._resolve_parent(path)
        old = parent[key] if (isinstance(parent, dict) and key in parent) or \
                             (isinstance(parent, list) and key < len(parent)) else None
        parent[key] = value
        return self._record(f'set {path}: {json.dumps(old, ensure_ascii=False)} -> '
                            f'{json.dumps(value, ensure_ascii=False)}')

    def delete(self, path: str) -> str:
        parent, key = self._resolve_parent(path)
        try:
            del parent[key]
        except (KeyError, IndexError):
            raise PatchError(f'Путь не найден: {path}')
        return self._record(f'delete {path}')

    # ---------- служебное ----------
    def _record(self, text: str) -> str:
        self.changes.append(text)
        return text

    def _select_models(self, model_name: str) -> List[dict]:
        if model_name in (None, '*', 'all'):
            return self.models
        models = [m for m in self.models if m.get('name') == model_name]
        if not models:
            raise PatchError(f'Модель {model_name} не найдена, есть: {self.model_names()}')
        return models

    @staticmethod
    def _remove_from_model(model: dict, column: str) -> bool:
        removed = False
        if model.get('column_weight') == column:  # без колонки весов модель обучается с весом 1
            model['column_weight'] = None
            removed = True
        dependent = {column}
        relative = []
        for r in model.get('relative_features') or []:
            if column in (r['name'], r['numerator'], r['denominator']):
                dependent.add(r['name'])
                removed = True
            else:
                relative.append(r)
        if model.get('relative_features') is not None:
            model['relative_features'] = relative

        features = [f for f in model.get('features') or [] if f['name'] not in dependent]
        if len(features) != len(model.get('features') or []):
            removed = True
            model['features'] = features

        if model.get('treatment_dict'):
            for name in dependent & set(model['treatment_dict']):
                del model['treatment_dict'][name]
        if model.get('cat_features_catboost'):
            model['cat_features_catboost'] = [c for c in model['cat_features_catboost'] if c not in dependent]
        if model.get('intersections'):
            model['intersections'] = [i for i in model['intersections']
                                      if not dependent & set(i.get('features_to_intersect', []))]
        return removed

    def _resolve_parent(self, path: str):
        norm = path.strip('/')
        if any(norm == p or norm.startswith(p + '/') for p in PROTECTED_PATHS):
            raise PatchError(f'Путь {path} управляется агентом и не может быть изменён')
        parts = norm.split('/')
        node = self.data
        for part in parts[:-1]:
            node = self._step(node, part, path)
        last = parts[-1]
        if isinstance(node, list):
            last = self._list_index(node, last, path)
        elif not isinstance(node, dict):
            raise PatchError(f'Путь {path} указывает внутрь скалярного значения')
        return node, last

    def _step(self, node, part, path):
        if isinstance(node, dict):
            if part not in node or node[part] is None:
                node[part] = {}
            return node[part]
        if isinstance(node, list):
            return node[self._list_index(node, part, path)]
        raise PatchError(f'Путь {path} указывает внутрь скалярного значения')

    @staticmethod
    def _list_index(node: list, part: str, path: str) -> int:
        if part.isdigit():
            if int(part) >= len(node):
                raise PatchError(f'Индекс {part} вне диапазона в {path}')
            return int(part)
        for i, item in enumerate(node):
            if isinstance(item, dict) and item.get('name') == part:
                return i
        raise PatchError(f'Элемент {part} не найден в {path}')
