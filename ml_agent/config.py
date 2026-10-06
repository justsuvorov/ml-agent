"""Конфиг проекта (передаётся как external_config, как в main_fit).

Если в проекте уже есть config.py — достаточно добавить в него блок «ML-агент: LLM».
Без .env: все значения задаются здесь.
"""
from pathlib import Path

# ── mldataworker ─────────────────────────────────────────────────────────────
work_type_fit = 'CPU'
work_type_hptune = 'CPU'

base_path = Path(__file__).resolve().parent
results_path = base_path / 'results'
prod_path = base_path
prod_models_folder = 'prod_models'
prod_models_path = base_path / prod_models_folder

mlflow_tracking_uri = 'http://localhost:5000'
mlflow_experiment = 'casco_agent'

connection_params = 'postgresql+psycopg2://user:password@host:5432/db'

# ── Почта (EMail из mldataworker) ─────────────────────────────────────────────
email_smtp_server = 'smtp.example.ru'
email_port = 25
email_sender = 'ml-agent@example.ru'
email_login = ''
email_pass = ''
email_receivers = ['analyst@example.ru']

# ── ML-агент: LLM ─────────────────────────────────────────────────────────────
# vsk | qwen | offline (без LLM, для отладки) | 'module:Class' (свой класс с response(query) -> str)
llm_model_type = 'vsk'
llm_timeout = 600
llm_verify_ssl = False
llm_json_retries = 2  # повторный запрос, если модель вернула не JSON

# VSK AI — OpenAI-совместимый /v1/chat/completions
vsk_api_url = 'https://<vsk-ai-host>/v1/chat/completions'
vsk_api_key = '<key>'
vsk_model_name = '<model-name>'
vsk_max_tokens = 100000
vsk_thinking_token_budget = 1000

# Qwen — OpenAI-совместимый /v1/completions
qwen_api_url = 'https://<qwen-host>/v1/completions'
qwen_model_name = '<model-name>'
qwen_max_tokens = 32000
