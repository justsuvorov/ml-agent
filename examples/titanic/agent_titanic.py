import os
import sys
from pathlib import Path

from loguru import logger

from ml_agent.chat import OpenAiChat
from ml_agent.checking import FeatureCheck
from ml_agent.fitting import Fitting
from ml_agent.loop import Loop
from ml_agent.outbox import LocalOutbox
from ml_agent.tools import Tools
from ml_agent.workspace import Folder

logger.remove()
logger.add(sys.stderr, level="WARNING")
sys.stdout.buffer.write(
    Loop(
        OpenAiChat(os.environ["LLM_URL"], os.environ["LLM_KEY"], os.environ["LLM_MODEL"]),
        Tools(
            {
                "fit_models": Fitting(
                    Folder(Path(__file__).resolve().parent),
                    LocalOutbox(Path(__file__).resolve().parent),
                ),
                "check_features": FeatureCheck(
                    Folder(Path(__file__).resolve().parent),
                    LocalOutbox(Path(__file__).resolve().parent),
                ),
            },
        ),
        12,
    ).answer(
        " ".join(sys.argv[1:])
        or "Проверь фичи PCLASS, EMBARKED, CABIN и NAME для модели first. "
        "Конфиг моделей — configs/titanic.json, конфиг AutoML — configs/automl.json.",
    ).encode("utf-8")
    + b"\n",
)
