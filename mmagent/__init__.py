# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import json
import os

# Hardcoded processing config
processing_config = {
    "logging": "INFO",
    "model": "gpt-5-mini",
    "log_dir": "logs"
}
logging_level = processing_config["logging"]
model = processing_config["model"]

# Configure root logger
os.makedirs(processing_config["log_dir"], exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG if logging_level == "DETAIL" else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(processing_config["log_dir"], "mmagent.log"))
    ]
)

# Disable third-party library logging
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

from . import retrieve
from . import memory_processing
from . import prompts
from . import videograph
from . import general

__all__ = [
    "retrieve",
    "memory_processing",
    "prompts",
    "videograph",
    "general",
]
