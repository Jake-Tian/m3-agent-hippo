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
import os
import re
import numpy as np
import pickle
import logging

# Configure logging
logger = logging.getLogger(__name__)

def load_video_graph(video_graph_path):
    """Load video graph from pickle file.
    """
    if not os.path.exists(video_graph_path):
        logger.warning(f"Video graph not found at {video_graph_path}")
        return None
    with open(video_graph_path, "rb") as f:  
        logger.info(f"Loading video graph from {video_graph_path}")
        return pickle.load(f)

def strip_code_fences(text) -> str:
    """
    Remove surrounding Markdown code fences (``` or ```json) from a string.
    Preserves inner content exactly.
    """
    if text is None:
        return ""
    if isinstance(text, tuple):
        text = text[0] if text else ""
    if not isinstance(text, str):
        return str(text)

    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the first fence line
        lines = stripped.splitlines()
        if lines:
            # Remove the opening fence (could be ``` or ```json)
            lines = lines[1:]
        # If the last line is a closing fence, drop it
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped

def cosine_similarity_embed(vec1, vec2):
    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (n1 * n2))

from mmagent.llm import get_embedding

def merge_character_appearances(characters_appearance, appearance_dict, similarity_threshold=0.85):
    """
    Merge/update character appearances into appearance_dict.
    """
    equivalence_list = []
    for character in characters_appearance:
        # old character
        if character.name in appearance_dict:
            if appearance_dict[character.name][0] != character.appearance:
                appearance_dict[character.name][0] = character.appearance
                appearance_dict[character.name][1] = get_embedding(character.appearance)
            continue

        embedding = get_embedding(character.appearance)
        best_similarity = 0.0
        best_match = None
        # new character
        for char_name, char_appearance in appearance_dict.items():
            name_inner = char_name.strip('<>')
            import re
            is_unknown = bool(re.match(r'^(male|female|character)_\d+$', name_inner, re.IGNORECASE) or name_inner.islower())
            if is_unknown:
                similarity = cosine_similarity_embed(embedding, char_appearance[1])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = char_name

        if best_similarity > similarity_threshold:
            # unknown → unknown
            name_inner_new = character.name.strip('<>')
            is_unknown_new = bool(re.match(r'^(male|female|character)_\d+$', name_inner_new, re.IGNORECASE) or name_inner_new.islower())
            if is_unknown_new:
                # Just keep the existing one (best_match) and rename character.name to it
                if character.name < best_match:
                    appearance_dict.pop(best_match, None)
                    appearance_dict[character.name] = [character.appearance, embedding]
                    equivalence_list.append([best_match, character.name])
                else:
                    appearance_dict[best_match] = [character.appearance, embedding]
                    equivalence_list.append([character.name, best_match])
            # named character → <character_X>
            else:
                appearance_dict.pop(best_match, None)
                appearance_dict[character.name] = [character.appearance, embedding]
                equivalence_list.append([best_match, character.name])
        else:
            appearance_dict[character.name] = [character.appearance, embedding]

    return equivalence_list
