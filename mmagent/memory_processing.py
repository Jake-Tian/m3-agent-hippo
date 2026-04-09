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
import base64
import json
import logging
from io import BytesIO
import re

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from mmagent.mllm_gpt import generate_messages, get_response as get_response_with_retry

from mmagent import processing_config
logging_level = processing_config["logging"]
MAX_RETRIES = 20
# Configure logging
logger = logging.getLogger(__name__)
    

def parse_video_caption(video_graph, video_caption):
    # video_caption is a string like this: <char_1> xxx <char_2> xxx or <Alice> xxx
    # extract all the elements wrapped by < and >
    def verify_entity(video_graph, entity_str):
        try:
            if "_" in entity_str:
                node_type, node_id = entity_str.split("_", 1)
                node_type = node_type.strip().lower()
                # allow 'char' as alias for 'character'
                if node_type == "char": node_type = "character"
                if node_type in ["face", "voice", "character"]:
                    return (node_type, node_id)
            
            # Fallback for names like <Alice> or <character1>
            # Treat as character node
            return ("character", entity_str)
        except Exception as e:
            return None

    pattern = r'<([^<>]*)>'
    entity_strs = re.findall(pattern, video_caption)
    entities = [verify_entity(video_graph, entity_str) for entity_str in entity_strs]
    entities = [entity for entity in entities if entity is not None]
    return entities

from mmagent.llm import get_multiple_embeddings

def process_memories(video_graph, memory_contents, clip_id, type='episodic', embeddings=None):
    def get_memory_embeddings(memory_contents):
        # calculate the embedding for each memory
        embeddings = get_multiple_embeddings(memory_contents)
        return embeddings

    def insert_memory(video_graph, memory, type='episodic'):
        # create a new text node for each memory
        new_node_id = video_graph.add_text_node(memory, clip_id, type)
        entities = parse_video_caption(video_graph, memory['contents'][0])
        for entity in entities:
            video_graph.add_character_node(entity[1])
            video_graph.add_edge(new_node_id, entity[1])

    def update_video_graph(video_graph, memories, type='episodic'):
        # append all episodic memories to the graph
        if type == 'episodic':
            # create a new text node for each memory
            for memory in memories:
                insert_memory(video_graph, memory, type)
        # semantic memories can be used to update the existing text nodes, or create new text nodes
        elif type == 'semantic':
            for memory in memories:
                entities = parse_video_caption(video_graph, memory['contents'][0])

                if len(entities) == 0:
                    insert_memory(video_graph, memory, type)
                    continue
                
                # update the existing text node for each memory, if needed
                positive_threshold = 0.85
                negative_threshold = 0
                
                # get all (possible) related nodes            
                node_id = entities[0][1]
                related_nodes = video_graph.get_connected_nodes(node_id, type=['semantic'])
                
                if not related_nodes:
                    insert_memory(video_graph, memory, type)
                    continue

                # Vectorized similarity check
                related_embeddings = np.array([video_graph.nodes[nid].embeddings[0] for nid in related_nodes])
                memory_emb = np.array(memory['embeddings'][0])
                
                # cosine similarity = (A . B) / (||A|| * ||B||)
                # Ensure correct shapes for broadcasting/matmul
                norms = np.linalg.norm(related_embeddings, axis=1) * np.linalg.norm(memory_emb)
                similarities = np.dot(related_embeddings, memory_emb) / np.where(norms == 0, 1, norms)
                
                create_new_node = True
                for i, node_id in enumerate(related_nodes):
                    # see if the memory entities are a subset of the existing node entities
                    related_node_entities = parse_video_caption(video_graph, video_graph.nodes[node_id].metadata['contents'][0])
                    if all(entity in related_node_entities for entity in entities):
                        similarity = similarities[i]
                        if similarity > positive_threshold:
                            video_graph.reinforce_node(node_id)
                            create_new_node = False
                        elif similarity < negative_threshold:
                            video_graph.weaken_node(node_id)
                            create_new_node = False
                
                if create_new_node:
                    insert_memory(video_graph, memory, type)
    
    if embeddings is None:
        memories_embeddings = get_memory_embeddings(memory_contents)
    else:
        memories_embeddings = embeddings

    memories = []
    for memory, embedding in zip(memory_contents, memories_embeddings):
        memories.append({
            'contents': [memory],
            'embeddings': [embedding]
        })

    update_video_graph(video_graph, memories, type)
