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
import json
import re
import logging
import random
from mmagent.mllm_gpt import generate_messages, get_response as get_response_with_retry
from mmagent.llm import get_multiple_embeddings as parallel_get_embedding, get_embedding as get_embedding_with_retry
from .prompts import *
from .memory_processing import parse_video_caption
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from mmagent import processing_config
MAX_RETRIES = 20
# Configure logging
logger = logging.getLogger(__name__)

def translate(video_graph, memories):
    new_memories = []
    # Ensure attributes exist for backward compatibility
    reverse_mappings = getattr(video_graph, 'reverse_character_mappings', {})
    
    for memory in memories:
        if memory.lower().startswith("equivalence: "):
            continue
        new_memory = memory
        entities = parse_video_caption(video_graph, memory)
        entities = list(set(entities))
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if reverse_mappings and entity_str in reverse_mappings:
                new_memory = new_memory.replace(f"<{entity[0]}_{entity[1]}>", f"<{reverse_mappings[entity_str]}>")
                # Also try replacing without brackets if the LLM output is inconsistent
                new_memory = new_memory.replace(entity_str, reverse_mappings[entity_str])
        new_memories.append(new_memory)
    return new_memories

def back_translate(video_graph, queries):
    translated_queries = []
    # Ensure attributes exist for backward compatibility
    mappings = getattr(video_graph, 'character_mappings', {})
    
    for query in queries:
        entities = parse_video_caption(video_graph, query)
        entities = list(set(entities))
        to_be_translated = [query]
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if mappings and entity_str in mappings:
                char_variants = mappings[entity_str]
                
                # Create new queries for each mapping
                new_queries = []
                for variant in char_variants:
                    for partially_translated in to_be_translated:
                        # Try replacing both with and without brackets
                        new_query = partially_translated.replace(f"<{entity_str}>", f"<{variant}>")
                        new_query = new_query.replace(entity_str, variant)
                        new_queries.append(new_query)
                
                # Update translated_query with all variants
                to_be_translated = new_queries
                
        # Add all variants of the translated query
        translated_queries.extend(to_be_translated)
    return translated_queries

# retrieve by clip
def retrieve_from_videograph(video_graph, query, topk=5, mode='max', threshold=0, before_clip=None):
    # find all CLIP_x in query
    pattern = r"CLIP_(\d+)"
    matches = re.finditer(pattern, query)
    top_clips = []
    for match in matches:
        try:
            clip_id = int(match.group(1))
            top_clips.append(clip_id)
        except ValueError:
            continue
    
    queries = back_translate(video_graph, [query])
    if len(queries) > 100:
        logger.error(f"Anomaly detected from query: {query}, randomly sample 100 translatedqueries")
        queries = random.sample(queries, 100)
    
    related_nodes = get_related_nodes(video_graph, query)

    model = "text-embedding-3-large"
    query_embeddings = parallel_get_embedding(queries)

    full_clip_scores = {}
    clip_scores = {}

    if mode not in ['sum', 'max', 'mean']:
        raise ValueError(f"Unknown mode: {mode}")

    # calculate scores for each node
    nodes = video_graph.search_text_nodes(query_embeddings, related_nodes, mode='max')
    
    # collect node scores for each clip
    for node_id, node_score in nodes:
        clip_id = video_graph.nodes[node_id].metadata['timestamp']
        if clip_id not in full_clip_scores:
            full_clip_scores[clip_id] = []
        full_clip_scores[clip_id].append(node_score)

    # calculate scores for each clip
    for clip_id, scores in full_clip_scores.items():
        if mode == 'sum':
            clip_score = sum(scores)
        elif mode == 'max':
            clip_score = max(scores)
        elif mode == 'mean':
            clip_score = np.mean(scores)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        clip_scores[clip_id] = clip_score

    # sort clips by score
    sorted_clips = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)
    # filter out clips that have 0 score and get top k clips
    if before_clip is not None:
        top_clips = [clip_id for clip_id, score in sorted_clips if score >= threshold and clip_id <= before_clip][:topk]
    else:
        top_clips = [clip_id for clip_id, score in sorted_clips if score >= threshold][:topk]
    return top_clips, clip_scores, nodes

def get_related_nodes(video_graph, query):
    related_nodes = []
    entities = parse_video_caption(video_graph, query)
    
    # Ensure attributes exist for backward compatibility
    mappings = getattr(video_graph, 'character_mappings', {})
    reverse_mappings = getattr(video_graph, 'reverse_character_mappings', {})
    
    for entity in entities:
        type = entity[0]
        node_id = entity[1]
        entity_tag = f"{type}_{node_id}"
        
        # Check if the entity is part of a character group
        if mappings and entity_tag in mappings:
            # Map character name back to underlying face/voice node IDs
            char_nodes = mappings[entity_tag]
            for node_tag in char_nodes:
                try:
                    # node_tag is e.g. "face_12"
                    related_nodes.append(int(node_tag.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        elif reverse_mappings and entity_tag in reverse_mappings:
            # If it's a specific node ID, find its character representative and then all siblings
            char_rep = reverse_mappings[entity_tag]
            if mappings and char_rep in mappings:
                for node_tag in mappings[char_rep]:
                    try:
                        related_nodes.append(int(node_tag.split("_")[1]))
                    except (IndexError, ValueError):
                        continue
        else:
            # If no complex mappings, just use the node_id directly if it's numeric
            try:
                related_nodes.append(int(node_id))
            except ValueError:
                continue
                
    return list(set(related_nodes))

def search(video_graph, query, current_clips, topk=5, mode='max', threshold=0, mem_wise=False, before_clip=None, episodic_only=False):
    top_clips, clip_scores, nodes = retrieve_from_videograph(video_graph, query, topk, mode, threshold, before_clip)
    
    if mem_wise:
        new_memories = {}
        top_nodes_num = 0
        # fetch top nodes
        for top_node, _ in nodes:
            clip_id = video_graph.nodes[top_node].metadata['timestamp']
            if before_clip is not None and clip_id > before_clip:
                continue
            if clip_id not in new_memories:
                new_memories[clip_id] = []
            new_ = translate(video_graph, video_graph.nodes[top_node].metadata['contents'])
            new_memories[clip_id].extend(new_)
            top_nodes_num += len(new_)
            if top_nodes_num >= topk:
                break
        # sort related_memories by timestamp
        new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
        new_memories = {f"CLIP_{k}": v for k, v in new_memories.items() if len(v) > 0}
        return new_memories, current_clips, clip_scores
    
    new_clips = [top_clip for top_clip in top_clips if top_clip not in current_clips]
    new_memories = {}
    current_clips.extend(new_clips)
    
    for new_clip in new_clips:
        if new_clip not in video_graph.text_nodes_by_clip:
            new_memories[new_clip] = [f"CLIP_{new_clip} not found in memory bank, please search for other information"]
        else:
            related_nodes = video_graph.text_nodes_by_clip[new_clip]
            new_memories[new_clip] = translate(video_graph, [video_graph.nodes[node_id].metadata['contents'][0] for node_id in related_nodes if (not episodic_only or video_graph.nodes[node_id].type != "semantic")])
                        
    # sort related_memories by timestamp
    new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
    new_memories = {f"CLIP_{k}": v for k, v in new_memories.items()}
    
    return new_memories, current_clips, clip_scores

def calculate_similarity(mem, query, related_nodes):
    related_nodes_embeddings = np.array([mem.nodes[node_id].embeddings[0] for node_id in related_nodes])
    query_embedding = np.array(get_embedding_with_retry(query)).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, related_nodes_embeddings)[0]
    return similarities.tolist()

def retrieve_all_episodic_memories(video_graph):
    episodic_memories = {}
    for node_id in video_graph.text_nodes:
        if video_graph.nodes[node_id].type == "episodic":
            clips_id = f"CLIP_{video_graph.nodes[node_id].metadata['timestamp']}"
            if clips_id not in episodic_memories:
                episodic_memories[clips_id] = []
            episodic_memories[clips_id].extend(video_graph.nodes[node_id].metadata["contents"])
    return episodic_memories

def retrieve_all_semantic_memories(video_graph):
    semantic_memories = {}
    for node_id in video_graph.text_nodes:
        if video_graph.nodes[node_id].type == "semantic":
            clips_id = f"CLIP_{video_graph.nodes[node_id].metadata['timestamp']}"
            if clips_id not in semantic_memories:
                semantic_memories[clips_id] = []
            semantic_memories[clips_id].extend(video_graph.nodes[node_id].metadata["contents"])
    return semantic_memories
