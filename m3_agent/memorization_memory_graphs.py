import os
import json
import logging
import argparse
import glob
import pickle
import sys
from pathlib import Path

from mmagent.videograph import VideoGraph
from mmagent.mllm_gpt import generate_messages, get_response
from mmagent.prompts import prompt_generate_full_memory_hippo
from mmagent.output_structure import FullMemoryFormat
from mmagent.general import merge_character_appearances
from mmagent.memory_processing import process_memories, parse_video_caption

logger = logging.getLogger(__name__)

def extract_and_add_characters(video_graph, text_list):
    for text in text_list:
        entities = parse_video_caption(video_graph, text)
        for entity in entities:
            if entity[0] == "character":
                video_graph.add_character_node(entity[1])

from mmagent.llm import get_multiple_embeddings

def streaming_process_video(video_graph, video_name, frames_dir, mem_path):
    # Get sorted image folders (clips)
    image_folders = sorted(
        [str(folder) for folder in Path(frames_dir).iterdir() if folder.is_dir()],
        key=lambda x: int(Path(x).name) if Path(x).name.isdigit() else x
    )

    if not image_folders:
        logger.warning(f"No frame clip folders found in {frames_dir}")
        return

    appearance_dict = dict()
    total_tokens = 0
    
    for clip_path in image_folders:
        clip_id = int(os.path.basename(clip_path))
        
        # Get frame images from the directory
        current_images = sorted(
            glob.glob(f"{clip_path}/*.jpg"),
            key=lambda p: int(os.path.basename(p).split('.')[0]) if os.path.basename(p).split('.')[0].isdigit() else p,
        )
        if not current_images:
            continue
            
        # Sample frames to reduce payload and speed up MLLM calls
        # Sample every 5th frame, but at most 10 frames
        sampled_images = current_images[::5][:10]
        # Ensure the last frame is included if not already
        if current_images[-1] not in sampled_images:
            sampled_images.append(current_images[-1])

        appearance_prompt_dict = {name: value[0] for name, value in appearance_dict.items()}
        prompt = "Character appearance from previous videos: \n" + json.dumps(appearance_prompt_dict) + "\n"
        
        current_main_character = video_graph.get_main_character()
        if current_main_character:
            prompt += f"The main character of this video is identified as: {current_main_character}\n"
        
        prompt += prompt_generate_full_memory_hippo
        
        messages = generate_messages(sampled_images, prompt)
        try:
            response, tokens = get_response(messages, FullMemoryFormat)
            total_tokens += tokens
        except Exception as e:
            try: 
                response, tokens = get_response(messages, FullMemoryFormat)
                total_tokens += tokens
            except Exception as e:
                logger.error(f"MLLM call failed for clip {clip_id}: {e}")
                continue
            
        episodic_memory = response.episodic_memory
        semantic_memory = response.semantic_memory
        characters_appearance = response.characters_appearance
        main_character = response.main_character

        if main_character:
            video_graph.set_main_character(main_character)

        # Update character appearance and mapping
        merge_character_appearances(characters_appearance, appearance_dict)

        # Batch embedding calls for both episodic and semantic memories
        all_memories = episodic_memory + semantic_memory
        if all_memories:
            all_embeddings = get_multiple_embeddings(all_memories)
            
            # Split embeddings back
            episodic_embeddings = all_embeddings[:len(episodic_memory)]
            semantic_embeddings = all_embeddings[len(episodic_memory):]

            # Store memories in the VideoGraph
            extract_and_add_characters(video_graph, episodic_memory)
            if episodic_memory:
                process_memories(video_graph, episodic_memory, clip_id, type="episodic", embeddings=episodic_embeddings)
                
            extract_and_add_characters(video_graph, semantic_memory)
            if semantic_memory:
                process_memories(video_graph, semantic_memory, clip_id, type="semantic", embeddings=semantic_embeddings)
        
        print(f"Processed clip {clip_id}")


    video_graph.token_consumption = total_tokens
    os.makedirs(os.path.dirname(mem_path), exist_ok=True)
    with open(mem_path, "wb") as f:
        pickle.dump(video_graph, f)
    logger.info(f"Saved video graph to {mem_path}")

    # Save token consumption to JSON
    token_json_dir = "data/memorization"
    os.makedirs(token_json_dir, exist_ok=True)
    token_json_path = os.path.join(token_json_dir, f"{video_name}.json")
    token_summary = {"total": total_tokens}
    with open(token_json_path, "w") as f:
        json.dump({"memory_token_summaries": token_summary}, f, indent=2)
    
    # Update aggregated tokens file
    agg_tokens_path = "data/memorization_tokens.json"
    agg_tokens = {}
    if os.path.exists(agg_tokens_path):
        try:
            with open(agg_tokens_path, "r") as f:
                agg_tokens = json.load(f)
        except:
            pass
    agg_tokens[video_name] = token_summary
    with open(agg_tokens_path, "w") as f:
        json.dump(agg_tokens, f, indent=2)
    logger.info(f"Saved token consumption to {token_json_path} and updated {agg_tokens_path}")

if __name__ == "__main__":

# Example usage: python -m m3_agent.memorization_memory_graphs --video_names _A9R3dlxh_o
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_names", type=str, nargs='+', help="Specific video names to process")
    parser.add_argument("--frames_dir", type=str, default="data/frames", help="Directory containing frames")
    parser.add_argument("--save_dir", type=str, default="data/graphs", help="Directory to save graphs")
    args = parser.parse_args()
    
    if args.video_names:
        available_videos = args.video_names
    else:
        # Process all videos in frames_dir
        if os.path.exists(args.frames_dir):
            available_videos = sorted([d for d in os.listdir(args.frames_dir) if os.path.isdir(os.path.join(args.frames_dir, d))])
        else:
            logger.error(f"Frames directory {args.frames_dir} not found")
            sys.exit(1)

    for video_name in available_videos:
        mem_path = os.path.join(args.save_dir, f"{video_name}.pkl")
        if os.path.exists(mem_path):
            logger.info(f"Skipping {video_name}, already processed.")
            continue
            
        logger.info(f"Processing video: {video_name}")
        frames_dir = os.path.join(args.frames_dir, video_name)
        video_graph = VideoGraph() 
        streaming_process_video(video_graph, video_name, frames_dir, mem_path)
