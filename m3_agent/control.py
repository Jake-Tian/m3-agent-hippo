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
import re
import os
import sys
import json
import time
import argparse
import mmagent.videograph
from mmagent.retrieve import search
from mmagent.general import load_video_graph
from mmagent.mllm_gpt import generate_messages
from mmagent.prompts import prompt_agent_verify_answer_referencing, prompt_generate_action_with_plan_structured
from mmagent.llm import generate_text_response
from mmagent.output_structure import ActionOutput

sys.modules["videograph"] = mmagent.videograph

# Hardcoded defaults
processing_config = {
    "logging": "INFO",
    "model": "gpt-5-mini",
    "batch_size": 64,
    "total_round": 5,
    "topk": 50
}

def eval_answer(question, predict, ground_truth):
    if predict == "":
        return False, 0
    try:
        prompt = f"Question: {question}\nGround Truth: {ground_truth}\nAgent Answer: {predict}\n\n{prompt_agent_verify_answer_referencing}"
        res, tokens = generate_text_response(prompt)
        result = res.lower()
    except Exception as e:
        print(f"Error verifying qa: {question} | {str(e)}")
        return False, 0
    return (True if "yes" in result else False), tokens

def process_action(data, response_json):
    if not data["finish"]:
        before_clip = data.get("before_clip", None)
        
        action = response_json.action
        content = response_json.content
        
        if action == "Answer":
            data["response"] = content
            data["finish"] = True
        else:
            new_memories = {}
            if content:
                mem_node = load_video_graph(data["mem_path"])
                if mem_node is None:
                    search_result = f"Error: Memory graph not found at {data['mem_path']}"
                else:
                    if before_clip is not None:
                        if hasattr(mem_node, 'truncate_memory_by_clip'):
                            mem_node.truncate_memory_by_clip(before_clip, False)
                    
                    memories, currenr_clips, _ = search(mem_node, content, data["currenr_clips"], threshold=0.2, topk=processing_config["topk"], before_clip=before_clip)
                    data["currenr_clips"] = currenr_clips
                    new_memories.update(memories)
                    search_result = "Searched knowledge: " + json.dumps(new_memories, ensure_ascii=False).encode("utf-8", "ignore").decode("utf-8")
            else:
                search_result = "Searched knowledge: {}"
                
            if len(new_memories) == 0 and content:
                search_result += "\n(The search result is empty. Please try searching from another perspective.)"
            data["conversations"].append({"role": "user", "content": search_result})
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_names", type=str, nargs='+', help="Specific video names to process")
    parser.add_argument("--questions_file", type=str, default="data/questions.jsonl")
    parser.add_argument("--mem_dir", type=str, default="data/graphs")
    parser.add_argument("--output_dir", type=str, default="data/results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all questions
    questions_data = {}
    if os.path.exists(args.questions_file):
        with open(args.questions_file, "r") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                vid = item.get("video_id")
                if vid not in questions_data:
                    questions_data[vid] = []
                questions_data[vid].append(item)
    else:
        print(f"Questions file {args.questions_file} not found")
        sys.exit(1)

    if args.video_names:
        videos_to_process = args.video_names
    else:
        videos_to_process = sorted(questions_data.keys())

    videos_to_process = videos_to_process[:2]

    for video_name in videos_to_process:
        print(f"================ processing video: {video_name} ================")
        video_questions = questions_data.get(video_name, [])
        if not video_questions:
            print(f"No questions found for video {video_name}")
            continue

        output_path = os.path.join(args.output_dir, f"{video_name}.jsonl")
        mem_path = os.path.join(args.mem_dir, f"{video_name}.pkl")

        # Clear old results for this video
        if os.path.exists(output_path):
            os.remove(output_path)

        video_total_tokens = 0
        for q_idx, qa in enumerate(video_questions):
            data = {
                "id": qa.get("question_id") if qa.get("question_id") is not None else f"{video_name}_{q_idx}",
                "mem_path": mem_path,
                "question": qa.get("question_text", qa.get("question")),
                "answer": qa.get("correct_answer", qa.get("answer")),
                "before_clip": qa.get("timestamp")
            }
            
            data["conversations"] = [
                {"role": "user", "content": "Searched knowledge: {}"}
            ]
            data["finish"] = False
            data["currenr_clips"] = []
            data["token_consumption"] = 0

            for idx in range(processing_config["total_round"]):
                if data["finish"]:
                    break
                
                knowledge_str = ""
                for msg in data["conversations"]:
                    if msg['role'] == 'user':
                        knowledge_str += f"{msg['content']}\n"
                
                final_round_hint = ""
                if idx == processing_config["total_round"] - 1:
                    final_round_hint = "\n(This is the final round. You MUST choose 'Answer'.)"
                
                prompt = prompt_generate_action_with_plan_structured.format(
                    question=data["question"],
                    knowledge=knowledge_str + final_round_hint
                )
                
                try:
                    response_json, tokens = generate_text_response(prompt, text_format=ActionOutput)
                    data["token_consumption"] += tokens
                except Exception as e:
                    print(f"Error calling LLM: {e}")
                    # Fallback
                    response_json = ActionOutput(reasoning="Error", action="Answer", content="Error occurred during LLM call.")
                
                data["conversations"].append({"role": "assistant", "content": f"Reasoning: {response_json.reasoning}\nAction: {response_json.action}\nContent: {response_json.content}"})
                data = process_action(data, response_json)

            # Evaluate the final answer
            if "response" in data:
                eval_res, eval_tokens = eval_answer(data["question"], data["response"], data["answer"])
                data["gpt_eval"] = eval_res
                data["token_consumption"] += eval_tokens
            else:
                data["gpt_eval"] = False
            
            video_total_tokens += data["token_consumption"]
            with open(output_path, "a") as f:
                f.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
            print(f"Processed question {data['id']} for video {video_name}")

        # Save aggregated token consumption for the video
        control_tokens_dir = "data/control_tokens"
        os.makedirs(control_tokens_dir, exist_ok=True)
        video_token_summary = {"total": video_total_tokens}
        with open(os.path.join(control_tokens_dir, f"{video_name}.json"), "w") as f:
            json.dump({"control_token_summaries": video_token_summary}, f, indent=2)
        
        # Update aggregated control tokens file
        agg_control_tokens_path = "data/control_tokens.json"
        agg_control_tokens = {}
        if os.path.exists(agg_control_tokens_path):
            try:
                with open(agg_control_tokens_path, "r") as f:
                    agg_control_tokens = json.load(f)
            except:
                pass
        agg_control_tokens[video_name] = video_token_summary
        with open(agg_control_tokens_path, "w") as f:
            json.dump(agg_control_tokens, f, indent=2)
        print(f"Saved control token consumption for {video_name}")
