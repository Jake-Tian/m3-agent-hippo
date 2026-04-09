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

prompt_generate_full_memory_hippo = """
You are given a 30-second video represented as sequential frames (pictures in chronological order). 

Your tasks: 

1. **Episodic Memory** (the ordered list of atomic captions)
- Describe each character's behavior in chronological order.
- Include:
  (a) Interaction with objects in the scene.
  (b) Interaction with other characters.
  (c) Actions and movements.
- Describe specific gesture, movement, or interaction performed by the characters.
- Quote—or, if necessary, summarize—what are spoken by the characters.
- Include precise location information for placement, retrieval, or movement when characters interact with objects.
- Each entry must describe exactly one event/detail. Split sentences if needed.

2. **Semantic Memory** (the ordered list of high-level reasoning-based conclusions)
- Produce concise, high-level reasoning-based conclusions:
  (a) Character-level Attributes – Infer abstract attributes: Name, Personality, Role, Interests.
  (b) Interpersonal Relationships & Dynamics – Describe relationships, emotions, tone, power dynamics.
  (c) Video-level Plot Understanding – Summarize narrative arc, tone, cause-effect.

3. **Characters' Appearance**
- Describe or update each character's appearance: facial features, clothing, body shape, hairstyle, or other distinctive characteristics.
- Format as a list of appearance objects with "name" and "appearance" strings.

4. **Main Character**
- Identify the main character of the video.

## Character Naming Rules:
- The character names from the video are initially unknown.
- You can refer to characters by <character_1>, <character_2>, etc., or by their job if it can be easily deduced (eg. <police>).
- Use the character's name only if it can be explicitly extracted from the conversation (eg. <Alice>).
- Use angle brackets to represent characters in episodic memory, semantic memory, and character appearance.

## Character Matching Rules:
For characters appearing in the video:
- First, check if the character name is known. If so, use it (eg. <Alice>).
- If high similarity is found between a newly named character and an unknown character (eg. <character_1>), include "Equivalence: <character_1>, <Alice>" in semantic memory.
- If equivalence is found, refer to this character by its character name instead of the placeholder in all output sections.
- Match characters by appearance consistency across clips. Minimize the number of unknown characters.

Strict Requirements:
- Output exactly as a JSON object with keys: "episodic_memory", "semantic_memory", "characters_appearance", "main_character".

Expected Output Format:
{
  "episodic_memory": ["<Alice> enters the room.", "<Alice> sits with <character_1>."],
  "semantic_memory": ["Equivalence: <character_1>, <Bob>", "<Alice> is <Bob>'s teacher."],
  "characters_appearance": [{"name": "<Alice>", "appearance": "ponytail, red shirt"}, {"name": "<Bob>", "appearance": "short hair, blue jeans"}],
  "main_character": "<Alice>"
}
"""

prompt_generate_action_with_plan_structured = """
You are given a question and some relevant knowledge about a specific video. 
Your task is to reason about whether the provided knowledge is sufficient to answer the question.

Input:
- Question: {question}
- Knowledge: {knowledge}

Output Requirements (Strict JSON):
Return a JSON object with the following keys:
1. "reasoning": A brief analysis of the question and current knowledge. Identify what is missing if information is insufficient.
2. "action": Either "Answer" (if sufficient) or "Search" (if more info is needed).
3. "content": 
   - If action is "Answer": Provide the final short, clear, and direct answer. Use character names if available.
   - If action is "Search": Provide a single, keyword-based search query to retrieve missing information.

Naming Rules for Search/Answer:
- Use character names (e.g., <Alice>) if known.
- Do NOT use generic tags like <character_1> if the name is available in the knowledge.

Output format example:
{{
  "reasoning": "We know <Alice> was in the kitchen, but we don't know what she took from the fridge.",
  "action": "Search",
  "content": "<Alice> fridge interaction"
}}
"""

prompt_agent_verify_answer_referencing = """You are provided with a question, a ground truth answer, and an answer from an agent model. Your task is to determine whether the ground truth answer can be logically inferred from the agent's answer, in the context of the question.

Do not directly compare the surface forms of the agent answer and the ground truth answer. Instead, assess whether the meaning expressed by the agent answer supports or implies the ground truth answer. If the ground truth can be reasonably derived from the agent answer, return "Yes". If it cannot, return "No".

Important notes:
	•	Do not require exact wording or matching structure.
	•	Semantic inference is sufficient, as long as the agent answer entails or implies the meaning of the ground truth answer, given the question.
	•	Only return "Yes" or "No", with no additional explanation or formatting.

Input fields:
	•	question: {question}
	•	ground_truth_answer: {ground_truth_answer}
	•	agent_answer: {agent_answer}

Output ('Yes' or 'No'):"""
