# M3-Agent Hippo

Lightweight memory-graph pipeline for long video QA with OpenAI models.

The project runs in two stages:
- Build per-video memory graphs from frame clips.
- Run a retrieval + reasoning control loop to answer questions.

## Requirements

- Python 3.9+ (tested with modern 3.x)
- OpenAI API key
- Linux/macOS shell

Set your API key:

```bash
export OPENAI_API_KEY=your_key_here
```

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data Layout

Expected folder/file structure:

```text
data/
  frames/
    <video_id>/
      <clip_id>/
        1.jpg
        2.jpg
        ...
  questions.jsonl
```

- `frames/<video_id>/<clip_id>/*.jpg` are clip frames.
- `questions.jsonl` should include `video_id`, question text, and answer fields used by `m3_agent/control.py`.

## Stage 1: Build Memory Graphs

Generate one memory graph (`.pkl`) per video from frame clips:

```bash
python m3_agent/memorization_memory_graphs.py \
  --frames_dir data/frames \
  --save_dir data/graphs
```

Optional: process only specific videos:

```bash
python m3_agent/memorization_memory_graphs.py \
  --video_names VIDEO_ID_1 VIDEO_ID_2 \
  --frames_dir data/frames \
  --save_dir data/graphs
```

Output:
- `data/graphs/<video_id>.pkl`

## Stage 2: QA with Retrieval + Agent Loop

Run question answering using stored memory graphs:

```bash
python m3_agent/control.py \
  --questions_file data/questions.jsonl \
  --mem_dir data/graphs \
  --output_dir data/results
```

Optional: process only specific videos:

```bash
python m3_agent/control.py \
  --video_names VIDEO_ID_1 VIDEO_ID_2 \
  --questions_file data/questions.jsonl \
  --mem_dir data/graphs \
  --output_dir data/results
```

Output:
- `data/results/<video_id>.jsonl` (per-question reasoning/action trace + prediction + evaluation)

## Main Components

- `m3_agent/memorization_memory_graphs.py`  
  Builds memory graphs from clip frames.
- `m3_agent/control.py`  
  Iterative retrieve-or-answer controller for QA.
- `mmagent/videograph.py`  
  Core graph structure and search primitives.
- `mmagent/retrieve.py`  
  Memory retrieval over graph content.
- `mmagent/memory_processing.py`  
  Parses and structures memory statements.
- `mmagent/mllm_gpt.py`, `mmagent/llm.py`  
  OpenAI calls for multimodal/text generation and embeddings.
- `mmagent/output_structure.py`  
  Pydantic schemas for structured model outputs.

## Notes

- Default text model: `gpt-5-mini`
- Default embedding model: `text-embedding-3-small`
- If a run is interrupted, scripts are generally safe to re-run:
  - memory build skips videos with existing `.pkl`
  - control appends results to output JSONL files
