# Towards Prompt Engineering and Large Language Models for Post-OCR correction in handwritten texts

This experiment aims to perform post-OCR spelling corrections on handwritten text recognition competition datasets.

To evaluate the performance of LLMs in this task, prompts with complete sentences/sentences are created for the models to correct.

### Create Conda environment:
```sh
conda create --name=llm_spelling_corrector python=3.10 -y
conda activate llm_spelling_corrector
pip install -r requirements.txt
```

### Run experiments:
```sh
# Remember create .env file from .env.example
python3 predict.py \
  --datasets bressay rimes iam \
  --ocr_predictions bluche flor puigcerver \
  --output_folder corrections/gpt_4o_mini \
  --provider openai \ # gemini or ollama
  --prompt_type 1 \ # 1 for our prompt, 2 for previous work's prompt
  --llm gpt-4o-mini # gemini-2.0-flash-lite, phi4 etc.
```

### Calculate metrics:
```sh
python3 metrics.py \
    --datasets bressay rimes iam \
    --ocr_predictions bluche flor puigcerver \
    --output_corrections corrections/gpt_4o_mini
```

### Observations:
- To support other datasets modify the `get_sentences` function to identify the lines and create sentences to prompt (this is necessary to add more context to the llm In the end the lines are split again).
- The `get_sentences` function expects the dataset to be sorted to get the lines of the paragraphs in the correct order.
- To support other providers and models modify `model.py` file.