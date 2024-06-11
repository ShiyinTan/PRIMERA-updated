

from datasets import load_metric

rouge = load_metric("rouge")
with open(f"test_hf_save/generated_summaries_beam_5_test.txt") as f:
    generated_summaries = []
    for line in f:
        generated_summaries.append(line.strip())
with open(f"test_hf_save/gt_summaries_beam_5_test.txt") as f:
    gt_summaries = []
    for line in f:
        gt_summaries.append(line.strip())
result = rouge.compute(predictions=generated_summaries, references=gt_summaries, rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
print("ROUGE scores:")
print(result)