# json-extraction-paper

## Running the code

Pre-train:

`python run.py --datasets=dbpedia,atis,slurp,snips --model_name=google/flan-t5-large --output_dir=outputs/pre-training --max_iterations=400000`

Fine-tune:

`python run.py --datasets=gpt-emails --model_name=outputs/pre-training/checkpoints/latest --output_dir=outputs/fine-tuning --max_iterations=10000`

Test:

`python run.py --datasets=gpt-emails --model_name=outputs/fine-tuning/checkpoints/best --output_dir=outputs/testing --no_train --test`