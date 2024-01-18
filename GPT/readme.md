# ENEM Solver - GPT

## Description
This script is designed to process and solve ENEM tests using OpenAI's models. It reads questions from specified CSV files, generates answers using a model, calculates the probabilities for each answer, and saves the results in new CSV files. 

NEEDS TO BE UPDATED!

## Requirements
- Python
- Pandas library
- NumPy library
- OpenAI library

## Usage
The script is executed from the command line and requires five arguments:

- input_path_1: Path to the first input CSV file containing the first day of the ENEM test.
- input_path_2: Path to the second input CSV file containing the second day of the ENEM test.
- target_path_1: Path where the first output CSV file will be saved.
- target_path_2: Path where the second output CSV file will be saved.
- api_key: OpenAI API key.
- model: OpenAI model that will perform the test.

### To run the script, use the following command format:
python script_name.py input_path_1 input_path_2 target_path_1 target_path_2 api_key model

## File Format
Ensure that the first day test don't have 'redação' in it's structure.

## Model Parameters
All model parameters are defined in the code. For any necessary changes, visit implementation. Here are the default values:

- seed=10 
- temperature=0 
- logprobs=True 
- top_logprobs=5 
- max_tokens=1
