import os
import pandas as pd
import openai
import numpy as np
import time
import sys

class DataHandler:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.data_files = {}

    def read_data(self):
        for file in os.listdir(self.input_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(self.input_folder, file)
                self.data_files[file] = pd.read_csv(file_path)

    def save_data(self):
        for file_name, data in self.data_files.items():
            output_file_name = f'solution_{file_name}'
            output_file_path = os.path.join(self.output_folder, output_file_name)
            data.to_csv(output_file_path, index=False)


class Solver:
    def __init__(self, model, seed=10, temperature=0, logprobs=True, top_logprobs=5, max_tokens=1):
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.max_tokens = max_tokens

    @staticmethod
    def softmax(logits):
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    @staticmethod
    def clean_probs(top_probs, valid_tokens={'A', 'B', 'C', 'D', 'E'}, default_prob=0.0001): 
        combined_probs = {}
        for token, prob in top_probs.items():
            stripped_token = token.strip()
            if stripped_token in valid_tokens:
                combined_probs[stripped_token] = combined_probs.get(stripped_token, 0) + prob
        for token in valid_tokens:
            combined_probs.setdefault(token, default_prob)
        total_prob = sum(combined_probs.values())
        if total_prob > 1.0:
            combined_probs = {token: prob / total_prob for token, prob in combined_probs.items()}
        return combined_probs

    def create_input_prompt(self, question_row):
        components = [question_row[key].replace('\n', '') for key in ['question', 'body', 'A', 'B', 'C', 'D', 'E']]
        return f"{components[0]} - {components[1]} A:{components[2]} B:{components[3]} C:{components[4]} D:{components[5]} E:{components[6]}"

    def create_completion_object(self, prompt):
        return openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "assistant", "content": "You are designed to answer the following multiple choice question. Without adding any extra characters, spaces, or newline characters in the answer, provide a single alternative as the answer."},
                {"role": "user", "content": prompt}
            ],
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            max_tokens=self.max_tokens,
            seed=self.seed,
            temperature=self.temperature
        )
    
    def manipulate_completion_response(self, response_object):
        top_logprobs = response_object.choices[0].logprobs.content[0].top_logprobs
        top_logprobs_dict = {obj["token"]: obj["logprob"] for obj in top_logprobs}
        softmax_values = self.softmax(list(top_logprobs_dict.values()))
        top_logprobs_dict = {key: softmax_value for key, softmax_value in zip(top_logprobs_dict.keys(), softmax_values)}
        return self.clean_probs(top_logprobs_dict)

    def get_answer_and_probs(self, question_row):
        input_prompt = self.create_input_prompt(question_row)
        response_object = self.create_completion_object(input_prompt)
        top_logprobs = self.manipulate_completion_response(response_object)
        response_series = pd.Series({
            f"{self.model}_answer": response_object.choices[0].message.content,
            f"{self.model}_probs": top_logprobs
        })
        time.sleep(3)
        return response_series


def main(input_folder, output_folder, api_key, model):
    openai.api_key = api_key

    data_handler = DataHandler(input_folder, output_folder)
    data_handler.read_data()

    solver = Solver(model=model)

    answer_col_name = f"{model}_answer"
    probs_col_name = f"{model}_probs"

    for file_name, data in data_handler.data_files.items():
        data[[answer_col_name, probs_col_name]] = data.apply(solver.get_answer_and_probs, axis=1)

    data_handler.save_data()
    

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script_name.py <input_folder> <output_folder> <api_key> <model>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    api_key = sys.argv[3]
    model = sys.argv[4]

    main(input_folder, output_folder, api_key, model)