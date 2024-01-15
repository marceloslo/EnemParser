import os
import pandas as pd
import openai
import numpy as np
import time

class DataHandler:
    def __init__(self, input_path_1, input_path_2, target_path_1, target_path_2):
        self.input_path_1 = input_path_1
        self.input_path_2 = input_path_2
        self.target_path_1 = target_path_1
        self.target_path_2 = target_path_2
        self.data_1 = None
        self.data_2 = None

    def read_data(self):
        self.data_1 = pd.read_csv(self.input_path_1)
        self.data_2 = pd.read_csv(self.input_path_2)

    def save_data(self):
        self.data_1.to_csv(self.target_path_1, index=False)
        self.data_2.to_csv(self.target_path_2, index=False)


class Solver:
    def __init__(self, model="gpt-4", seed=10, temperature=0, logprobs=True, top_logprobs=5, max_tokens=1):
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
                {"role": "assistant", "content": "Answer multiple choice questions with a single alternative (A B C D E)."},
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
            "gpt-4_answer": response_object.choices[0].message.content,
            "gpt-4_probs": top_logprobs
        })
        time.sleep(3)
        return response_series


def main(input_path_1, input_path_2, target_path_1, target_path_2, api_key):
    openai.api_key = api_key

    data_handler = DataHandler(input_path_1, input_path_2, target_path_1, target_path_2)
    data_handler.read_data()

    solver = Solver()

    data_handler.data_1[['gpt-4_answer', 'gpt-4_probs']] = data_handler.data_1.apply(solver.get_answer_and_probs, axis=1)
    data_handler.data_2[['gpt-4_answer', 'gpt-4_probs']] = data_handler.data_2.apply(solver.get_answer_and_probs, axis=1)

    data_handler.save_data()

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python solver_gpt4.py <input_path_1> <input_path_2> <target_path_1> <target_path_2> <api_key>")
        sys.exit(1)

    input_path_1 = sys.argv[1]
    input_path_2 = sys.argv[2]
    target_path_1 = sys.argv[3]
    target_path_2 = sys.argv[4]
    api_key = sys.argv[5]

    main(input_path_1, input_path_2, target_path_1, target_path_2, api_key)