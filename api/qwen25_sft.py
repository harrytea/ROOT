import os
import sys

sys.path.append(os.getcwd())
sys.path.append("..")

from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen25Handler:
    def __init__(self):
        pass
    
    def initialize_llm(self, checkpoint):
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
        # self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="float32", device_map="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def run_llm(self, query):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
    
        while True:
            try:
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                break
            except:
                print()
        return answer


if __name__ == '__main__':
    my_vlm = Qwen25Handler()
    my_vlm.initialize_llm(checkpoint="Qwen/Qwen2.5-3B-Instruct")
    answer = my_vlm.run_llm("hi")
    print(answer)


