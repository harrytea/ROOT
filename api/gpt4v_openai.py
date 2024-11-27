#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: harryyhwang
# Date: 2024/7/3
# Description: gpt-4v api

import json
import os
import sys
import base64
from openai import OpenAI

sys.path.append(os.getcwd())
sys.path.append("..")

class GPT4VHandler:
    def __init__(self):
        pass
    
    def initialize_llm(self, api_key=None):
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=api_key)

    def encode_image(self, _image_path):
        with open(_image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def run_llm(self, query, image_path, sys_message="You are an AI assistant that helps people desc image."):
        image = self.encode_image(image_path)

        while True:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {"role": "system", "content": sys_message},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                            {"type": "text", "text": query}
                        ]}
                    ],
                    temperature=0,
                    max_tokens=1024
                )
                answer = response.choices[0].message.content
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                return "Error"
        return answer


if __name__ == '__main__':
    my_vlm = GPT4VHandler()
    my_vlm.initialize_llm()
    answer = my_vlm.run_llm("what is this", "path/to/your/image.jpg")
    print(answer)
