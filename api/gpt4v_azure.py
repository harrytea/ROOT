#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: harryyhwang
# Date: 2024/6/25
# Description: gpt-4v api

import os
import json
import base64

import openai
from openai import AzureOpenAI

import sys

sys.path.append(os.getcwd())
sys.path.append("..")
import pdb

class GPT4VHandler:
    def __init__(self):
        pass
    
    def initialize_llm(self, host="", token=""):
        host = host
        service_platform = "azure"
        project_en_name = "public"
        # token = os.environ["OPENAI_API_KEY"]
        token = token


        openai.api_type = "azure"
        openai.api_base = f"{host}/{service_platform}/{project_en_name}"
        openai.api_version = "2024-07-01-preview"
        openai.api_key = token
        openai.base_url = f"{host}/{service_platform}/{project_en_name}"

        # gets the API Key from environment variable AZURE_OPENAI_API_KEY
        client = AzureOpenAI(
            api_version="2024-07-01-preview",
            api_key=token,
            azure_endpoint="",
        )

        self.client = client

    def encode_image(self, _image_path):
        with open(_image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def run_llm(self, query, image_path, sys_message="You are an AI assistant that helps people desc image."):
        sys_message = sys_message
        image = self.encode_image(image_path)

        while True:
            try:
                response = self.client.chat.completions.create(
                            model="gpt-4-vision-preview",  # e.g. gpt-35-instant
                            messages = [
                                {"role": "system", "content": [{"type": "text",  "text": sys_message},],},
                                {"role": "user", "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                                    {"type": "text",  "text": "{}".format(query)}
                                    ],
                                }
                            ],
                            temperature=0,
                            max_tokens=1024
                        )
                answer = json.loads(response.model_dump_json())
                answer = answer["choices"][0]["message"]["content"]
                break
            except:
                return "Error"
                break
        return answer

    


if __name__ == '__main__':
    my_vlm = GPT4VHandler()
    my_vlm.initialize_llm()
    answer = my_vlm.run_llm("what is this", "/llm-cfs-nj/person/harryyhwang/2D-Scene-data/0001909.jpg")
    print(answer)