#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: harryyhwang
# Date: 2024/7/3
# Description: Base API handler

import os
import sys

sys.path.append(os.getcwd())
sys.path.append("..")

class BaseAPIHandler:
    def __init__(self):
        pass

    def initialize_llm(self, token):
        pass

    def run_llm(self, query, sys_message="You are an AI assistant."):
        pass

if __name__ == '__main__':
    my_vlm = BaseAPIHandler()
    my_vlm.initialize_llm()
    answer = my_vlm.run_llm("waht is your name")
    print(answer)