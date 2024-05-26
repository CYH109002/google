# ChatGPT Fruit Information Query

這是一個用於與 OpenAI 的 ChatGPT 模型進行對話的簡單 Python 函數範例。該函數會向 ChatGPT 發送一個有關水果的問題並打印回答。

## 代碼說明

這段代碼主要由以下部分組成：

1. **導入所需模組**：
   ```python
   import time
   import keyboard
   from keras.models import load_model  # 使用 TensorFlow 的 Keras 模型
   import cv2  # OpenCV 用於圖像處理
   import numpy as np  # 用於數據處理
   from openai import OpenAI  # OpenAI 客戶端
   from ChatGPTAPI import Connet_ChatGPT  # 匯入自定義的 ChatGPT 函數

2. __初始化客戶端__:
   ```python
   client = OpenAI()

3. __定義函數 Connet_ChatGPT__:
   ```python
   def Connet_ChatGPT(fruit_name):

4. __初始化消息列表__:
   ```python
   messages = []
