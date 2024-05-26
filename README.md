# ChatGPT fruit information identification

這個專案是一個範例，展示了如何使用 Python 與多個人工智慧相關的庫（如 TensorFlow、OpenCV 和 OpenAI 的 ChatGPT）來實現幾個功能：

This project serves as an example demonstrating how to utilize Python along with several artificial intelligence-related libraries such as TensorFlow, OpenCV, and OpenAI's ChatGPT to achieve several functionalities:

1. 從攝像頭中捕獲影像。Capturing images from a webcam.
2. 使用已經訓練好的模型對捕獲的影像進行水果類別的分類。

   Performing fruit category classification on the captured images using a pre-trained model.
3. 通過按下鍵盤上的特定按鍵（例如 Enter 鍵）與 OpenAI 的 ChatGPT 模型進行對話，並獲取有關水果的相關信息。

   Engaging in conversation with OpenAI's ChatGPT model by pressing specific keys on the keyboard (e.g., the Enter key) to retrieve relevant information about fruits.

總體來說，這個專案結合了影像處理、機器學習和自然語言處理等技術，提供了一個有趣的示例，展示了如何將這些技術結合起來應用於實際應用中。

In essence, this project combines techniques from image processing, machine learning, and natural language processing, providing an intriguing example of how these technologies can be integrated for practical applications.

## 代碼Code
### 代碼簡介Code Overview

1. __初始化與設置Imports and Setup：__
   - 匯入所需模組，設置 OpenAI 客戶端和自定義函數。

     Import necessary modules, set up the OpenAI client, and import the custom Connet_ChatGPT function.
   - 加載 TensorFlow 模型和標籤文件，初始化攝像頭。

     Load the TensorFlow model and label file, and initialize the webcam.
3. __主循環：__
   - 延遲 5 秒以確保攝像頭準備就緒。
   
     Sleep for 5 seconds to ensure the webcam is ready.
   - 從攝像頭讀取影像，並將其調整為模型所需的尺寸（224x224）。

     Process the image: resize it to the required size (224x224), display it in a window, and preprocess it for classification.
   - 將處理後的影像顯示在窗口中。

     Display the processed image in a window.
   - 將影像轉換為數組並預處理，以便進行分類。

     Convert the image to an array and preprocess it for classification.
   - 使用 TensorFlow 模型進行圖像分類，並獲取分類結果和置信度。

     Perform image classification using the TensorFlow model, obtain the classification result and confidence score.
   - 輸出分類結果和置信度。

     Output the classification result and confidence score.
   - 監聽鍵盤輸入：

     Listen for keyboard input:
   - 如果按下 Enter 鍵，調用自定義的 Connet_ChatGPT 函數，並將分類結果傳遞給該函數。

     If the Enter key is pressed, call the custom Connet_ChatGPT function with the classification result.
   - 如果按下 Esc 鍵，退出循環並結束程式。

     If the Esc key is pressed, exit the loop and end the program.
4. __自定義函數 "Connet_ChatGPT"：__
   - 創建一個消息列表，並將用戶的問題添加到消息列表中。

     Create a message list and add the user's question to it.
   - 使用 OpenAI API 向 ChatGPT 發送消息並獲取回應。

     Use the OpenAI API to send the message to ChatGPT and retrieve the response.
   - 打印 ChatGPT 的回答。

     Print ChatGPT's response.
5. __清理Cleanup:__
   - 釋放攝像頭並關閉所有窗口。

     Release the webcam and close all windows.
這段程式碼實現了從攝像頭獲取影像、進行圖像分類，並與 ChatGPT 互動的完整流程。
This code implements the complete process of capturing images from a webcam, performing image classification, and interacting with ChatGPT.
---
### 代碼說明

這段代碼主要由以下部分組成：
This code consists of the following parts:

1. **導入所需模組Imports and Setup**：
   ```python
   import time
   import keyboard
   from keras.models import load_model  # 使用 TensorFlow 的 Keras 模型
   import cv2  # OpenCV 用於圖像處理
   import numpy as np  # 用於數據處理
   from openai import OpenAI  # OpenAI 客戶端
   from ChatGPTAPI import Connet_ChatGPT  # 匯入自定義的 ChatGPT 函數

2. __設定Setup__:
   ```python
   np.set_printoptions(suppress=True)  # 禁用科學計數法以提高可讀性
   model = load_model("keras_Model.h5", compile=False)  # 加載模型
   class_names = open("labels.txt", "r").readlines()  # 加載標籤
   camera = cv2.VideoCapture(0)  # 設置攝像頭


3. __主程式Main Program__:
   ```python
   time.sleep(5)  # 延遲 5 秒以確保攝像頭準備就緒

   if __name__ == '__main__':
    while True:
        ret, image = camera.read()  # 從攝像頭讀取影像

        # 處理影像
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imshow("Webcam Image", image)  # 顯示影像
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1  # 預處理影像數據
        
        # 進行圖像分類
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # 輸出分類結果
        print("Fruit:", class_name[2:], end="")  # 去掉標籤中的前兩個字符
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        fruit = class_name[2:]
    
        # 讀取鍵盤輸入
        keyboard_input = cv2.waitKey(1)
        if keyboard_input == 13:  # Enter 鍵
            Connet_ChatGPT(class_name[2:])  # 與 ChatGPT 進行對話
            time.sleep(3) 
        if keyboard_input == 27:  # Esc 鍵
            break  # 結束程式

   camera.release()
   cv2.destroyAllWindows()  # 釋放攝像頭並關閉所有窗口


4. __自定義函數Custom Function__:
   ```python
   from openai import OpenAI
   client = OpenAI()

   def Connet_ChatGPT(fruit_name):
    print("")
    messages = []
    message = f"what is {fruit_name}"
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    desired_message = response.choices[0].message.content
    print(f"This is {fruit_name}")
    print(f"ChatGPT:{desired_message}")

