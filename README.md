# ChatGPT Fruit Information Query

這是一個用於與 OpenAI 的 ChatGPT 模型進行對話的簡單 Python 函數範例。該函數會向 ChatGPT 發送一個有關水果的問題並打印回答。

## 代碼

這段程式碼結合了 TensorFlow、OpenCV 和 OpenAI 的 ChatGPT，用於從攝像頭獲取影像、進行圖像分類，並在按下 Enter 鍵時與 ChatGPT 進行對話。

### 代碼簡介

1. __初始化與設置：__
   - 匯入所需模組，設置 OpenAI 客戶端和自定義函數。
   - 加載 TensorFlow 模型和標籤文件，初始化攝像頭。
  
2. __主循環：__
   - 延遲 5 秒以確保攝像頭準備就緒。
   - 從攝像頭讀取影像，並將其調整為模型所需的尺寸（224x224）。
   - 將處理後的影像顯示在窗口中。
   - 將影像轉換為數組並預處理，以便進行分類。
   - 使用 TensorFlow 模型進行圖像分類，並獲取分類結果和置信度。
   - 輸出分類結果和置信度。
   - 監聽鍵盤輸入：
   - 如果按下 Enter 鍵，調用自定義的 Connet_ChatGPT 函數，並將分類結果傳遞給該函數。
   - 如果按下 Esc 鍵，退出循環並結束程式。

3. __自定義函數 "Connet_ChatGPT"：__
   - 創建一個消息列表，並將用戶的問題添加到消息列表中。
   - 使用 OpenAI API 向 ChatGPT 發送消息並獲取回應。
   - 打印 ChatGPT 的回答。

4. __清理:__
   - 釋放攝像頭並關閉所有窗口。

這段程式碼實現了從攝像頭獲取影像、進行圖像分類，並與 ChatGPT 互動的完整流程。

### 代碼說明

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

2. __設定__:
   ```python
   np.set_printoptions(suppress=True)  # 禁用科學計數法以提高可讀性
   model = load_model("keras_Model.h5", compile=False)  # 加載模型
   class_names = open("labels.txt", "r").readlines()  # 加載標籤
   camera = cv2.VideoCapture(0)  # 設置攝像頭


3. __主程式__:
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


4. __初始化消息列表__:
   ```python
   messages = []
