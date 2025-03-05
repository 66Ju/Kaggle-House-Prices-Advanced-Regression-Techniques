# Kaggle-House-Prices-Advanced-Regression-Techniques

本專案使用 Keras 建立神經網路模型，來預測房屋的銷售價格。透過數據預處理、特徵選取以及深度學習模型訓練，提升預測精準度。此模型基於歷史房價數據進行學習，並可用於未來房價預測。


使用模型
本專案選用 深度神經網路 (DNN) 作為預測模型，並使用 Keras 搭配 TensorFlow 進行訓練。模型架構如下：
輸入層：標準化後的房屋特徵
隱藏層：7 層全連接層 (Dense)，每層包含 ReLU activation function 與 Dropout 來防止 overfitting
輸出層：單一神經元，使用 linear activation function 來輸出預測的房價


為何選擇此模型？
適用於非線性數據：房價與影響因子（如房屋大小、位置、裝潢）之間的關係較為複雜，DNN 可以學習這些非線性模式。
特徵提取能力強：深度學習擁有強大的特徵學習能力，比傳統回歸方法更能發掘影響房價的重要因素。
良好的泛化能力：使用 Dropout 來減少 overfitting，使模型能夠在新數據上保持良好的預測能力。
高效優化：使用 Adam 優化器來加速收斂，提高訓練效率。


準備數據
載入 train.csv 和 test.csv
進行標籤編碼 (Label Encoding) 與標準化處理
選擇與 SalePrice 相關係數高於 0.6 的特徵


訓練模型
使用 train_test_split 將數據集劃分為訓練集與驗證集
設定 DNN 模型並進行訓練 (epochs = 600, batch_size = 512)
使用 ModelCheckpoint 保存最佳權重


測試與預測
加載訓練好的權重 (good.weights.h5)
標準化測試數據並進行房價預測
儲存預測結果至 house_predict.csv


<img src="https://github.com/user-attachments/assets/fca13c80-9d33-471f-b83d-b7442cee4c8b" width="400">

預測的房價結果

![image](https://github.com/user-attachments/assets/4cba00dc-bcbc-481a-91c8-5f4c9555825b)

結論
本模型透過深度學習技術，有效學習房價數據的特徵，並提供較準確的價格預測。未來可進一步調整超參數或嘗試其他機器學習方法，以進一步提升模型性能。
