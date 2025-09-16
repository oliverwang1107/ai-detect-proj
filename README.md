# Multi-Modal AI 生成圖像檢測器 (PRNU / ELA / CLIP)

這是一個功能強大的 AI 生成圖像檢測系統，它整合了三種不同的特徵分析方法（PRNU、ELA、CLIP）來提高檢測的準確性和穩健性。系統採用滑動窗口（Tiling）的方式對任意尺寸的圖像進行全面分析，並透過一個訓練好的融合模型（Logistic Regression）來結合各模型的判斷，最終給出一個綜合評分。

專案內含完整的數據預處理、特徵提取、模型訓練到最終推論與可視化的流程，並提供一個基於 Gradio 的互動式 Web UI 介面，方便使用者上傳圖像進行即時檢測。

## 系統特色

  * **多模態融合 (Multi-Modal Fusion)**：結合三種獨立的特徵來進行判斷：
      * **PRNU (光響應不均勻性)**：利用感光元件的固有噪聲模式來識別圖像來源。
      * **ELA (錯誤級別分析)**：透過分析 JPEG 壓縮的錯誤級別差異來檢測圖像的偽造區域。
      * **CLIP (對比語言-圖像預訓練)**：利用大型視覺模型的深層特徵來區分真實與 AI 生成的圖像風格。
      
  * **滑動窗口推論 (Tiling Inference)**：可分析任意尺寸的圖像。系統會將大圖切分為 256x256 的圖塊（Tile），逐塊分析後再進行綜合評分。
  * **GPU 加速**：核心運算利用 PyTorch (CUDA) 和 RAPIDS cuML 函式庫，大幅提升訓練與推論效率。
  * **互動式 Web UI**：內建 Gradio 介面 (`inf.ipynb`, `inf_full.ipynb`)，使用者可輕鬆上傳圖片、調整參數並即時查看熱度圖（Heatmap）和檢測結果。
  * **彈性的彙總策略**：提供多種策略（如 `mean_prob`, `max_prob`, `topk_mean`）將所有圖塊的檢測分數彙總為一個最終的全圖分數。
  * **廣泛的格式支援**：除了標準的 JPG/PNG，還整合了 `pillow-heif` 和 `pillow-avif-python`，可直接處理 HEIC 和 AVIF 格式的圖像。

## 系統流程

整個repo主要分為三個部分，特徵提取、模型訓練、推論與可視化

1.  **特徵提取 (`multi_feature.ipynb`)**

      * **輸入**：包含真實圖像和 AI 生成圖像的資料夾。
      * **過程**：遍歷所有圖像，提取 PRNU、ELA 和 CLIP 特徵。
      * **輸出**：將每個圖像的每個特徵保存為獨立的 `.npy` 檔案，存放在 `features_256/` 目錄下。

2.  **模型訓練 (`model_train.ipynb`)**

      * **輸入**：上一步產生的 `.npy` 特徵檔案和一個定義了訓練/驗證/測試集的 `split.json` 檔案。
      * **過程**：
          * 分別為 PRNU 和 ELA 特徵訓練輕量級的 `FastCNN` 模型。
          * 在 CLIP 特徵上訓練一個 `LogisticRegression` 分類器。
          * 在驗證集上取得三種模型的 Logit 輸出，並用這些 Logit 作為特徵，訓練一個最終的 `LogisticRegression` 融合模型。
      * **輸出**：所有訓練好的模型（`.pt` 和 `.pkl` 檔案）保存在 `saved_models/` 目錄下。

3.  **推論與可視化 (`inf.ipynb` / `inf_full.ipynb`)**

      * **輸入**：一張待檢測的圖像。
      * **過程**：
          * 載入 `saved_models/` 中的所有預訓練模型。
          * 對圖像進行切塊，逐塊提取特徵並送入對應模型計算 Logit。
          * 使用融合模型結合三者的 Logit，得出最終的「偽造機率」。
          * 根據所有圖塊的機率，生成熱度圖並計算全圖的綜合分數。
      * **輸出**：包含預測標籤、偽造機率、圖塊詳細資訊以及熱度圖路徑的 JSON 結果。

## 安裝與設定

建議使用 Conda 或 Mamba 來管理環境，特別是為了安裝 RAPIDS cuML。

**主要依賴套件:**

  * `PyTorch` (CUDA 版本)
  * `RAPIDS cuML` (用於 GPU 加速的機器學習庫)
  * `open_clip_torch`
  * `scikit-learn`, `scikit-image`
  * `Pillow`, `pillow-heif`, `pillow-avif-python`
  * `gradio`, `numpy`, `joblib`

**環境建議 (以 `mamba` 為例):**

```bash
# 創建一個名為 rapids-25 的新環境並安裝核心依賴
mamba create -n rapids-25 -c rapidsai -c nvidia -c conda-forge \
    cuml=24.* python=3.10 cuda-version=12.* -y

# 啟用環境
mamba activate rapids-25

# 安裝其他 Python 套件
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch scikit-image scikit-learn joblib Pillow pillow-heif pillow-avif-python gradio
```

## 使用說明

### 1\. 執行推論 (Gradio UI)

這是最直接的使用方式，用於檢測單張圖片。

1.  **準備模型**：確保所有預訓練模型（`*.pt`, `*.pkl`）都已放置在 `saved_models/` 資料夾中。
2.  **執行 Notebook**：打開 `inf.ipynb` 或 `inf_full.ipynb`，並按順序執行所有儲存格。
3.  **啟動 UI**：執行最後一個儲存格後，Gradio 介面將會出現在 Notebook 輸出中。
4.  **操作**：
      * 在左側上傳或貼上圖片。
      * 調整圖塊大小（Tile）、步長（Stride）和視覺化透明度（Alpha）等參數。
      * 點擊「開始推論」按鈕。
      * 右側將顯示最終預測結果、可視化熱度圖以及包含所有圖塊詳細資訊的原始 JSON 數據。

### 2\. 訓練自己的模型 (可選)

如果您想用自己的資料集來訓練模型，請依序執行以下步驟：

1.  **特徵提取 (`feature.ipynb`)**：

      * 在此 Notebook 的 `CONFIG` 區塊設定您的真實圖片 (`REAL_DIR`) 和 AI 圖片 (`FAKE_DIR`) 的路徑。
      * 執行 Notebook，它將自動處理所有圖片並將 `.npy` 特徵檔案存放在 `features_256/`。

2.  **模型訓練 (`model_train.ipynb`)**：

      * 準備一個 `split.json` 檔案來定義您的訓練、驗證和測試集。
      * 在 Notebook 中設定 `FEA_ROOT` 和 `SPLIT_JSON` 的路徑。
      * 依序執行儲存格，將會分別訓練 PRNU-CNN、ELA-CNN、CLIP-LogReg 以及最終的 Fusion-LR 模型。
      * 所有訓練完成的模型將保存在 `saved_models/` 中。

## 檔案結構說明

```
.
├── inf.ipynb                # 主要推論 Notebook，包含 Gradio UI
├── inf_full.ipynb           # 功能更完整的推論 Notebook，支援 HEIC/AVIF
├── model_train.ipynb        # 用於訓練所有模型的 Notebook
├── feature.ipynb      # 用於從圖像資料集提取特徵的 Notebook
│
├── saved_models/            # 存放預訓練模型
│   ├── prnu_fastcnn_best.pt
│   ├── ela_fastcnn_best.pt
│   ├── clip_logreg_gpu.pkl
│   └── fusion_lr.pkl
│
├── features_256/            # (由 multi_feature.ipynb 生成) 存放特徵 .npy 檔
│   ├── prnu_real_npy/
│   ├── prnu_fake_npy/
│   ├── ela_real_npy/
│   ├── ...
│
└── splits/                  # 存放資料集劃分定義
    └── combined_split.json
```

## 模型架構

  * **FastCNN**：用於 PRNU 和 ELA 特徵的輕量級卷積神經網絡，採用深度可分離卷積（Depthwise-Separable Convolutions）以提高效率。
  * **CLIP Head**：使用預訓練的 `ViT-L-14(laion2b_s32b_b82k)` 作為骨幹網絡，並在其提取的特徵之上訓練一個 cuML Logistic Regression 分類器。
  * **融合模型 (Fusion Model)**：一個 Logistic Regression 分類器，其輸入為上述三個獨立模型的 Logit 分數，輸出最終的偽造機率。

## 訓練與評估結果

以下為最近一次以「CLIP 後端使用 cuML LogisticRegression (logit)」所得到的驗證/測試成績（IID split）：

### 單模態（val）

- PRNU：acc=0.9153，auc=0.9613
- ELA：acc=0.9365，auc=0.9779
- CLIP：acc=0.9321，auc=0.9754

融合（以驗證集 logits 訓練 LR）

- Fusion (LR on val) AUC=0.9875

### 單模態（test）

- PRNU：acc=0.9181，auc=0.9628
- ELA：acc=0.9365，auc=0.9779
- CLIP：acc=0.9292，auc=0.9767

### FUSION（test）

- Fusion：acc=0.9608，auc=0.9882

混淆矩陣：

```
[[8151  249]
 [ 401 7771]]
```

分類報告：

```
              precision    recall  f1-score   support

        real     0.9531    0.9704    0.9617      8400
        fake     0.9690    0.9509    0.9599      8172

    accuracy                         0.9608     16572
   macro avg     0.9610    0.9606    0.9608     16572
weighted avg     0.9609    0.9608    0.9608     16572
```

模型與結果檔案：

- 儲存的融合模型：`Script/saved_models/fusion_lr.pkl`
- 融合模型中繼資料：`Script/saved_models/fusion_lr_meta.json`
- 測試評估分數輸出：`Script/exports/fusion_eval/iid`
