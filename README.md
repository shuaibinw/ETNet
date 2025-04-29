# ETNet
Transformer-based Transferable Deep Learning Framework for Enhancer-Enhancer Interaction Prediction
## 1. Clone the UniChrom repository:
git clone https://github.com/shuaibinw/ETNet.git
<br>cd ETNet
## 2. Install the required dependencies:
tensorflow>=2.4.0
<br>keras>=2.4.0
<br>pandas>=1.2.0
<br>numpy>=1.19.0
<br>scipy>=1.7.3
<br>matplotlib>=3.3.0
<br>seaborn>=0.12.2
<br>plotly>=4.14.0
<br>scikit-learn>=1.0.2
<br>shap>=1.0.0
<br>tqdm>=4.66.4
<br>pyfaidx>=0.8.10.3
<br>seaborn>=0.12.2
<br>deeplift>=0.6.13.0
<br>h5py>=2.10.0




## 3. Run ETNet:
python train_GM12878.py
<br>python train_K562.py
<br>python train_MCF-7.py
## predict:
eg:python  predict.py chr22:46995378-46997378 chr8:129203880-129205880
