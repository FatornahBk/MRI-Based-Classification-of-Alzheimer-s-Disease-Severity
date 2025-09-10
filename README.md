# MRI-Based Classification of Alzheimer's Disease Severity (Inception v3)

## Run local
pip install -r requirements.txt
streamlit run app.py

## Deploy (Streamlit Cloud)
- Put your model checkpoint in GitHub Releases.
- Go to Settings → Secrets and set:
  MODEL_URL = https://github.com/FatornahBk/MRI-Based-Classification-of-Alzheimer-s-Disease-Severity/releases/download/cnn/inception_v3_checkpoint_fold0.pt
