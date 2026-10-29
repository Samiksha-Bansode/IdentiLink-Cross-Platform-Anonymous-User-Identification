# IdentiLink — Cross-Platform Anonymous User Identification

## Project Structure

IdentiLink/
├── backend/
│   ├── app/
│   │   ├── main.py                          
│   │   ├── api/routes/                      
│   │   │   ├── identify_routes.py           
│   │   │   ├── fake_detection_routes.py     
│   │   │   └── trust_score_routes.py        
│   │   ├── services/
│   │   │   ├── identity_service.py          
│   │   │   ├── fake_detection_service.py    
│   │   │   └── trust_score_service.py       
│   │   ├── feature_engineering/             
│   │   ├── analysis/content_analysis.py     
│   │   ├── dataset_pipeline/                
│   │   ├── ml_models/ , training/           
│   ├── dataset/{raw,processed}/
│   └── requirements.txt
└── frontend/
    └── index.html                           


#Commands

cd backend
pip install -r requirements.txt
python -m textblob.download_corpora   
uvicorn app.main:app --reload



## API Endpoints

All three endpoints take a JSON body like {"username": "nikhil123"}.

POST /identify returns up to 10 matching accounts with a match percentage
POST /fake_detection returns account_type (real or fake) and a fake_probability
POST /trust_score returns the trust_score, the account type and the closest matching account
GET / health check



### Match % Formula

Same username on another platform:
Score = 0.30 × Username_similarity
      + 0.35 × BERT_similarity
      + 0.25 × TF-IDF_similarity
      + 0.10 × Stylometry_similarity

Username contains the same name (e.g. nikhil.ai for nikhil123):
Score = 0.55 × Username_similarity
      + 0.20 × BERT_similarity
      + 0.15 × TF-IDF_similarity
      + 0.10 × Stylometry_similarity

Only a similar username:
Score = 0.40 × Username_similarity
      + 0.30 × BERT_similarity
      + 0.20 × TF-IDF_similarity
      + 0.10 × Stylometry_similarity


### Fake Detection Formula

risk = 0.45 × Follow/Activity_pattern
     + 0.25 × Digits_in_username
     + 0.20 × Generic_bio
     + 0.10 × Short_bio

risk >= 0.58  →  Fake


### Trust Score Formula

authenticity  = (1 − fake_probability) × 100
corroboration = min(100, best_match% × 0.8 + min(number_of_matches, 5) × 4)

trust = 0.5 × authenticity + 0.5 × corroboration

Fake account:  trust = trust × 0.7


### Sentiment
TextBlob polarity → Positive / Negative / Neutral
