# NLP
* This repo contains the NLP projects. Each folder container the notebook using which the model is trained and the deployment procedure.
* Dockerization step is deliberately skipped here.
* To launch the streamlit app to interact with the model, follow these steps:
    * Clone the repo.
    * Goto to respective folder and type `pip install -r req.txt`. This will install all the dependencies which are required to run the model.
    * Once, the packages are installed, type the `streamlit run main.py` command in terminal. This will launch the streamlit app in the browser.
    * Now enter the text and hit enter and you will get the result. 
    * Due to hardware constraints, there were some limitations in training the model so you might get false results sometimes.
    * To use BERT model, use this link to download the model and keep the model in the same folder: 
    `https://drive.google.com/drive/folders/1ZMCiXK1nmZ4VIt32C91HAJ9tLrSIfvM2?usp=sharing`


For other projects with model training notebooks:
bike demand prediction: `https://github.com/sankettgorey/Bike-Demand-Prediction`
ten year heart disease prediction: `https://github.com/sankettgorey/Ten-Year-CHD-Prediction`