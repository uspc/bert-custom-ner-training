Step 1 - Generate sample incident file with custom entities for training in JSONL formate
Program data_gen_custom_entities.py generates custom_incidents.jsonl file

Step 2 - Assign BIO (Beginning Inside Outside) mapping for the entities identified 
Program generate_bio_file.py uses custom_incidents.jsonl and generates bio_dataset.jsonl with BIO mapping for custom entities 

Step 3 - Use the bio_dataset.jsonl to train the transformer 
Program train_model.py uses  bio_dataset.jsonl to train the. Model. The generated model is output into /results folder 

Step 4 - Use the BIO file to generate IDs which the BERT model understands 
Program prepare_ids.py uses bio_dataset.jsonl and generates IDs into processed_dataset folder 

Step 5 - Use the processed_dataset file to train the model 
Program train_bert.py uses processed dataset to train the model and creates results folder 

Step 6 - Use trained model from results folder to infer 
Use inference program infer.py for generating custom entities on new data 


Step XX - After step3 this program does both id and training. 
model_train_from_bio_format uses biod_dataset.jsnol file to both generate the Ids and train the model 
Test model through inference 
Program modelInference.py uses trained model available in results folder for inference



Step XX - After step 1 use model_train_from_custom_entities.py for BIO, Ids and model training 
