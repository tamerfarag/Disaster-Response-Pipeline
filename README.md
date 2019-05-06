# Disaster Response Pipeline

This project analyzes historical communication messages that are captured after some disasters to be filtered and submitted to the concerned disaster response organizations.
<p align="center">
<img src='images/sample.png' width="600" height="600" />
</p>

### Reading and processing the data
	* Merge data from two input files, disaster_messages.csv and disaster_categories.csv then remove duplicates and save into one database.
	* Apply NLP pipeline process "tokenization and limmetization" to the merged data.
	* Add a MultiOutputClassifier to the pipeline.
	* Tune the parameters and apply a gridsearch.
	* Train and save the model.

### Instructions:
	1. Run the following commands in the project's root directory to set up your database and model.

		- To run ETL pipeline that cleans data and stores in database
			`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
		- To run ML pipeline that trains classifier and saves
			`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

	2. Run the following command in the app's directory to run your web app.
		`python run.py`

	3. Go to http://0.0.0.0:3001/
