# Disaster Response Pipeline Project
## In fulfilment of the Udacity Data Science Nanodegree Program

## Table of Contents
1. [Project Summary](#summary)
2. [Folders and Files](#folders_files)
	1. data 	
	2. models	
	3. app	
3. [Installation and Execution](#InstAndExec)
	1. Dependencies
	2. Clone Repository
	3. How to execute the application
4. [Authors](#Authors)
5. [Licence](#Licence)
6. [References](#References)

+++++++++++++++++++++++++++
<a name="summary"></a>
1. Project Summary

<a name="folders_files"></a>
2. Folders and Files
	1. data 	
	2. models	
	3. app	

<a name="InstAndExec"></a>
3. Installation and Execution
	1. Dependencies
	2. Clone Repository
	3. How to execute the application
		1. Run the following commands in the project's root directory to set up your database and model.
		    - To run ETL pipeline that cleans data and stores in database
		     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
		    - To run ML pipeline that trains classifier and saves
		     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
		2. Run the following command in the app's directory to run your web app.
		    `python run.py`
		3. Go to http://0.0.0.0:3001/
<a name="Authors"></a>
4. Authors

<a name="Licence"></a>
5. Licence

<a name="References"></a>
6. References
