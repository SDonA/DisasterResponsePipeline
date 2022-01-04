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

1. Project Summary <a name="summary"></a>
	The project uses a disaster dataset provided by Figure Eight to build an ML based API that classifies disaster messages into categories such that the messages can be given the right attention by the appropriate agencies. The ML classifier sits at the heart of a web application built with Flask. The application provided an interface for users to enter a message and get a most likely classification of the message.

	This project is a requirement to complete the Udacity Data Science Nanodegree program.

2. Folders and Files <a name="folders_files"></a>
	There are 3 subfolders in the application root directory, viz:
	1. <em>data</em>:
		The data folder contains the csv dataset (provided by Figure Eight) used to build the ML model.

		It also contains an ETL script - process_data.py. This scrips processes the dataset and stores the processed dataset in an SQLite database (DisasterResponse.db)
	2. models	
	3. app	


3. Installation and Execution <a name="InstAndExec"></a>
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


4. Authors <a name="Authors"></a>


5. Licence <a name="Licence"></a>


6. References<a name="References"></a>
