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

		It also contains an ETL script - process_data.py. This scrips processes the dataset and stores the processed dataset in an SQLite database (DisasterResponse.db).	
		There is also an ETL Pipiline Preparation.ipynb notebook file that served as a guide (for code development and testing) to build the process_data.py script. it provided a "scratch board" for testing ideas before implementation in proces_data.py
	2. <em>models</em>:
		The models folder contains a ML Pipeline Preparation.ipynb notebook file that served as a guide (for code development and testing) to build the train_classifier.py script. it provided a "scratch board" for testing ideas before implementation in train_classifier.py	
		Train classifier.py is the script that reads in the dataset from the SQLite database, develops an ML model and stores the model to disk as classifier.pkl (for example)
		classifier.pkl is the trained ML model stored as a pickle file.

	3. <em>app</em>"	
		the app folder contains files pertinent to run the flask web App. it contains run.py which contains the script (routes, ML model import, charts, etc) necessary to run the app. 
		The folder contains a subfolder - templates; this contains the 2 html files with codes for the 2 pages of the application

3. Installation and Execution <a name="InstAndExec"></a>
	1. Dependencies
		- Python 3.8.5+
		- Natural Language Toolkit: NLTK
		- ML Libraries: NumPy, Pandas, Sciki-Learn
		- SQLlite Database library: SQLalchemy
		- Model saving and loading: Pickle
		- Web app and data visualization: Flask, Plotly

	2. Clone Repository
		code base can be cloned from github: git clone https://github.com/SDonA/DisasterResponsePipeline.git

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
	- Success Attoni
	- Team Udacity

5. Licence <a name="Licence"></a>


6. References<a name="References"></a>
	- [Udacity](https://www.udacity.com/): providing boilerplate code and guide to create the webapp
	- [Udacity](https://www.udacity.com/) course material
	- [Figure Eight](https://www.figure-eight.com/) dataset
	- [Medium](https://wwww.medium.com)
	- Documentations of scikit-learn, nltk, pandas, etc.

 






