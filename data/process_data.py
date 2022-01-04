
"""
Data processing
This module contains functions to read-in the messages and categories csv data files, clean the data and write it
to an SQLite database

Run syntax:
    python process_data.py <path to disaster_messages.csv> <Path to disaster_categories.csv> <path to destination SQLite database>
Run example:
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

Arguments:
    -   disaster_messages.csv: file containing messages
    -   disaster_categories.csv: file containing message categories

"""


# import required libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Reads in the messages and categories csv data files and outputs a single dataframe with the datasets combined
    
    Arguments:
        message_filepath (string): path to the CSV file containing the messages
        categories_filepath (string): path to the CSV File containging the categories
    
    Returns:
        df (pandas dataframe): Pandas dataframe with messages and categories combined.
    """
    # read in the datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge the dataset
    df = pd.merge(messages, categories, how = 'inner', on=['id'])
    return df
    


def clean_data(df):
    """
    Cleans the combined dataset as passed via df.

    Arguments:
        df (pandas dataframe): dataframe with messages and categories combined
    
    Returns:
        df (pandas dataframe): dataframe containing cleaned combined dataset.
    
    """
    # create a dataframe with the  individual category columns
    categories = df['categories'].str.split(pat = ';', expand =True)

    # Use first row of categories to create column names for the categories dataframe
    row = categories.iloc[0]
    category_colnames = list(row.apply(lambda x: x.split('-')[0]))
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to tbe the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)


    # Drop original categories column from the df dataframe since it is no longer needing 
    df = df.drop('categories', axis = 1 )
    
    # concatenate categories and df dataframes and subsequently drop duplicates
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Saves the dataframe df to a SQLite database

    Arguments:
        df (pandas dataframe): Messages and Categories combined and cleaned dataframe
        database_filename (string): name of the destination SQLite Database

    returns: nothing
    """

    database_name = database_filename
    table_name = 'disaster_response_tbl'

    engine = create_engine('sqlite:///' + database_name)
    df.to_sql(table_name, engine, index=False, if_exists = 'replace')
     


def main():
    """
    Main function (entry point) that will call the various functions in this module to: 
        1. read in the relevant
        2. Clean and Transform the data
        3. write the processed data to a database
    """
    
    if len(sys.argv) == 4:

        # extract command line arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] 

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        # notify user when wrong entries are provided
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()