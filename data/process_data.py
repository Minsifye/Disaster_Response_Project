# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    load raw data from csv file, cleaning and transforming the data
    Args:
    messages_filepath: message csv file name
    categories_filepath: categories csv file name
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner', on='id') # merge datasets
    print('Cleaning data...')
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    categories.columns = list(categories.head(1).transpose()[0].str.split("-", expand=True)[0])

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-", expand=True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)


    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop the original categories column from `df`
    df.drop(columns=['original', 'categories'], axis=1, inplace=True)


    # check number of duplicates
    dup_count = df[df.duplicated()].shape[0]
    print("Number of duplicate messages: ", dup_count)
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    save cleaned data to sql file
    Args:
    df: cleaned and transformed dataframe with messages and categories information
    database_filename: Output DB file name along with filepath
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    """
    Main function to run the ETL pipeline
    Args:
    No arguments passed to this function, but this function checks for
    4 arguments passed to script at run time.
    Arg1:Default script name itself.
    Arg2:messages_filepath - messages.csv file name and path
    Arg3:categories_filepath - categories.csv file name and path
    Arg4:database_filepath - database name with path.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
