import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

import psycopg2
from psycopg2.sql import SQL, Identifier
from psycopg2.extras import execute_values
import getpass
import os
import pandas as pd

# I am using tfidf scores to extract meaningful features to be used for classification, hence 
# I am only writing category and abstract as raw strings to the Database or DataFrame. 
# If I were to write category and the whole tfidf matrix it would become cumbersome and it would take 
# too much space since we would have one feature for each unique lemma present in the whole collection of abstracts.


def connect():
    
    hostname = input("hostname: ")
    port = input("port (press Enter to default to 5432): ")
    username = getpass.getpass("username: ")  # does not show input when typing
    password = getpass.getpass("password: ")
    dbname = input("database: ")

    if port == "":
        port = "5432"  # default port

    print("Don't forget to close the connection when you are done")
    return psycopg2.connect(host=hostname, port=port,
                            user=username, password=password, dbname=dbname)


def create_table(connection, table_name):
    
    cursor = connection.cursor()
    cursor.execute(SQL("CREATE TABLE IF NOT EXISTS {} (arxiv_category text, arxiv_abstract text);").format(Identifier(table_name)))
    cursor.close()
    connection.commit()
    
    
def delete_table(connection, table_name):
    
    cursor = connection.cursor()
    cursor.execute(SQL("DROP TABLE {};").format(Identifier(table_name)))
    cursor.close()
    connection.commit()
    
    
def insert_new_values(new_papers_df, connection, table_name):
    
    cursor = connection.cursor()
    query = SQL("INSERT INTO {table_name} VALUES %s").format(table_name=Identifier(table_name))
    psycopg2.extras.execute_values(cursor, query, zip(new_papers_df.arxiv_category, new_papers_df.arxiv_abstract))
    cursor.close()
    connection.commit()
        
    
def scrape_and_write(categories: list, start_date, end_date, table_name,
                     which_date="submitted", sort_by="submitted", 
                     batchsize=500, iter_sleep=10, csv_folder_path=None, 
                     write_db=False, connection=None):
    
    base = importr("base")
    aRxiv = importr("aRxiv")
    dplyr = importr("dplyr")
    
    robjects.r('''source('scrape.r')''')
    retrieve_papers = robjects.globalenv['retrieve_papers']
            
    # TODO: should be able to transform R data.frame into Pandas dataframe and avoid saving/reading csv
    if csv_folder_path is None:
        raise NotImplementedError
    else:
        df_list = []
        for categ in categories:
            # retrieve_papers returns an R data.frame. For now I am writing and reading a csv insteading of reusing the R data.frame
            category_papers = retrieve_papers(category=categ, start_date=start_date, end_date=end_date,
                                              which_date=which_date, sort_by=sort_by, output_format="data.frame",
                                              batchsize=batchsize, iter_sleep=iter_sleep, 
                                              csv_path=os.path.join(csv_folder_path, f"{categ}_{start_date}-{end_date}.csv"))
            
            category_papers = pd.read_csv(os.path.join(csv_folder_path, f"{categ}_{start_date}-{end_date}.csv"))
            df_list.append(category_papers)
            
            if write_db:
                insert_new_values(new_papers_df=category_papers, connection=connection, table_name=table_name)
            
        df_final = pd.concat(df_list, axis=0, ignore_index=True)
        return df_final
