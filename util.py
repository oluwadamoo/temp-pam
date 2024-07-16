# pip install mysql-connector-python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
import os
import joblib
from dotenv import load_dotenv
import mysql.connector
import pandas as pd
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('procurement_dev.csv')



def load_credentials():
    #variables from .env file
    load_dotenv()

    # Access environment variables
    host = os.getenv('host')
    user = os.getenv('user')
    password = os.getenv('password')
    database = os.getenv('database')

    return host, user, password, database


def fetch_data_to_dataframe(host, user, password, database, query):
    """
    Fetches data from a MySQL database and returns it as a Pandas DataFrame.

    Parameters:
    - host: str. The hostname of the MySQL server.
    - user: str. The username to use for the connection.
    - password: str. The password to use for the connection.
    - database: str. The name of the database to query.
    - query: str. The SQL query to execute.

    Returns:
    - DataFrame: A Pandas DataFrame containing the data fetched from the database.
    """
    try:
        # Establish a connection to the MySQL database
        db_connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        # Execute the query and return the results as a DataFrame
        return pd.read_sql_query(query, db_connection)

    except mysql.connector.Error as error:
        print(f"Error: {error}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

    finally:
        # Ensure the database connection is closed
        if db_connection.is_connected():
            db_connection.close()


# technicians info
t_query = """
SELECT DISTINCT
    t.id as technician_id,
    wr.id AS work_request_id,
    -- wr.name AS work_request_name,
    -- wr.code AS work_request_code,
    -- wr.description AS work_request_description,
    -- wr.created_at AS work_request_created_at,
    u.name AS technician_name,
    u.phone AS technician_phone
FROM
    vampfi_live.work_requests wr
LEFT JOIN
    vampfi_live.technician_task_work_request ttwr ON wr.id = ttwr.work_request_id
JOIN
    vampfi_live.technician_tasks tt ON ttwr.technician_task_id = tt.id
JOIN
    vampfi_live.technician_technician_task tttt ON tt.id = tttt.technician_task_id
JOIN
    vampfi_live.technicians t ON tttt.technician_id = t.id
JOIN
    vampfi_live.users u ON t.user_id = u.id;"""


# sla info
sla_query = """
SELECT 
    S.id as sla_id, T.id as work_category_id,
    S.name SLAName, T.name Work_Category,
    CASE 
        WHEN S.measurement_id = 1 THEN S.expected_time * 1
        WHEN S.measurement_id = 4 THEN S.expected_time * 24
        WHEN S.measurement_id = 5 THEN S.expected_time * 168
        ELSE NULL
    END AS Sla_time
    
FROM vampfi_live.slas S
LEFT JOIN vampfi_live.trades T ON S.trade_id = T.id
"""


# vendor and material info
material_query = """
SELECT 
    TPWR.work_request_id,
    M.id as material_id,
    M.name AS Material_name,
    M.cost AS Material_cost,
    V.business_name AS Vendor_name,
    V.business_phone AS Vendor_number
FROM 
    vampfi_live.purchase_orders PO
    INNER JOIN vampfi_live.quotations Q ON PO.quotation_id = Q.id
    INNER JOIN vampfi_live.tender_processes TP ON Q.tender_process_id = TP.id
    LEFT JOIN vampfi_live.tender_process_work_request TPWR ON TP.id = TPWR.tender_process_id
    LEFT JOIN vampfi_live.tender_process_items TPI ON TP.id = TPI.tender_process_id
    LEFT JOIN vampfi_live.materials M ON TPI.material_id = M.id
    LEFT JOIN vampfi_live.vendors V ON PO.vendor_id = V.id
"""


# work request info
work_query ="""
SELECT 
    WR.id AS work_request_id,
    WR.code AS work_request_code,
    WR.name AS work_request_description,
    -- WR.description AS work_request_description2,
    WR.created_at,
    WR.updated_at,
    TIMESTAMPDIFF(HOUR, WR.created_at, WR.updated_at) AS Time_taken,
    WR.outsourced AS Outsourced_Status,
    Us.name AS FM,
    JB.name AS Job_Status,
    
    -- WR.job_status_id,
    WR.sla_id,
    Ut.name AS Unit_name
    
FROM vampfi_live.work_requests WR
INNER JOIN vampfi_live.units Ut ON WR.unit_id = Ut.id
INNER JOIN vampfi_live.users Us ON WR.created_by = Us.id
INNER JOIN vampfi_live.job_statuses as JB ON WR.job_status_id = JB.id

WHERE WR.created_at > '2022-12-31' and WR.job_status_id NOT IN ('23')
ORDER by WR.created_at desc

"""



def process_id_columns(df):
    # Loop through each column in the DataFrame
    for col in df.columns:
        # Check if 'id' is in the column name
        if 'id' in col:
            # Convert the column to string
            df[col] = df[col].astype(str)
            # Remove any trailing '.0'
            df[col] = df[col].apply(lambda x: x[:-2] if x.endswith('.0') else x)
    return df


def get_updated_data(save=True, save_as='filename.csv', database = 'vampfi_live'):
    """
    Fetches data from a Vampfi_live database, merges it, and returns the result as a Pandas DataFrame.
    Optionally, the merged data can be saved to a CSV file.

    This function performs several steps:
    - Loads individual tables from the database: technician, SLA, material, and work requests.
    - Merges these tables into a single DataFrame.
    - Optionally saves the merged DataFrame to a CSV file.

    Parameters:
    - save (bool): Determines whether the merged DataFrame should be saved as a CSV file.
    - save_as (str): The filename for the CSV file where the data will be saved if `save` is True. Default is 'filename.csv'.

    Returns:
    - pandas.DataFrame: A DataFrame containing the merged data from the database.


    """

    # load credentials
    host, user, password, database = load_credentials()

    try:
        # Load data from the database into DataFrames
        print('Loading technician table...')
        technician_df = fetch_data_to_dataframe(host, user, password, database, t_query)
        print(f'Technician table loaded. Size: {technician_df.shape}')

        print('Loading SLA table...')
        sla_df = fetch_data_to_dataframe(host, user, password, database, sla_query)
        print(f'SLA table loaded. Size: {sla_df.shape}')

        print('Loading material table...')
        material_df = fetch_data_to_dataframe(host, user, password, database, material_query)
        print(f'Material table loaded. Size: {material_df.shape}')

        print('Loading work requests table...')
        work_df = fetch_data_to_dataframe(host, user, password, database, work_query)
        print(f'Work requests table loaded. Size: {work_df.shape}')

        # Merging the tables into a single DataFrame
        print('Merging tables...')
        combined_df = pd.merge(work_df, sla_df, on='sla_id', how='left')
        combined_df = pd.merge(combined_df, technician_df, on='work_request_id', how='left')
        combined_df = pd.merge(combined_df, material_df, on='work_request_id', how='left')
        print(f'Merged DataFrame size: {combined_df.shape}')
        print('Data merging complete.')

        if save:
            # Save the merged DataFrame to a CSV file
            combined_df.to_csv(save_as, index=False)
            print(f'Data saved to {save_as}.')

    except Exception as e:
        print(f'An error occurred: {e}')


    #ensuring id doesnt have '.0'
    combined_df_clean = process_id_columns(combined_df)

    return combined_df_clean


def load_model():
    # Load saved model and matrix
    tfidf_vectorizer = joblib.load("PAM_vectorizer.joblib")
    tfidf_matrix = joblib.load("PAM_matrix.joblib")
    df = pd.read_csv("pam_data.csv")

    return tfidf_vectorizer, tfidf_matrix, df


def best_in_each_column(df):

    results = {}


    columns_to_process = df.columns.difference(['similarity_score'])


    for col in columns_to_process:
        # Get the mode of the first 5 rows
        most_common = df[col].head(5).value_counts()

        if len(most_common):
            most_common = most_common.idxmax()

        # Check if all values in the top 5 are the same
        if len(set(df[col].head(5))) == 1:
            # If all values are the same, take the value from the row with the highest score
            results[col] = df[col].iloc[0]
        else:
            # Otherwise, take the mode value
            results[col] = most_common

   
    print(results['Work_Category'])
    output = {'data': {
                'materials': {
                    'id': str(results['material_id'])[:-2] if results['material_id'].any() else 'N/A',
                    "name":results['Material_name'] if results['material_id'].any() else 'N/A'
                },
                'outsourced': {
                    'id': str(results['Outsourced_Status']) if results['Outsourced_Status'].any() else 'N/A'
                },
                'sla': {
                    'service_type': results['SLAName'] if results['sla_id'].any() else 'N/A',
                    'service_type_id': str(results['sla_id'])[:-2] if results['sla_id'].any() else 'N/A'
                },
                'technicians': {
                    'id': str(results['technician_id'])[:-2] if results['technician_id'].any() else 'N/A',
                    'name': results['technician_name'] if results['technician_id'].any() else 'N/A'
                },
                'trade': {
                    'id': str(results['work_category_id'])[:-2] if results['work_category_id'].any() else 'N/A',
                    'name': results['Work_Category'] if results['work_category_id'].any() else 'N/A'
                },
                'timeline': {
                    'service_time': str(results['Sla_time'])[:-2] if results['Sla_time'].any() else 'N/A',
                    'service_time_taken': str(results['Time_taken'])[:-2] if results['Time_taken'].any() else 'N/A'
                }

                    }
             }

    return output


def find_similar_descriptions(new_description, top_n=5):

    # Load saved model and matrix
    tfidf_vectorizer, tfidf_matrix, df = load_model()

    # Preprocess the new description
    new_description_preprocessed = new_description.lower().replace(r'\W', ' ')
    new_description_vector = tfidf_vectorizer.transform([new_description_preprocessed])

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(new_description_vector, tfidf_matrix).flatten()

    # Get the top N similar descriptions' indices
    similar_indices = cosine_similarities.argsort()[:-top_n - 1:-1]

    # Return the rows from the DataFrame that correspond to the most similar descriptions
    similar_rows = df.iloc[similar_indices]

    # If you want to include similarity scores as a new column in the returned DataFrame:
    similar_rows['similarity_score'] = cosine_similarities[similar_indices]

    output = best_in_each_column(similar_rows)

    return output







# data = pd.read_csv('procurement_pam.csv')


data['material_name'] = data['material_name'].fillna('')
data['service_name'] = data['service_name'].fillna('')
data['material_cost'] = data['material_cost'].fillna(0)
data['service_cost'] = data['service_cost'].fillna(0)


le_vendor = LabelEncoder()
data['vendor_name_encoded'] = le_vendor.fit_transform(data['vendor_name'])

data['cost'] = data['material_cost'] + data['service_cost']


train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(train_data[['vendor_name_encoded', 'cost']])


def recommend_material_ml(material_name, data, model):
    material_data = data[data['material_name'].str.lower() == material_name.lower()]
    if material_data.empty:
        return "No data found for the given material."

    X = material_data[['vendor_name_encoded', 'cost']]
    distances, indices = model.kneighbors(X)

    knn_indices = indices.flatten()
    knn_indices = [i for i in knn_indices if i < len(data)]

    recommendations = data.iloc[knn_indices]
    recommendations = pd.concat([material_data, recommendations])

    recommendations = recommendations[['material_name','vendor_name', 'vendor_phone', 'material_cost']].drop_duplicates().sort_values(
        by='material_cost').head(7)

    output = [
        {
            "Material name": row['material_name'],
            "Vendor name": row['vendor_name'],
            "Vendor phone number": row['vendor_phone'],
            "Material cost": row['material_cost']
        }
        for index, row in recommendations.iterrows()
    ]
    return output


def recommend_service_ml(service_name, data, model):
    service_data = data[data['service_name'].str.lower() == service_name.lower()]
    if service_data.empty:
        return "No data found for the given service."

    X = service_data[['vendor_name_encoded', 'cost']]
    distances, indices = model.kneighbors(X)

    knn_indices = indices.flatten()
    knn_indices = [i for i in knn_indices if i < len(data)]

    recommendations = data.iloc[knn_indices]
    recommendations = pd.concat([service_data, recommendations])

    recommendations = recommendations[['vendor_name', 'vendor_phone', 'service_cost']].drop_duplicates().sort_values(
        by='service_cost').head(5)

    output = [
        {
            "Vendor name": row['vendor_name'],
            "Vendor phone number": row['vendor_phone'],
            "Service cost": row['service_cost']
        }
        for index, row in recommendations.iterrows()
    ]
    return output
