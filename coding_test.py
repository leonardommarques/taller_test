import os
import numpy as np
import requests
from openai import OpenAI

import pandas as pd

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

openai = OpenAI()

# preview if api is ok
message = "What is the capital of Brazil"

response = openai.chat.completions.create(
    model="gpt-4o-mini"
    , messages=[
        {"role":"system", "content": 'you only answer in rhymes'}
        , {"role":"user", "content": message}
    ]
)
print(response.choices[0].message.content)


# --------------------------------------------- #
# code test
# --------------------------------------------- #


"""
Develop a Python script that:
1- Reads and preprocesses customer service conversations from a CSV file.
2- Generates embeddings for customer messages using sentence-transformers.
3- Stores embeddings in a vector database (FAISS) for fast similarity search.
4- Implements a function to retrieve the top 3 most relevant responses based on a user query.


ID, customer_message, agent_response
1, "My order hasn’t arrived yet. Can you check?"			, "I’m sorry for the delay! Can you provide your order number?"
2, "I need to reset my password but the link is broken."	, "I understand. Let me generate a new reset link for you."
3, "Do you have a refund policy?"							, "Yes! We offer a 30-day money-back guarantee. Would you like me to process a refund?"
4, "The product I received is defective."					, "I'm sorry to hear that. We can arrange a replacement or ref
4, "The product I received is defective."					, "I'm sorry to hear that. We can arrange a replacement or refund. What would you prefer?"
5, "How can I contact support?"							, "You can reach us via email at support@example.com or call our helpline."
"""

# -------------------------------------------------------------------------------- #
# 2- Generates embeddings for customer messages using sentence-transformers.
# -------------------------------------------------------------------------------- #
da = pd.read_csv('C:/tmp/ttest.txt')


# Code bellow generated using chatgpt
import openai

client = openai.OpenAI()  # Make sure to set OPENAI_API_KEY in your environment

def get_embedding(text: str, model="text-embedding-ada-002") -> list:
    """
    Generates an embedding for the given text using OpenAI's API.
    """
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Example usage
#text = "This is an example sentence."
#embedding = get_embedding(text)
#print(embedding)

# -- get embedings for customer message
da_embedings = da[' customer_message'].apply(lambda x: get_embedding(x))
da['customer_message_embedings'] = da_embedings

# ----------------------------------------------------------------------------- #
# 3- Stores embeddings in a vector database (FAISS) for fast similarity search.
# ----------------------------------------------------------------------------- #

da_embedings.to_csv('C:/tmp/ttest_with_embedings.csv')

# -------------------------------------------------------------------------------------------------------------- #
# 4- Implements a function to retrieve the top 3 most relevant responses based on a user query.
# -------------------------------------------------------------------------------------------------------------- #

def similarity(x, y):
    cosim = cosine_similarity(
    np.array(x).reshape(1, -1)
    , np.array(y).reshape(1, -1)
    )

    return cosim[0]


def get_closest_empeding(user_query, historic_da):
    # historic_da = da
    # user_query = da['customer_message_embedings'][0]
    # user_query = 'what is the capital ob brazil?'

    user_query_embeding = get_embedding(user_query)

    historic_da = historic_da.copy()
    historic_da['sims'] = historic_da['customer_message_embedings'].apply(lambda x: similarity(x, user_query_embeding))

    historic_da = historic_da.sort_values('sims', ascending=False)
    top_3 = historic_da.head(3)

    top_3_list = list(top_3[' agent_response'].values)

    return top_3_list

get_closest_empeding('what is the refund policy?', da)

get_closest_empeding('How can I talk to you?', da)

