import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="testing.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

# current_trends = "lime color, Split-Hem Leggings, sling bags, denim, stripe, pendants"

template = """
You are a world class cloth recommender
I will share a list of clothes with you with details for the type of cloth you will want to search and you will give me the best answer that corresponds to the best result in the data i have given you
and you will follow ALL of the rules below:

1/ Response should be from the available products

2/ Say i found these products and write about them in natural language i.e. the best matching products from the data i have given

3/ Each of the products in the list has a link make sure to include that in the bottom of you recommendation so that the viewer can click on i

4/ Each of the products also has an image link make sure to also write that below everything as well as image link

5/ Make it so that the results can be formatted through the categories for title, description, link, image, brand, cost, in stock or out of stock

Below is a message you need to understand and find best answers for
{message}

Here is a list of products you will use for this purpose
{best_practice}

these are the current trends make sure to consider these if the request is open ended
lime color, Split-Hem Leggings, sling bags, denim, stripe, pendants

Please write the best response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Cloth Recommendor", page_icon=":bird:")

    st.header("Clothes recommendation Generator :bird:")
    message = st.text_area("What are you looking for ?")

    if message:
        st.write("Generating most similar products to the search...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()
