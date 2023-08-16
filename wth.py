import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

load_dotenv()

# Define a function to fetch recent trends from the specified URL

user_info = "Name: temp, Age: 30, Country: india, Gender: maleHouse of Pataudi Men Black Printed Straight Kurta,The White Willow Unisex Off-White Therapedic Memory Foam PillowCategory: Trends The Mesh Flat Is the New Face of Impractical Footwear Category: Trends Overalls Are Always a Good Choice Category: Trends It’s a Jorts Summer Category: Trends Denim Is Doing the Most These Days Category: Trends Can Capris Be Cute? Category: Trends The Exposed Bra Trend Is Officially Back Category: Style What Is Mermaidcore? Category: Style Indigenous Designers To Support On Earth Day and Beyond Category: Trends How To Dress Sustainably for Festival Season Category: Trends What Do Long Denim Skirts Say About the Economy? Category: Trends Everyone Will Be Wearing Short Suits This Spring Category: Trends Fashion Is Going Through a Minimalist Vibe Shift Category: Trends Get Up to Speed on Motorsport Style Category: Trends The Case for Dressing Like Your Grandmother Category: Shopping Mary Janes Are All the Way Back Category: Style Prepare to See Rosettes Everywhere Category: Style Romcom Core Is About Romanticizing Your Life Category: Style Cartoon Fashion Is In Category: Style I’m Sorry, But the Peplum Is Back Category: Style I Tried Wearing Tights As Pants The Resort 2024 Trend Report: Capes Swirl, Hems Are Unpredictable, and the Waist Stays in Focus The Fall 2023 Couture Trend Report: Beauty and Body Anxiety Come Together at the Shows The Top Street Style Trends From the Fall 2023 Couture Shows in Paris 10 Trends from the Spring 2024 Men’s Collections—Boyish Suits, Sheer Everything, Massive Pants, and More 10 Menswear Street Style Trends From the Spring 2024 Shows for Your Very Own Hot Boy Summer For the Most Directional Fall 2023 Menswear, Don’t Miss the Women’s Shows The Fall 2023 Jewelry Trend Report Say Hello to Real Life: The 11 Accessory Trends From Fall 2023 You’ll Actually Wear The Street Style Trends That Defined the Fall 2023 Season"


def get_recent_trends(url):
    trends = []

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Find all trend title elements within the specified container
            trend_elements = soup.select('section.post-feed a[aria-label]')
            for trend_element in trend_elements:
                trend_text = trend_element['aria-label']
                trends.append(trend_text)
    except requests.exceptions.RequestException as e:
        print("An error occurred while fetching the page:", e)

    return trends


def get_recent_trends2(url):
    response = requests.get(url)
    html_content = response.content

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Find the container element
    container = soup.find("div", class_="CarouselListWrapper-jhefUk iWgJZu")

    # Find all h3 elements within the container
    h3_elements = container.find_all(
        "h3", class_="SummaryItemHedBase-hiFYpQ gcQpFI summary-item__hed")

    # Extract the plain text from the h3 elements and store in a list
    trends = [h3_element.get_text(strip=True) for h3_element in h3_elements]

    return trends


# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="testing.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=10)

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

5/ Make it so that the results can be formatted through the categories for title, description, link, image, brand, cost, in stock or out of stock and each of these should be in a different line

Below is a message you need to understand and find best answers for
{message}

Here is a list of products you will use for this purpose
{best_practice}

these are the current trends make sure to consider these if the request is open ended
{trends_string}

Please write the best response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice", "trends_string"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message, trends_string):
    best_practice = retrieve_info(message)
    response = chain.run(
        message=message, best_practice=best_practice, trends_string=trends_string)
    return response


# 5. Build an app with streamlit
def main():

    trends_url1 = "https://fashionmagazine.com/category/style/trends/"
    trends_url2 = "https://www.vogue.com/fashion/trends"
    recent_trends = get_recent_trends(trends_url1)
    recent_trends2 = get_recent_trends2(trends_url2)
    recent_trends += recent_trends2
    trends_string = "\n".join(recent_trends)
    trends_string += user_info

    st.set_page_config(
        page_title="Cloth Recommendor", page_icon=":bird:")

    st.header("Clothes recommendation Generator :bird:")
    message = st.text_area("What are you looking for ?")

    if message:
        st.write("Generating most similar products to the search...")

        result = generate_response(message, trends_string)

        st.info(result)

    # Fetch recent trends from the specified URL
    st.write("Recent Fashion Trends:")
    st.write(trends_string)


if __name__ == '__main__':
    main()
