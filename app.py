import streamlit as st
import pandas as pd
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

# Load CSV data
csv_path = "testing.csv"
data = pd.read_csv(csv_path)

# Define a function to fetch recent trends from the specified URL

if "cart_items" not in st.session_state:
    st.session_state.cart_items = []


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

# Define different "pages" using functions


def my_information_page():
    st.title("My Information")
    st.header("User Information :computer:")

    if st.session_state.user_data_string:
        # st.write(st.session_state.user_data_string)
        user_data_string2 = st.session_state.user_data_string
        # Splitting the user data string into individual data points
        user_data_list = user_data_string2.split(",")
        for data_point in user_data_list:
            st.write(f"- {data_point.strip()}")

    else:
        st.write("User information not available.")

    st.header("Cart Items :shopping_trolley:")

    if hasattr(st.session_state, "cart_items") and len(st.session_state.cart_items) > 0:
        for item in st.session_state.cart_items:
            st.write(f"Title: {item['title']}")
            st.write(f"Link: {item['link']}")
            st.image(item['image'], width=150)
            st.write("---")
    else:
        st.write("No items in the cart.")


def main_page(session):
    st.title("AI Search :gear:")
    st.header("Clothes Recommendation Generator :sparkles:")

    user_info = session.user_data_string
    cart_titles = [item['title'] for item in session.cart_items]
    cart_info = ",".join(cart_titles)
    current_trends = session.trends_string
    user_info += " Cart Info : " + cart_info
    user_info += " Current Trends : " + current_trends

    # Initialize the session state variable if not present
    if "main_page_enter_pressed" not in session:
        session.main_page_enter_pressed = False

    message = st.text_area("What are you looking for ?")

    if message:
        loader = CSVLoader(file_path="testing.csv")
        documents = loader.load()
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(documents, embeddings)

        def retrieve_info(query):
            similar_response = db.similarity_search(query, k=10)
            page_contents_array = [
                doc.page_content for doc in similar_response]
            return page_contents_array

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

        def generate_response(message, user_info):
            similar_products = retrieve_info(message)
            similar_products_text = "\n".join(
                similar_products)  # List of strings
            # Convert to a single string

            template = """
            You are a world class cloth recommender
            I will share a list of clothes with you with details for the type of cloth you will want to search and you will give me the best answer that corresponds to the best result in the data i have given you
            and you will follow ALL of the rules below:
            
            1/ Response should be from the available products
            2/ Say i found these products and write about them in natural language i.e. the best matching products from the data i have given
            3/ Display each of the products one by one display the information for each product in the following order:
            4/ Each of the products also has an image link make sure to also write that as well as image link
            First Display the Image then show product Title,price,link to buy,company,color,material and then the reason why this product was recommended for this user based on the user's personal info and recent trends make sure to display these in different lines then finally show all other product links and titles required to complete an outfit that has this product in it
            Show all the details in seperate lines of course each products image should be the first to show
            
            Below is a message you need to understand and find best answers for
            {message}
            
            Here is a list of products you will use for this purpose
            {similar_products_text}
    
            Show products according to the user's personal data and trends
            {user_info}
            Show the products according to the gender you identified from the user info you have, if the gender is male show outfits made for male only and if its female show outfits for female only unless user has clearly stated in the prompt otherwise.
            Please write the best response using all the above information:
            """

            prompt = PromptTemplate(
                input_variables=["message",
                                 "similar_products_text", "user_info"],
                template=template
            )

            chain = LLMChain(llm=llm, prompt=prompt)

            response = chain.run(
                message=message, similar_products_text=similar_products_text, user_info=user_info)
            return response

        st.write("Generating most similar products to the search...")

        result = generate_response(message, user_info)

        st.info(result)

        session.main_page_enter_pressed = True

    if session.main_page_enter_pressed:
        session.main_page_enter_pressed = False

    # st.write("Recent Fashion Trends:")
    # st.write(str(session.trends_string))

    # Example trends string
    trends_string = str(session.trends_string)

    # Splitting the trends string into individual trends
    trends_list = trends_string.split("Category: Trends")

    # Remove any empty strings from the list
    trends_list = [t.strip() for t in trends_list if t.strip()]

    # Header
    st.title("Fashion Trends :chart_with_upwards_trend:")

    # Display trends as a bullet-pointed list
    st.markdown("### Recent Fashion Trends:")
    for trend in trends_list:
        st.write(f"- {trend}")

    # Display user information
    # st.write(f"User Information: {user_info}")


def profile_page():
    st.title("Profile Page :keyboard:")
    st.write("Please provide your information:")

    if "user_data" not in st.session_state:
        st.session_state.user_data = {
            "username": "", "age": 0, "country": "", "gender": ""}

    st.session_state.user_data["username"] = st.text_input(
        "Username", st.session_state.user_data["username"])
    st.session_state.user_data["age"] = st.number_input(
        "Age", min_value=0, max_value=120, value=st.session_state.user_data["age"])
    st.session_state.user_data["country"] = st.text_input(
        "Country", st.session_state.user_data["country"])
    st.session_state.user_data["gender"] = st.text_input(
        "Gender", st.session_state.user_data["gender"])

    # Combine all user data into a single string
    st.session_state.user_data_string = f"Name: {st.session_state.user_data['username']}, Age: {st.session_state.user_data['age']}, Country: {st.session_state.user_data['country']}, Gender: {st.session_state.user_data['gender']}"


def product_page():
    st.title("Product Page :department_store:")
    st.write("Explore our latest products here!")

    # Display each row of the CSV data
    for index, row in data.iterrows():
        st.write(f"## Product {index + 1}")

        # Display product details
        try:
            st.image(row["images"].split(" | ")[0], width=150)
        except AttributeError:
            pass  # Ignore the error and continue
        st.write(f"**Title:** {row['title']}")
        st.write(f"**Brand:** {row['brand']}")
        st.write(f"**Price:** {row['variant_price']}")
        st.write(f"**Color:** {row['dominant_color']}")
        st.write(f"**Product Type:** {row['product_type']}")
        st.write(f"**Size:** {row['size']}")
        st.write(f"**Is In Stock:** {row['is_in_stock']}")

        # Display link
        st.write(f"**Link:** [{row['link']}]({row['link']})")

        # Display "Add to Cart" button
        button_key = f"add_to_cart_{index}"
        if row['is_in_stock'] == "In Stock":
            if st.button("Add to Cart", key=button_key):
                if "cart_items" not in st.session_state:
                    st.session_state.cart_items = []
                st.session_state.cart_items.append({
                    "title": row['title'],
                    "link": row['link'],
                    "image": row["images"].split(" | ")[0]
                })
                st.write("Added to cart!")
        else:
            st.write("Out of Stock")

        # Add some spacing between products
        st.write("---")


def initialize_session():
    if not hasattr(st.session_state, 'trends_string'):
        trends_url1 = "https://fashionmagazine.com/category/style/trends/"
        trends_url2 = "https://www.vogue.com/fashion/trends"
        recent_trends = get_recent_trends(trends_url1)
        recent_trends2 = get_recent_trends2(trends_url2)
        recent_trends += recent_trends2
        st.session_state.trends_string = "\n".join(recent_trends)

# Create navigation buttons


def main():
    initialize_session()

    st.set_page_config(
        page_title="Cloth Recommender",
        page_icon=":shirt:",  # Replace with a relevant icon
        layout="wide",  # Set layout to wide
        initial_sidebar_state="expanded",  # Expand sidebar by default
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
    body {
        background-color: #f5f5f5; /* Light gray background color for the entire app */
        font-family: 'Helvetica Neue', sans-serif;
    }

    .sidebar .sidebar-content {
        background-color: #ffffff; /* White background color for the sidebar */
        border-right: 1px solid #e0e0e0;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.05);
    }

    .stButton button {
        background-color: #007bff; /* Blue button color */
        color: #ffffff;
        border: none;
        border-radius: 4px;
        padding: 8px 12px;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .stButton button:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }

    /* Add more custom styles here */
    </style>
    """, unsafe_allow_html=True)

    nav_selection = st.sidebar.radio(
        "Navigation", ["Home Page", "AI search", "Product Page", "My Information"])

    # Initialize session state variables if not present
    if "main_page_enter_pressed" not in st.session_state:
        st.session_state.main_page_enter_pressed = False
    if "enter_pressed" not in st.session_state:
        st.session_state.enter_pressed = False

    if nav_selection == "Home Page":
        profile_page()
    elif nav_selection == "AI search":
        st.session_state.main_page_enter_pressed = True
        main_page(st.session_state)
    elif nav_selection == "Product Page":
        product_page()
    elif nav_selection == "My Information":
        my_information_page()


if __name__ == '__main__':
    main()
