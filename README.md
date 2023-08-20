# Clothing Recommendation System using Generative AI and Langchain

Welcome to the README for the **Clothing Recommendation System** project. This project utilizes Generative AI and Langchain to create a clothing recommendation system that leverages user's personal data, cart items, and recent fashion trends scraped from different fashion websites. The system allows users to interact with the program using natural language to search for different clothing products from a CSV file containing scraped items from MyNtra. The project is built using Python and makes use of technologies like OpenAI, Langchain, Pandas, Streamlit, BeautifulSoup, and more.

## Project Overview

The Clothing Recommendation System aims to provide personalized clothing recommendations based on the user's preferences, cart items, personal information, and the latest fashion trends. The recommendation system utilizes a combination of Generative AI and Langchain technologies to enhance the user experience.

## Technologies Used

- **Python**: The project is primarily developed using Python programming language.
- **OpenAI**: OpenAI's GPT-3.5 Turbo model is used to generate natural language responses and interactions.
- **Langchain**: Langchain is used for managing and processing natural language interactions, including document loading, vector storage, embeddings, and chat models.
- **Pandas**: Pandas is used for data manipulation and analysis with the CSV file.
- **Streamlit**: Streamlit is used to create a user-friendly web interface for interacting with the recommendation system.
- **BeautifulSoup**: BeautifulSoup is used for web scraping and extracting fashion trend data from websites.
- **dotenv**: The `dotenv` library is used for managing environment variables.

## Concepts Utilized

- **Generative AI**: The project employs OpenAI's GPT-3.5 Turbo model to generate natural language responses and interact with users.
- **Web Scraping**: BeautifulSoup is used to scrape fashion trend data from different fashion websites.
- **Prompt Engineering**: The project involves crafting prompts that guide the user interaction and generate meaningful responses.
- **Session State Management**: The Streamlit app manages user session state, including cart items and main page interactions.

## Project Structure

- `main.py`: The main script that runs the Streamlit app and handles user interactions.
- `testing.csv`: The CSV file containing scraped clothing items from MyNtra.
- `templates/`: Contains prompt templates used for generating AI responses.
- `styles.css`: Custom CSS styles used for styling the Streamlit app.

## Getting Started

1. Clone the repository.
2. Install the required Python packages using `pip install -r requirements.txt`.
3. Create an OpenAI account and obtain an API key.
4. Set up your `.env` file with your OpenAI API key.

## How to Run

1. Open a terminal and navigate to the project directory.
2. Run the Streamlit app using the command: `streamlit run main.py`.
3. The app will open in a browser, and you can explore the different pages and functionalities.

## Project Pages

1. **Home Page**: This page allows users to provide their information, including username, age, country, and gender.
2. **AI Search**: Users can interact with the AI system using natural language to search for clothing recommendations based on their preferences, cart items, and fashion trends.
3. **Product Page**: Displays the latest clothing products from the CSV file. Users can view product details and add items to their cart.
4. **My Information**: Displays user information and cart items.

## Acknowledgments

This project is created for educational purposes and demonstrates the integration of AI, web scraping, and user interaction techniques. Special thanks to OpenAI, Langchain, and Streamlit for their tools and libraries that made this project possible.
