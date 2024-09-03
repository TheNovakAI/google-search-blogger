import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os

# Load the OpenAI API key from Streamlit secrets
api_key = st.secrets["openai_api_key"]
organization = st.secrets.get("openai_organization", None)  # Optional
project = st.secrets.get("openai_project", None)  # Optional

# Initialize the OpenAI client
client = OpenAI(api_key=api_key, organization=organization, project=project)

# SerpAPI credentials from secrets
serp_api_key = st.secrets["serp_api_key"]

def search_web(query):
    """Searches the web using SerpAPI."""
    url = f"https://serpapi.com/search.json?q={query}&num=30&api_key={serp_api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"An error occurred during web search: {e}")
        return None

def scrape_html(url):
    """Scrapes the entire HTML content from a webpage."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.prettify()
    except Exception as e:
        st.error(f"An error occurred while scraping: {e}")
        return None

def parse_html_with_gpt(html_content):
    """Parses the HTML content using GPT-4o to extract meaningful text."""
    system_prompt = """
    You are an AI assistant tasked with extracting meaningful content from raw HTML. Extract the main text content and ignore ads, menus, footers, and other non-relevant parts.
    """

    user_prompt = f"""
    Here is the raw HTML content: {html_content}
    Extract the main text content, such as article text, main headings, and important points, while ignoring ads, menus, footers, and other irrelevant parts.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000  # Adjusted for parsing tasks
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while parsing HTML with GPT-4o: {e}"

def create_blog_post(summaries):
    """Combines multiple summaries into a single blog post using GPT-4o."""
    system_prompt = """
    You are a professional blog writer. Combine the provided summaries from multiple sources into a single cohesive, engaging, and informative blog post. Make sure the post has a logical flow, includes an introduction, key points from all sources, and a conclusion.
    """

    user_prompt = f"""
    Here are the summaries from multiple sources: {summaries}
    Please combine these summaries into a single well-written blog post.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while creating the blog post with GPT-4o: {e}"

# Streamlit UI
st.title("AI Blog Post Generator")

search_topic = st.text_input("Enter a search topic or keywords:")

if st.button("Generate Blog Post"):
    # Step 1: Perform web search
    search_results = search_web(search_topic)

    if search_results:
        # Extract URLs from search results
        urls = [result['link'] for result in search_results.get('organic_results', []) if 'link' in result]

        # Step 2: Scrape and parse all HTML content from URLs
        parsed_summaries = []
        for url in urls[:30]:  # Limit to the first 30 URLs
            html_content = scrape_html(url)
            if html_content:
                parsed_summary = parse_html_with_gpt(html_content)
                if "An error occurred" not in parsed_summary:
                    parsed_summaries.append(parsed_summary)

        if parsed_summaries:
            # Step 3: Combine parsed summaries into a cohesive blog post
            blog_post = create_blog_post(parsed_summaries)
            if isinstance(blog_post, str) and "An error occurred" in blog_post:
                st.error(blog_post)
            else:
                st.markdown(blog_post)
        else:
            st.error("No valid content could be extracted from the provided URLs.")
    else:
        st.error("Failed to perform web search.")
