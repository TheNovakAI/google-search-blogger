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
    url = f"https://serpapi.com/search.json?q={query}&num=20&api_key={serp_api_key}"  # Limit to top 20 results
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"An error occurred during web search: {e}")
        return None

def scrape_relevant_content(url):
    """Scrapes only relevant HTML content and metadata from a webpage."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract key elements like title, headings, main content, and meta tags
        title = soup.title.string if soup.title else ""
        meta_tags = {meta.attrs.get('name', meta.attrs.get('property', '')): meta.attrs.get('content', '')
                     for meta in soup.find_all('meta') if meta.attrs.get('content')}
        headings = [heading.get_text(strip=True) for heading in soup.find_all(['h1', 'h2', 'h3'])]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]

        # Combine relevant text parts while limiting content size
        combined_content = " ".join([title] + headings + paragraphs[:100])  # Limit to first 100 paragraphs

        # Return combined content and meta information
        return {
            'content': combined_content,
            'meta_tags': meta_tags,
            'url': url
        }
    except Exception as e:
        st.error(f"An error occurred while scraping: {e}")
        return None

def parse_html_with_gpt(content_data):
    """Parses the extracted content using GPT-4o-mini to keep only the most relevant parts."""
    system_prompt = """
    You are an AI assistant tasked with refining web content. Your job is to extract meaningful content, including the main text, important keywords, and useful metadata, while excluding ads, menus, and other irrelevant sections.
    Be sure to capture key sources or citations from the content.
    """

    user_prompt = f"""
    Here is the extracted web content: 
    Content: {content_data['content']}
    Meta Tags: {content_data['meta_tags']}
    URL: {content_data['url']}
    
    Extract only the most relevant and meaningful text content, keeping the main points, essential information, and any citations or references. Exclude irrelevant details like ads, menus, or repeated sections.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100000  # Increase token limit for comprehensive parsing
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while parsing content with GPT-4o-mini: {e}"

def create_blog_post(summaries):
    """Combines multiple refined summaries into a unique, engaging, and well-cited blog post using GPT-4o."""
    system_prompt = """
    You are a professional blog writer. Combine the provided refined summaries into a single cohesive, engaging, and informative blog post.
    Ensure the post has a unique tone, a logical flow, uses top keywords from the sources optimally, includes an introduction, key points from all sources, and a conclusion. Cite all sources and references used.
    """

    user_prompt = f"""
    Here are the refined summaries from multiple sources: {summaries}
    Please combine these summaries into a single well-written blog post that is unique, informative, and optimally uses keywords. Include citations for all referenced content.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=10000  # Increased token limit for blog post generation
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while creating the blog post with GPT-4o: {e}"

# Streamlit UI
st.title("AI-Powered Blog Post Generator")

search_topic = st.text_input("Enter a search topic or keywords:")

if st.button("Generate Blog Post"):
    # Step 1: Perform web search
    search_results = search_web(search_topic)

    if search_results:
        # Extract URLs from search results
        urls = [result['link'] for result in search_results.get('organic_results', []) if 'link' in result]

        # Step 2: Scrape and parse relevant content from URLs
        parsed_summaries = []
        for url in urls[:20]:  # Limit to the top 20 URLs
            content_data = scrape_relevant_content(url)
            if content_data:
                parsed_summary = parse_html_with_gpt(content_data)
                if "An error occurred" not in parsed_summary:
                    parsed_summaries.append(parsed_summary)

        if parsed_summaries:
            # Step 3: Combine parsed summaries into a unique and cohesive blog post
            blog_post = create_blog_post(parsed_summaries)
            if isinstance(blog_post, str) and "An error occurred" in blog_post:
                st.error(blog_post)
            else:
                st.markdown(blog_post)
        else:
            st.error("No valid content could be extracted from the provided URLs.")
    else:
        st.error("Failed to perform web search.")
