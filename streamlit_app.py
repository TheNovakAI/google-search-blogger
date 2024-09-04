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

        if combined_content.strip():  # Ensure that there is some content to return
            st.write(f"Scraped content from {url} (Preview):\n", combined_content[:500] + "...")  # Show a preview
            if st.checkbox(f"Show full scraped content for {url}", False):
                st.text_area(f"Full scraped content from {url}", combined_content, height=300)
            return {
                'content': combined_content,
                'meta_tags': meta_tags,
                'url': url
            }
        else:
            st.warning(f"No relevant content found on {url}.")
            return None
    except Exception as e:
        st.error(f"An error occurred while scraping {url}: {e}")
        return None

def extract_verbatim_with_gpt(content_data, search_topic):
    """Extracts verbatim details relevant to the search topic using GPT-4o-mini."""
    system_prompt = """
    You are an AI assistant tasked with identifying and preserving all relevant details related to a given search topic from web content. 
    Extract the most meaningful information verbatim, ensuring that any quotes, statistics, or key phrases can be accurately cited later.
    """

    user_prompt = f"""
    Search Topic: {search_topic}
    URL: {content_data['url']}
    Extracted Content: {content_data['content']}
    
    Please extract the relevant details verbatim from the provided content that relates directly to the search topic. 
    Ensure that any key phrases, quotes, or statistics are preserved exactly as they appear so they can be used for direct quotations and citations in future blog generation.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=15000  # Set within model's limit for verbatim extraction
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred while extracting verbatim content with GPT-4o-mini: {e}")
        return None

def create_blog_post(summaries, search_topic):
    """Combines multiple refined summaries into a unique, engaging, and well-cited blog post using GPT-4o."""
    system_prompt = """
    You are a professional blog writer. Combine the provided refined summaries into a single cohesive, engaging, and informative blog post.
    Ensure the post has a unique tone, a logical flow, uses top keywords from the sources optimally, includes an introduction, key points from all sources, and a conclusion. 
    Cite all sources and references used, and make sure any quotes or verbatim text is clearly indicated and attributed.
    Use the provided search topic as the title and basis for the description of the blog post.
    """

    user_prompt = f"""
    Search Topic: {search_topic}
    Refined Summaries (verbatim where relevant): {summaries}
    
    Please combine these summaries into a single well-written blog post that is unique, informative, and optimally uses keywords. 
    Use the search topic as the title and description, and ensure to cite all referenced content. Preserve any verbatim text exactly and attribute it properly.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000  # Increased token limit for blog post generation
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred while creating the blog post with GPT-4o: {e}")
        return None

# Streamlit UI
st.title("AI-Powered Blog Post Generator")

search_topic = st.text_input("Enter a search topic or keywords:")

if st.button("Generate Blog Post"):
    # Step 1: Perform web search
    search_results = search_web(search_topic)

    if search_results:
        # Extract URLs from search results
        urls = [result['link'] for result in search_results.get('organic_results', []) if 'link' in result]

        # Progress indicator
        progress_bar = st.progress(0)
        total_urls = len(urls[:20])
        st.write(f"Processing {total_urls} links...")

        # Step 2: Scrape and parse relevant content from URLs
        parsed_summaries = []
        for index, url in enumerate(urls[:20]):  # Limit to the top 20 URLs
            st.write(f"Scraping and processing URL {index + 1} of {total_urls}: {url}")
            content_data = scrape_relevant_content(url)
            if content_data:
                extracted_summary = extract_verbatim_with_gpt(content_data, search_topic)
                if extracted_summary:
                    parsed_summaries.append(extracted_summary)
            # Update progress
            progress_bar.progress((index + 1) / total_urls)

        if parsed_summaries:
            # Step 3: Combine parsed summaries into a unique and cohesive blog post
            st.write("Combining content into a blog post...")
            blog_post = create_blog_post(parsed_summaries, search_topic)
            if isinstance(blog_post, str) and "An error occurred" in blog_post:
                st.error(blog_post)
            else:
                st.markdown(blog_post)
        else:
            st.error("No valid content could be extracted from the provided URLs.")
    else:
        st.error("Failed to perform web search.")
