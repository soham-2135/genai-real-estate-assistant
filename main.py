import streamlit as st
import requests

st.title("Real Estate Research Tool")

# Sidebar inputs
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

API_URL = "http://backend:8000"

# Process URLs button
if st.sidebar.button("Process URLs"):
    urls = [url for url in (url1, url2, url3) if url]

    if len(urls) == 0:
        st.error("You must provide at least one valid URL")
    else:
        st.info("Processing URLs...")

        try:
            response = requests.post(
                f"{API_URL}/process-urls",
                json={"urls": urls},
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()

                if "steps" in data:
                    for step in data["steps"]:
                        st.write(step)
            else:
                st.error(f"Backend error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Connection error: {e}")

# Question input (FIXED)
query = st.text_input("Ask a question")

if query:
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"query": query}
        )

        if response.status_code == 200:
            data = response.json()

            answer = data.get("answer", "")
            sources = data.get("sources", "")

            st.header("Answer:")
            st.write(answer)

            if sources:
                st.subheader("Sources:")
                for source in sources.split(","):
                    st.write(source)

        else:
            st.error(f"Backend error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Connection error: {e}")