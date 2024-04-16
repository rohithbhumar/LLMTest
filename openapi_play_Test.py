import streamlit as st
from openai import OpenAI
from openai_key import openai_key  # Make sure to have a separate file 'openai_key.py' with your API key

# Initialize OpenAI client
client = OpenAI(api_key=openai_key)

# Function to generate summary
def help_rewrite_summary(message):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        # model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a professional resume editor assistant who writes the summary in 3rd person"
            },
            {
                "role": "user",
                "content": message
            },
        ],
        temperature=0.7,
        max_tokens=100,
        top_p=1
    )
    resp = completion.choices[0].message.content
    return resp

# Streamlit app
def main():
    st.title("Resume Summary Generator")

    # Text input for user's resume details
    user_input = st.text_area("Enter resume summary:", height=150)

    if st.button("Generate Summary"):
        if user_input:
            # Generate summary using OpenAI
            summary = help_rewrite_summary(user_input)
            # Display the generated summary
            st.write("🤖 AI Generated Summary:")
            st.write(summary)
        else:
            st.write("Please enter some text for the summary.")

if __name__ == "__main__":
    main()
