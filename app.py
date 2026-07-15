import os

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY is None:
    raise ValueError("Gemini API Key not found.")

genai.configure(api_key=API_KEY)


model = genai.GenerativeModel("gemini-2.5-flash")

def generate_email(user_prompt: str) -> str:

    system_prompt = f"""
You are an expert professional Email Writer.

Your job is to convert the user's request into a complete email.

Instructions:

1. Generate only the email.
2. Keep the tone professional unless specified.
3. Include
   - Subject
   - Greeting
   - Body
   - Closing
4. Don't explain anything.
5. Return only the email.

User Request:

{user_prompt}

"""

    response = model.generate_content(system_prompt)

    return response.text

def main():

    print("=" * 50)
    print("        AI EMAIL GENERATOR")
    print("=" * 50)

    while True:

        prompt = input("\nEnter your email request:\n")

        if prompt.lower() == "exit":
            print("\nGoodbye!")
            break

        print("\nGenerating Email...\n")

        email = generate_email(prompt)

        print("-" * 60)
        print(email)
        print("-" * 60)


if __name__ == "__main__":
    main()