import openai
import json
import os

from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 1. Our "Test Case"
test_prompt = "Explain how a microwave works to a 5-year-old."

def run_judge_eval():
    # --- STEP 1: Get the "Student" answer ---
    print("Generating student response...")
    student_response = client.chat.completions.create(
        model="gpt-4.1-mini", # The model we are testing
        messages=[{"role": "user", "content": test_prompt}]
    ).choices[0].message.content

    print(f"\n--- STUDENT ANSWER ---\n{student_response}\n")

    # --- STEP 2: The "Teacher" grades the work ---
    print("Teacher is grading...")
    
    # We give the Judge a strict persona and rubric
    judge_prompt = f"""
    You are a professional teacher. Rate the following AI response based on:
    1. Simplicity: Is it easy for a 5-year-old?
    2. Accuracy: Is the science correct?
    
    Student Answer: "{student_response}"
    
    Provide your output in this JSON format:
    {{
        "score": (1-10),
        "reasoning": "your explanation here"
    }}
    """

    judge_review = client.chat.completions.create(
        model="gpt-5-mini", # The "Smarter" Judge
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={ "type": "json_object" } # Ensures we get clean data
    ).choices[0].message.content

    # --- STEP 3: Display results ---
    result = json.loads(judge_review)
    print(f"--- JUDGE RESULT ---")
    print(f"Score: {result['score']}/10")
    print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    run_judge_eval()