import openai
import os

from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 2. Our "Test Set" (Input and Expected Answer)
test_cases = [
    {"input": "I love this new phone!", "expected": "Positive"},
    {"input": "This is the worst cake I've ever had.", "expected": "Negative"},
    {"input": "The weather is okay, I guess.", "expected": "Neutral"}
]

def run_eval():
    score = 0
    
    for case in test_cases:
        print(f"Testing: {case['input']}")
        
        # 3. Get the AI's response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Classify the sentiment of this text as Positive, Negative, or Neutral: {case['input']}"}]
        )
        
        actual_result = response.choices[0].message.content.strip()
        
        # 4. The Comparison (The Grader)
        if actual_result == case['expected']:
            print("✅ Pass!")
            score += 1
        else:
            print(f"❌ Fail! Expected {case['expected']}, but got {actual_result}")
    
    print(f"\nFinal Score: {score}/{len(test_cases)}")

if __name__ == "__main__":
    run_eval()