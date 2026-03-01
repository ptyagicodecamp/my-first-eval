import openai
import csv
import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def run_batch_eval(input_file, output_file):
    results = []

    # 1. Read the CSV file
    with open(input_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            question = row['question']
            expected = row['expected_concept']
            
            print(f"Testing: {question}...")

            # 2. Get Student Answer
            student_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Explain this to a child: {question}"}]
            ).choices[0].message.content

            # 3. Judge the Answer
            judge_prompt = f"""
            Task: {question}
            Student Answer: {student_response}
            Key concept to include: {expected}

            Grade 1-5 on:
            1. Accuracy: Does it mention the key concept?
            2. Simplicity: Is it child-friendly?

            Return JSON: {{"score": 1-5, "explanation": "why"}}
            """

            judge_raw = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"}
            ).choices[0].message.content
            
            grade = json.loads(judge_raw)

            # 4. Store the result
            results.append({
                "Question": question,
                "Student_Answer": student_response,
                "Score": grade['score'],
                "Explanation": grade['explanation']
            })
            
            # Small pause to be nice to the API
            time.sleep(1)

    # 5. Save results to a NEW CSV
    keys = results[0].keys()
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    print(f"\nDone! Results saved to {output_file}")

if __name__ == "__main__":
    run_batch_eval('test_cases.csv', 'eval_results.csv')