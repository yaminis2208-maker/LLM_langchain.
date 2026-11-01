
import csv
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


# ==========================================
# 1. PROJECT SETUP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

DATASET_DIR.mkdir(exist_ok=True)

llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.2
)

parser = StrOutputParser()


# ==========================================
# 2. FIVE PROMPT TEMPLATES
# ==========================================

prompts = {

    "general": """
You are a helpful College Assistant.
Answer college-related questions in simple English.
Do not invent college-specific information.

Question: {question}
Answer:
""",

    "admission": """
You are a College Admission Assistant.
Explain admission procedures, eligibility and documents.
Do not invent college-specific details.

Question: {question}
Answer:
""",

    "attendance": """
You are a College Attendance Assistant.
Explain attendance, leave procedures and attendance rules.
Do not assume the college's specific attendance policy.

Question: {question}
Answer:
""",

    "library": """
You are a College Library Assistant.
Explain library facilities, book borrowing and library rules.
Do not invent specific library timings.

Question: {question}
Answer:
""",

    "exam": """
You are a College Examination Assistant.
Explain examinations, hall tickets, results and preparation.
Do not invent examination dates.

Question: {question}
Answer:
"""
}


# ==========================================
# 3. CREATE PROMPT TEMPLATES
# ==========================================

prompt_templates = {}

for name, template in prompts.items():

    prompt_templates[name] = PromptTemplate(
        input_variables=["question"],
        template=template
    )


# ==========================================
# 4. BUILD FIVE LANGCHAIN CHAINS
# ==========================================

chatbot_chains = {}

for name, prompt in prompt_templates.items():

    chatbot_chains[name] = (
        prompt | llm | parser
    )


# ==========================================
# 5. CATEGORY DETECTION
# ==========================================

def detect_category(question):

    question = question.lower()

    if any(word in question for word in
           ["admission", "course", "eligibility", "documents"]):

        return "admission"

    elif any(word in question for word in
             ["attendance", "absent", "leave", "percentage"]):

        return "attendance"

    elif any(word in question for word in
             ["library", "book", "borrow", "reading"]):

        return "library"

    elif any(word in question for word in
             ["exam", "examination", "hall ticket",
              "result", "marks"]):

        return "exam"

    else:

        return "general"


# ==========================================
# 6. CSV SAVE FUNCTION
# ==========================================

def save_csv(filename, headers, rows):

    filepath = DATASET_DIR / filename

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=headers
        )

        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# 7. SAVE PROMPT TEMPLATES
# ==========================================

def save_prompts():

    rows = []

    for name, template in prompts.items():

        rows.append({
            "prompt_name": name,
            "prompt_template": template.strip()
        })

    save_csv(
        "prompts.csv",
        ["prompt_name", "prompt_template"],
        rows
    )

    print("Five prompt templates saved successfully!")


# ==========================================
# 8. CHATBOT RESPONSE
# ==========================================

def answer_question(question):

    category = detect_category(question)

    chain = chatbot_chains[category]

    response = chain.invoke({
        "question": question
    })

    return category, response


# ==========================================
# 9. SUMMARIZATION CHAIN
# ==========================================

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a summarization assistant.

Summarize the following text in five
simple and important points.

Text:
{text}

Summary:
"""
)

summary_chain = summary_prompt | llm | parser


# ==========================================
# 10. QUESTION ANSWERING CHAIN
# ==========================================

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a College Question Answering Assistant.

Answer using only the given context.
If the answer is not available, say
"Information is not provided in the context."

Context:
{context}

Question:
{question}

Answer:
"""
)

qa_chain = qa_prompt | llm | parser


# ==========================================
# 11. TEST SUMMARIZATION AND QA CHAINS
# ==========================================

def test_additional_chains():

    college_text = """
    The college provides undergraduate courses
    in Computer Science, Commerce and Mathematics.

    Students attend lectures and practical sessions.
    The library provides books and study materials.

    Internal assessments and semester examinations
    are conducted for students.
    """

    # Summarization

    summary = summary_chain.invoke({
        "text": college_text
    })

    print("\n========== SUMMARIZATION ==========")
    print(summary)

    # Question Answering

    question = "What courses are offered?"

    answer = qa_chain.invoke({
        "context": college_text,
        "question": question
    })

    print("\n========== QUESTION ANSWERING ==========")
    print("Question:", question)
    print("Answer:", answer)

    rows = [
        {
            "chain_name": "Summarization",
            "input": college_text,
            "output": summary
        },
        {
            "chain_name": "Question Answering",
            "input": question,
            "output": answer
        }
    ]

    save_csv(
        "chain_results.csv",
        ["chain_name", "input", "output"],
        rows
    )


# ==========================================
# 12. TEST 20 USER QUERIES
# ==========================================

test_queries = [

    "What is artificial intelligence?",
    "What courses are available in college?",
    "How can I apply for admission?",
    "What documents are required for admission?",
    "What is the eligibility for undergraduate admission?",

    "What is attendance percentage?",
    "How can I improve my attendance?",
    "What should I do if I am absent?",
    "How can I apply for leave?",
    "Why is attendance important?",

    "What facilities are available in a college library?",
    "How can I borrow a book?",
    "What are library rules?",
    "Why is reading important for students?",
    "How can I find study materials?",

    "How should I prepare for semester exams?",
    "What is an internal examination?",
    "What is a hall ticket?",
    "How can I check examination results?",
    "How can I improve my exam performance?"
]


def run_test_queries():

    query_rows = []
    response_rows = []

    print("\n========== TESTING 20 QUERIES ==========")

    for i, question in enumerate(test_queries, start=1):

        print("\nQuery", i, ":", question)

        try:

            category, response = answer_question(question)

            print("Category:", category)
            print("Chatbot:", response)

            query_rows.append({
                "user_query": question,
                "category": category
            })

            response_rows.append({
                "user_query": question,
                "category": category,
                "response": response
            })

        except Exception as error:

            print("Error:", error)

    save_csv(
        "user_queries.csv",
        ["user_query", "category"],
        query_rows
    )

    save_csv(
        "chatbot_responses.csv",
        ["user_query", "category", "response"],
        response_rows
    )

    print("\n20-query testing completed.")


# ==========================================
# 13. ORIGINAL AND IMPROVED PROMPT
# ==========================================

def compare_prompt_versions():

    question = (
        "How can students prepare for semester examinations?"
    )

    # Original prompt

    original_prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer this question: {question}"
    )

    # Improved prompt

    improved_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
You are an experienced college academic assistant.

Answer the question using:
1. Simple English.
2. Five clear points.
3. Practical examples.
4. Short explanations.
5. Student-friendly language.

Question: {question}

Answer:
"""
    )

    original_chain = (
        original_prompt | llm | parser
    )

    improved_chain = (
        improved_prompt | llm | parser
    )

    original_output = original_chain.invoke({
        "question": question
    })

    improved_output = improved_chain.invoke({
        "question": question
    })

    print("\n========== ORIGINAL PROMPT ==========")
    print(original_output)

    print("\n========== IMPROVED PROMPT ==========")
    print(improved_output)

    rows = [
        {
            "chain_name": "Original Prompt",
            "input": question,
            "output": original_output
        },
        {
            "chain_name": "Improved Prompt",
            "input": question,
            "output": improved_output
        }
    ]

    save_csv(
        "prompt_comparison.csv",
        ["chain_name", "input", "output"],
        rows
    )

    print("\nPrompt comparison saved successfully!")


# ==========================================
# 14. INTERACTIVE CHATBOT
# ==========================================

def interactive_chatbot():

    print("\n====================================")
    print("     COLLEGE ASSISTANT CHATBOT")
    print("====================================")

    print("Ask your college-related questions.")
    print("Type 'exit' to stop the chatbot.")

    while True:

        question = input("\nEnter your question: ").strip()

        if question.lower() == "exit":

            print("Thank you for using College Assistant!")
            break

        if not question:

            print("Please enter a question.")
            continue

        try:

            category, response = answer_question(question)

            print("\nCategory:", category)
            print("\nChatbot Response:")
            print(response)

        except Exception as error:

            print("Error:", error)


# ==========================================
# 15. MAIN FUNCTION
# ==========================================

def main():

    print("\n====================================")
    print("       WEEK 10 LANGCHAIN PROJECT")
    print("====================================")

    try:

        save_prompts()

        run_test_queries()

        test_additional_chains()

        compare_prompt_versions()

        interactive_chatbot()

        print("\nProject execution completed!")

    except Exception as error:

        print("\nPROJECT ERROR:", error)

        print("\nCheck Ollama and the installed packages.")


# ==========================================
# 16. PROGRAM EXECUTION
# ==========================================

if __name__ == "__main__":

    main()
