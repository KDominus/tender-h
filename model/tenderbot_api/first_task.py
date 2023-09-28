from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


def send_question(question: str) -> str:
    # Content list
    answers = []

    # Answers writing
    for filename in os.listdir("content"):
        file_path = os.path.join("./content", filename)

        with open(file_path, 'r') as file:
            answers.append(file.read())

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))

    # Convert text to TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(answers)

    # Query processing
    query = question

    # Convert query to TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query])

    # Calculating cosine similarity between query and content
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

    # Get content indexes with sort similarities
    doc_indices = cosine_similarities.argsort()[0][::-1]

    # Send the best answer
    most_similar_ans = answers[doc_indices[0]]

    print("Наиболее подходящий ответ: ")
    print(most_similar_ans)

    return most_similar_ans


def main():
    print(send_question("Что такое котировочная сессия?"))


if __name__ == "__main__":
    main()
