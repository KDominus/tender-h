from fastapi import FastAPI
from pydantic import BaseModel

from portal_searcher_v2 import send_question

app = FastAPI()


class Question(BaseModel):
    text: str


@app.post("/get_answer_pdf")
def ask_ai(question: Question):
    response = send_question(question.text, 'content_pdf')
    return {"answer": response}


@app.post("/get_answer_xlsx")
def ask_ai(question: Question):
    response = send_question(question.text, 'content_xlsx')
    return {"answer": response}
