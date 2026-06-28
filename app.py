from flask import Flask, render_template, request, redirect
from chatbot import get_answer

app = Flask(__name__)

chat_history = []

@app.route("/", methods=["GET", "POST"])
def home():
    global chat_history

    if request.method == "POST":
        question = request.form["question"]
        answer = get_answer(question)

        chat_history.append({
            "question": question,
            "answer": answer
        })

    return render_template(
        "index.html",
        chat_history=chat_history
    )


@app.route("/clear", methods=["POST"])
def clear():
    global chat_history
    chat_history.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)