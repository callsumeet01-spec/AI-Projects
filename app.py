from flask import Flask, render_template, request
from research_agent import search_web, summarize_research

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    summary = None
    if request.method == "POST":
        topic = request.form.get("topic")
        if topic:
            results = search_web(topic)
            if results:
                raw_summary = summarize_research(topic, results)
                summary = "\n".join([line.strip() for line in raw_summary.split("\n") if line.strip()])
            else:
                summary = "No search results found."
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
