from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
from util import find_similar_descriptions, data, recommend_material_ml, recommend_service_ml, knn, test_data
import openai
from openai import OpenAI
import os

api_key = "sk-proj-DtALlWXK3I7nLeqheNacT3BlbkFJXvNh83vOCPtvShVJaRMG"

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

instructions = ("You are a prescriptive agent designed to interact with a file system to offer tailored recommendations for a Facility Management company. Use file data to drive insights and actions. "
                "NOTE: Instead of just explaining what the work request is, provide insights and suggest actions."
                "Given an answer template, analyze file contents to craft an answer that closely matches the template."
                "Understand the file structure and data types contained within the files."
                "Limit your analyses to the most relevant data points, focusing on key insights."
                "Organize and summarize data by relevant categories to highlight the most pertinent information."
                "Analyze only the necessary data related to the question at hand."
                "Ensure accuracy in your data interpretation and revise if discrepancies arise."
                "Base your response solely on the analyzed data and the calculations derived from them. Adhere closely to the provided template."
                "Use Markdown for formatting your response."
                "If a question is not directly related to the data in the files, derive a relevant insight based on available information."
                "Utilize only the files confirmed through your tool interactions for constructing analyses."
            )

assistant = client.beta.assistants.create(
    instructions=instructions,
    name="PAM",
    model="gpt-4-turbo",
    tools=[{"type": "file_search"}],
)


# Path to your predefined file
FILE_PATH = ["./uploads/filenamee.pdf"]

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pre-process the predefined file
# with open(FILE_PATH, "rb") as f:
#     file_stream = f.read()

file_stream = [open(path, "rb") for path in FILE_PATH]


vector_store = client.beta.vector_stores.create(name="PAM")

file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id,
    files=file_stream
)


assistant = client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)

message_file = client.files.create(
    file=open("./uploads/filenamee.pdf", "rb"), purpose="assistants"
)


# vector_store = client.beta.vector_stores.create(name="PAM")


@app.route('/suggest/fm', methods=['POST'])
@cross_origin()
def main():
    datar = request.get_json()
    new_description = datar['description']
    top_n = int(datar.get('top_n', 5))
    
    output = find_similar_descriptions(new_description, top_n)
    
    return output


@app.route('/recommend/material', methods=['POST'])
@cross_origin()
def recommend_material():
    try:
        request_data = request.get_json()
        material_name = request_data.get('material_name')

        if not material_name:
            return jsonify({"error": "material_name is required"}), 400

        material_data = data[data['material_name'].str.lower() == material_name.lower()]

        if material_data.empty:
            return jsonify({"error": f"No data found for material: {material_name}"}), 404

        recommendations = material_data[['material_name', 'vendor_name', 'vendor_phone', 'material_cost']].drop_duplicates().sort_values(
            by='material_cost').head(3)
        result = recommendations.to_dict(orient='records')

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/recommend/service', methods=['POST'])
@cross_origin()
def recommend_service():
    try:
        request_data = request.get_json()
        service_name = request_data.get('service_name')

        if not service_name:
            return jsonify({"error": "service_name is required"}), 400

        service_data = data[data['service_name'].str.lower() == service_name.lower()]

        if service_data.empty:
            return jsonify({"error": f"No data found for service: {service_name}"}), 404

        recommendations = service_data[['service_name', 'vendor_name', 'vendor_phone', 'service_cost']].drop_duplicates().sort_values(
            by='service_cost').head(3)
        result = recommendations.to_dict(orient='records')

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_insight', methods=['POST'])
@cross_origin()
def get_insight():
    description = request.json['description']

    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": description,
                "attachments": [
                    {"file_id": message_file.id, "tools": [{"type": "file_search"}]}
                ],
            }
        ]
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )

    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")

    return jsonify({
        "message": message_content.value,
        # "citations": "\n".join(citations)
    })



if __name__ == '__main__':
    app.run(debug=True, port=4000)
