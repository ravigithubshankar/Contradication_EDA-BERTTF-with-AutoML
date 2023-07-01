#we deploved small web interface model deployment using popular tool called Gradio interface

#let see implementation about that
#first install gradio library through pip install gradio

import gradio as ml

#predictions

def predict_text(hypothesis, premise):
    inputs = bert_encoder([hypothesis], [premise], tokenizer)
    prediction = np.argmax(model.predict(inputs))
    return label_names[prediction]

# Create the Gradio interface
iface = ml.Interface(
    fn=predict_text,
    inputs=["text", "text"],
    outputs="text",
    title="BERT Text Classification",
    description="Predict the label of a text pair using BERT.",
    examples=[
        ["This is an example.", "It is a test."],
        ["Another example.", "It is not a test."]
    ]
)

# Launch the interface
iface.launch()

