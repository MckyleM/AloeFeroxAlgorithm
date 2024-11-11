import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
import pickle


# Load the model 

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Aloe Ferox Phenology Predictor"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload'),
    html.Div(id='prediction-output')
])

def predict_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    class_names = ['flowers', 'buds', 'fruits', 'No Evidence', 'flowers and fruits', 'flowers and buds', 'buds and fruits', 'flowers, fruits and buds']
    return class_names[predicted.item()]

@app.callback(Output('output-image-upload', 'children'),
              Output('prediction-output', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output(content, filename):
    if content is not None:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        
        prediction = predict_image(image)
        
        return html.Div([
            html.H5(filename),
            html.Img(src=content, style={'height': '300px'}),
        ]), html.H3(f"Prediction: {prediction}")
    
    return html.Div(), html.Div()

if __name__ == '__main__':
    app.run_server(debug=True)