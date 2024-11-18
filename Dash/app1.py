import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import base64
import io
from Dash.NeuralClass import SingleLabelCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('final_model2.pth', map_location=device)
model.eval()
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

label_map = {
    0: 'Flowers',
    1: 'Buds',
    2: 'Fruit',
    3: 'No Evidence',
    4: 'Buds & Flowers',
    5: 'Flowers & Buds',
    6: 'Flowers & Fruit',
    7: 'Flower, Buds & Fruit'
}
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(image).unsqueeze(0)
    return img



def predict(image):
    image = preprocess_image(image).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    return label_map[predicted_class.item()]
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.Div(className='navbar', children=[
        html.Ul([
            html.Li(html.A('Overview', href='/overview')),
            html.Li(html.A('Image Identifier', href='/index')),
            html.Li(html.A('About Us', href='/about_us'))
        ])
    ]),
    html.Header([
        html.H1('Image Identifying Model')
    ]),
    html.Main([
        html.Div(className='container', children=[
            html.Div(className='image-container', children=[
                html.H2('Images'),
                html.Div(id='image-gallery')
            ]),
            html.Div(className='data-analysis-container', children=[
                html.H2('Data Analysis'),
                html.Div(id='data-analysis'),
                html.Div(className='button-container', children=[
                    dcc.Upload(id='upload-excel', children=html.Button('Upload Excel File')),
                    dcc.DatePickerSingle(id='date-input'),
                    html.Button('Get Weather Data', id='get-weather-btn')
                ])
            ])
        ])
    ])
])

about_us_page = html.Div([
    html.H1('About Us'),
    html.Div(className='navbar', children=[
        html.Ul([
            html.Li(html.A('Overview', href='/overview')),
            html.Li(html.A('Image Identifier', href='/index')),
            html.Li(html.A('About Us', href='/about_us'))
        ])
    ]),
    html.Div([
        html.H2('Meet The Team!'),
        html.Div(className='profile', children=[
            html.H2('Mcklye Meyer'),
            html.P('Machine Learning Engineer'),
            html.Img(src='/assets/Images/MMeyer.jpg', className='pic'),
            html.P('I have been coding since 2018...')
        ]),
        html.Div(className='profile', children=[
            html.H2('Micah Arif Samuel'),
            html.P('UX Designer'),
            html.Img(src='/assets/Images/MASamuel.jpg', className='pic'),
            html.P('insert paragraph about yourself')
        ]),
        html.Div(className='profile', children=[
            html.H2('Michael Pautz'),
            html.P('Machine Learning Engineer/Data Analyst'),
            html.Img(src='/assets/Images/MPautz.jpg', className='pic'),
            html.P('I am 21 years old and a 3rd year BComp student...')
        ]),
        html.Div(className='profile', children=[
            html.H2('Michael-John Smith'),
            html.P('Machine Learning Engineer'),
            html.Img(src='/assets/Images/MJSmith.jpg', className='pic'),
            html.P('I am a 3rd year student at Belgium Campus who enjoys gaming and pool...')
        ]),
        html.Div(className='profile', children=[
            html.H2('Maryam Jhavary'),
            html.P('Web Designer'),
            html.Img(src='/assets/Images/MJhavary.jpg', className='pic'),
            html.P('I\'m a 21 year-old BComp student at Belgium Campus ITversity...')
        ])
    ])
])


overview_page = html.Div([
    html.H1('Overview'),
    html.Div(className='navbar', children=[
        html.Ul([
            html.Li(html.A('Overview', href='/overview')),
            html.Li(html.A('Image Identifier', href='/index')),
            html.Li(html.A('About Us', href='/about_us'))
        ])
    ]),
    html.Div([
        html.H2('Welcome to Our Aloe Ferox Phenology Identifier'),
        html.P('This project aims to identify the different stages of the phenology of the Aloe Ferox using machine learning. Our goal is to develop a system that can accurately classify images of Aloe Ferox plants into their respective phenological stages, providing valuable insights into the plant\'s growth patterns and responses to environmental factors.')
    ]),
    html.Div([
        html.H2('About The Project'),
        html.P('This project is part of the Plant Identification Programme (PID) at Belgium Campus ITversity, and is focused on the phenology of the Aloe Ferox plant. By leveraging machine learning algorithms, we aim to analyze images of the plant and identify the various stages of their development, including buds, flowers, and fruits.')
    ]),
    html.Div(className='feroxImage', children=[
        html.Img(src='/assets/Images/Aloe ferox (Cape Aloe) - World of Succulents.jpg'),
        html.Img(src='/assets/Images/wallpaperflare.com_wallpaper (1).jpg'),
        html.Img(src='/assets/Images/wallpaperflare.com_wallpaper (2).jpg'),
        html.Img(src='/assets/Images/wallpaperflare.com_wallpaper.jpg')
    ]),
    html.Div(className='feroxInfo', children=[
        html.H2('Aloe Ferox Overview'),
        html.P('Aloe Ferox, commonly known as Cape Aloe or bitter aloe, is a plant species indigenous to South Africa. It is found in the Southern Cape, Eastern Cape, and southern parts of KwaZulu-Natal, the Free State, and Lesotho.'),
        html.Div(className='info-grid', children=[
            html.Div(className='phenology', children=[
                html.H3('Phenology'),
                html.P('The growth cycle of Aloe Ferox typically consists of three stages:'),
                html.Ol([
                    html.Li([
                        html.B('Vegetative growth: '),
                        'During this stage, the plant focuses on leaf growth and development.'
                    ]),
                    html.Br(),
                    html.Li([
                        html.B('Flowering: '),
                        'The plant produces flowers, which are typically red or yellow in colour.'
                    ]),
                    html.Br(),
                    html.Li([
                        html.B('Fruit: '),
                        'The plant produces fruits, which contain seeds that can be used for propagation.'
                    ])
                ])
            ]),
            html.Div(className='env-factors', children=[
                html.H3('Environmental Factors'),
                html.P('The growth and phenology of Aloe Ferox are influenced by various environmental factors, including:'),
                html.Ol([
                    html.Li([
                        html.B('Temperature: '),
                        'Aloe Ferox grows optimally in temperatures between 15°C and 25°C.'
                    ]),
                    html.Br(),
                    html.Li([
                        html.B('Rainfall: '),
                        'The plant requires adequate rainfall, with an annual rainfall of at least 400 mm.'
                    ]),
                    html.Br(),
                    html.Li([
                        html.B('Soil: '),
                        'Aloe Ferox prefers well-drained soil with a pH range of 6.0 to 7.0.'
                    ])
                ])
            ]),
            html.Div(className='cultivation-uses', children=[
                html.H3('Cultivation and Uses'),
                html.P('Aloe Ferox is a valuable crop in South Africa, with various uses, including:'),
                html.Ol([
                    html.Li([
                        html.B('Medicinal purposes: '),
                        'The plant contains aloin, a compound with medicinal properties.'
                    ]),
                    html.Br(),
                    html.Li([
                        html.B('Cosmetic industry: '),
                        'Aloe Ferox is used in the production of skincare products and cosmetics.'
                    ]),
                    html.Br(),
                    html.Li([
                        html.B('Food and beverage industry: '),
                        'The plant is used as a natural sweetener and flavoring agent.'
                    ])
                ])
            ])
        ])
    ]),
    html.Div(className='ref', children=[
        html.H2('References'),
        html.P('This information was sourced from:'),
        html.A('Growth of Aloe ferox Mill. at selected sites in the Makana region of the Eastern Cape', 
               href='https://www.researchgate.net/publication/29806740_Growth_of_Aloe_ferox_Mill_at_selected_sites_in_the_Makana_region_of_the_Eastern_Cape', 
               target='_blank', 
               rel='noopener noreferrer')
    ]),
    html.Div(className='footer')
])
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Plant Phenology Classification"), className="text-center")
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id="upload-image",
                children=html.Div(["Drag and Drop or ", html.A("Select an Image")]),
                style={
                    "width": "100%", "height": "60px", "lineHeight": "60px",
                    "borderWidth": "1px", "borderStyle": "dashed",
                    "borderRadius": "5px", "textAlign": "center"
                },
                multiple=False
            ),
            html.Div(id="output-image-upload")
        ], width=6),
        dbc.Col([
            html.H3("Predicted Class"),
            html.Div(id="prediction-output", className="text-center")
        ], width=6)
    ])
])


@app.callback(
    [Output("output-image-upload", "children"),
     Output("prediction-output", "children")],
    [Input("upload-image", "contents")]
)
def update_output(content):
    if content is not None:
        # Decode the image
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))

        # Predict the class
        prediction = predict(image)

        # Display the uploaded image
        children = html.Div([
            html.H5("Uploaded Image"),
            html.Img(src=content, style={"width": "100%", "height": "auto"})
        ])

        return children, html.H3(prediction)

    return None, "No image uploaded"

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/about_us':
        return about_us_page
    elif pathname == '/overview':
        return overview_page
    else:
        return index_page

if __name__ == '__main__':
    app.run_server(debug=True)
