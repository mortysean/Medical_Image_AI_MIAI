import dash
from dash import dcc, html, Input, Output, State
import base64
import requests
import io

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("🧠 Breast Cancer Segmentation"),
    dcc.Upload(
        id='upload-image',
        children=html.Div(['📤 Drag and Drop or ', html.A('Select Image')]),
        style={
            'width': '60%',
            'height': '80px',
            'lineHeight': '80px',
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '10px',
            'textAlign': 'center',
            'margin': '20px auto',
        },
        multiple=False
    ),
    html.Div(id='output-image'),
    html.Div(id='output-mask')
])

def base64_to_file(base64_str, filename):
    content = base64_str.split(',')[1]
    return (filename, io.BytesIO(base64.b64decode(content)), 'image/png')

@app.callback(
    [Output('output-image', 'children'),
     Output('output-mask', 'children')],
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return None, None

    # 将 base64 图像内容转换为文件上传
    files = {'image': base64_to_file(contents, filename)}

    try:
        response = requests.post("http://127.0.0.1:8010/api/predict/", files=files)
        response.raise_for_status()
        result = response.json()
        mask_img = result['mask']
    except Exception as e:
        return html.Div([f"❌ Error: {e}"]), None

    return (
        html.Div([
            html.H4("Original Image"),
            html.Img(src=contents, style={'width': '300px'})
        ]),
        html.Div([
            html.H4("Predicted Mask"),
            html.Img(src=mask_img, style={'width': '300px'})
        ])
    )

if __name__ == '__main__':
    app.run(debug=True, port=8050)

