import dash
from dash import dcc, html, Input, Output, State
import base64
import requests
import io

app = dash.Dash(__name__)
API_URL = "http://127.0.0.1:8010/api/predict/"
app.title = "Breast Cancer Segmentation"

# ===== 页面布局 =====
app.layout = html.Div([
    html.H1("🧬 Breast Cancer Segmentation", style={
        'textAlign': 'center',
        'fontWeight': 'bold',
        'fontSize': '40px',
        'marginTop': '30px',
        'marginBottom': '30px',
        'background': 'linear-gradient(to right, #00BFFF, #8A2BE2)',
        'WebkitBackgroundClip': 'text',
        'WebkitTextFillColor': 'transparent'
    }),

    dcc.Upload(
        id='upload-image',
        children=html.Div([
            html.Span("📤 Drag and Drop or ", style={'fontSize': '36px'}),
            html.A("Select Image", style={'fontSize': '36px', 'fontWeight': 'bold'})
        ]),
        style={
            'width': '70%',
            'height': '40vh',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'center',     # 垂直居中
            'alignItems': 'center',         # 水平居中
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '12px',
            'textAlign': 'center',
            'margin': '0 auto 40px auto',
            'backgroundColor': '#f8f9fb',
            'cursor': 'pointer',
            'boxShadow': '0 4px 10px rgba(0,0,0,0.08)'
        },
        multiple=False
    ),

    html.Div([
        html.Div(id='output-image', className='card'),
        html.Div(id='output-mask', className='card'),
        html.Div(id='output-info', className='card')
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'flexWrap': 'wrap',
        'gap': '20px',
        'paddingLeft': '15%',
        'paddingRight': '15%',
        'marginBottom': '60px'
    })
], style={
    "backgroundImage": "url('/assets/background.png')",
    "backgroundSize": "cover",
    "backgroundAttachment": "fixed",
    "backgroundRepeat": "no-repeat",
    "backgroundPosition": "center",
    "minHeight": "100vh",
    "paddingBottom": "60px"
})

# ===== 卡片样式 =====
CARD_STYLE = {
    'backgroundColor': 'rgba(255, 255, 255, 0.92)',
    'padding': '20px',
    'borderRadius': '15px',
    'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.08)',
    'width': '320px',
    'textAlign': 'center',
    'verticalAlign': 'top',
    'minHeight': '420px',
    'backdropFilter': 'blur(3px)'
}

# ===== 工具函数：base64 转文件 =====
def base64_to_file(base64_str, filename):
    content = base64_str.split(',')[1]
    return (filename, io.BytesIO(base64.b64decode(content)), 'image/png')

# ===== 回调：处理上传预测并显示 =====
@app.callback(
    [Output('output-image', 'children'),
     Output('output-mask', 'children'),
     Output('output-info', 'children')],
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return None, None, None

    files = {'image': base64_to_file(contents, filename)}

    try:
        response = requests.post(API_URL, files=files)
        response.raise_for_status()
        result = response.json()
        mask_img = result.get('mask')
        lesions = result.get('lesions', [])
        accuracy = result.get('accuracy', None)
    except Exception as e:
        return html.Div([
            html.H4("❌ Prediction failed."),
            html.Pre(str(e))
        ], style=CARD_STYLE), None, None

    lesion_info = [
        html.Li(
            f"Lesion {i+1}: (x={lesion['x']}, y={lesion['y']}), "
            f"size={lesion['width']}×{lesion['height']}, area={lesion['area']} px²"
        )
        for i, lesion in enumerate(lesions)
    ]

    acc_display = html.P(
        f"🎯 Accuracy: {accuracy * 100:.2f}%" if accuracy is not None else "Accuracy: N/A",
        style={'color': '#00BFFF', 'fontWeight': 'bold'}
    )

    return (
        html.Div([
            html.H4("🖼️ Original Image", style={'fontSize': '22px', 'fontWeight': 'bold'}),
            html.Img(src=contents, style={'width': '100%', 'borderRadius': '10px'})
        ], style=CARD_STYLE),

        html.Div([
            html.H4("🧪 Predicted Mask", style={'fontSize': '22px', 'fontWeight': 'bold'}),
            html.Img(src=mask_img, style={'width': '100%', 'borderRadius': '10px'})
        ], style=CARD_STYLE),

        html.Div([
            html.H4("📊 Prediction Info", style={'fontSize': '22px', 'fontWeight': 'bold'}),
            acc_display,
            html.Ul(lesion_info)
        ], style=CARD_STYLE)
    )

if __name__ == '__main__':
    app.run(debug=True, port=8050)
