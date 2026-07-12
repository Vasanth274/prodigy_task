from flask import Flask, request, render_template_string
from diffusers import StableDiffusionPipeline
import torch
import os


app = Flask(__name__)

print("Loading AI model... (This may take a few minutes the first time)")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>AI Image Generator</title>

<style>

body{
background:linear-gradient(135deg,#4facfe,#667eea);
font-family:Arial;
display:flex;
justify-content:center;
align-items:center;
min-height:100vh;
}

.container{
width:700px;
background:white;
padding:30px;
border-radius:20px;
text-align:center;
box-shadow:0px 10px 25px rgba(0,0,0,.3);
}

input{
width:95%;
padding:15px;
font-size:18px;
margin-bottom:20px;
border-radius:10px;
border:1px solid gray;
}

button{
padding:15px 35px;
font-size:18px;
background:#007BFF;
color:white;
border:none;
border-radius:10px;
cursor:pointer;
}

button:hover{
background:#0056b3;
}

img{
margin-top:25px;
width:100%;
border-radius:15px;
}

</style>

</head>

<body>

<div class="container">

<h1>🎨 AI Image Generator</h1>

<form method="POST">

<input
type="text"
name="prompt"
placeholder="Describe your image..."
required>

<br>

<button>
Generate Image
</button>

</form>

{% if image %}

<img src="{{image}}">

<br><br>

<a href="{{image}}" download>
<button>Download Image</button>
</a>

{% endif %}

</div>

</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():

    image=None

    if request.method=="POST":

        prompt=request.form["prompt"]

        img = pipe(
    prompt,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

        os.makedirs("static",exist_ok=True)

        path="static/generated.png"

        img.save(path)

        image="/" + path

    return render_template_string(HTML,image=image)

if __name__=="__main__":
    app.run(debug=True)