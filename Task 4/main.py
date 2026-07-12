from flask import Flask, request, render_template_string
from PIL import Image
import cv2
import numpy as np
import os

app = Flask(__name__)
REPLICATE_API_TOKEN="r8_clEo82pX4bFB5chVWPYpEx3fIjlN5jU4Pjqqn"

os.makedirs("static", exist_ok=True)

html = """

<!DOCTYPE html>

<html>

<head>

<title>Image-to-Image Translation</title>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:Arial,sans-serif;
}

body{

background:linear-gradient(135deg,#4facfe,#00c6ff,#667eea);
display:flex;
justify-content:center;
align-items:center;
min-height:100vh;
padding:30px;

}

.container{

width:900px;
background:rgba(255,255,255,.15);
backdrop-filter:blur(15px);
padding:40px;
border-radius:20px;
box-shadow:0 10px 30px rgba(0,0,0,.3);
color:white;

}

h1{

text-align:center;
margin-bottom:20px;

}

input{

width:100%;
padding:15px;
border-radius:10px;
margin-bottom:20px;

}

button{

width:100%;
padding:15px;
font-size:18px;
background:#0072ff;
color:white;
border:none;
border-radius:10px;
cursor:pointer;

}

button:hover{

background:#0055cc;

}

.images{

display:flex;
justify-content:space-between;
margin-top:30px;
gap:20px;

}

.box{

width:48%;
text-align:center;

}

img{

width:100%;
border-radius:15px;
box-shadow:0 0 15px rgba(0,0,0,.3);

}

h3{

margin-bottom:15px;

}

a{

display:inline-block;
margin-top:20px;
padding:12px 20px;
background:#28a745;
color:white;
text-decoration:none;
border-radius:10px;

}

</style>

</head>

<body>

<div class="container">

<h1>🖼 Image-to-Image Translation</h1>

<form method="POST" enctype="multipart/form-data">

<input type="file" name="image" required>

<button>Translate Image</button>

</form>

{% if output %}

<div class="images">

<div class="box">

<h3>Original Image</h3>

<img src="{{input}}">

</div>

<div class="box">

<h3>Translated Image</h3>

<img src="{{output}}">

<a href="{{output}}" download>Download</a>

</div>

</div>

{% endif %}

</div>

</body>

</html>

"""

@app.route("/",methods=["GET","POST"])

def home():

    input_image=None
    output_image=None

    if request.method=="POST":

        file=request.files["image"]

        input_path="static/input.jpg"
        output_path="static/output.jpg"

        file.save(input_path)

        img=cv2.imread(input_path)

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        invert=255-gray

        blur=cv2.GaussianBlur(invert,(21,21),0)

        invert_blur=255-blur

        sketch=cv2.divide(gray,invert_blur,scale=256)

        cv2.imwrite(output_path,sketch)

        input_image="/"+input_path
        output_image="/"+output_path

    return render_template_string(html,input=input_image,output=output_image)

if __name__=="__main__":

    app.run(debug=True)