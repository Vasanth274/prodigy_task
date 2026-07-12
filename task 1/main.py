from flask import Flask, request, render_template_string
from transformers import pipeline

app = Flask(__name__)

generator = pipeline("text-generation", model="gpt2")

html = """
<!DOCTYPE html>
<html>
<head>
<title>AI Text Generator</title>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:Arial,Helvetica,sans-serif;
}

body{

background:linear-gradient(135deg,#4facfe,#00f2fe,#667eea);
height:100vh;
display:flex;
justify-content:center;
align-items:center;

}

.container{

width:700px;
background:rgba(255,255,255,.15);
padding:40px;
border-radius:20px;
backdrop-filter:blur(15px);
box-shadow:0 15px 35px rgba(0,0,0,.3);
color:white;

}

h1{

text-align:center;
margin-bottom:20px;

}

input{

width:100%;
padding:15px;
border:none;
border-radius:10px;
font-size:18px;
margin-bottom:20px;

}

button{

width:100%;
padding:15px;
font-size:18px;
background:#00c6ff;
color:white;
border:none;
border-radius:10px;
cursor:pointer;
transition:.3s;

}

button:hover{

background:#0072ff;
transform:scale(1.03);

}

.output{

margin-top:25px;
background:white;
color:black;
padding:20px;
border-radius:10px;
line-height:1.7;

/* Scrollable output */
max-height:300px;
overflow-y:auto;
overflow-x:hidden;
word-wrap:break-word;

}

</style>

</head>

<body>

<div class="container">

<h1>🤖 GPT-2 AI Text Generator</h1>

<form method="POST">

<input
type="text"
name="prompt"
placeholder="Enter your prompt..."
required>

<button type="submit">
Generate Text
</button>

</form>

{% if output %}

<div class="output">

<h2>Generated Text</h2>
<p>{{output}}</p>

</div>

{% endif %}

</div>

</body>
</html>
"""

@app.route("/",methods=["GET","POST"])
def home():

    output=""

    if request.method=="POST":

        prompt=request.form["prompt"]

        result=generator(
            prompt,
            max_length=100,
            truncation=True
        )

        output=result[0]["generated_text"]

    return render_template_string(html,output=output)

if __name__=="__main__":
    app.run(debug=True)