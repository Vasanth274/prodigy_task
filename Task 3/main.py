from flask import Flask, request, render_template_string
import random

app = Flask(__name__)

html = """

<!DOCTYPE html>
<html>

<head>

<title>Markov Chain Text Generator</title>

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

}

.container{

width:800px;
background:rgba(255,255,255,.15);
backdrop-filter:blur(12px);
padding:40px;
border-radius:20px;
box-shadow:0 10px 30px rgba(0,0,0,.3);
color:white;

}

h1{

text-align:center;
margin-bottom:25px;

}

textarea{

width:100%;
height:180px;
padding:15px;
font-size:17px;
border:none;
border-radius:12px;
margin-bottom:20px;
resize:none;

}

button{

width:100%;
padding:15px;
font-size:18px;
background:#0072ff;
color:white;
border:none;
border-radius:12px;
cursor:pointer;

}

button:hover{

background:#0057cc;

}

.output{

margin-top:25px;
background:white;
color:black;
padding:20px;
border-radius:12px;
font-size:18px;
line-height:1.7;

}

</style>

</head>

<body>

<div class="container">

<h1>📝 Markov Chain Text Generator</h1>

<form method="POST">

<textarea
name="text"
placeholder="Enter training text here..."
required></textarea>

<button>
Generate Text
</button>

</form>

{% if output %}

<div class="output">

<b>Generated Text:</b>

<br><br>

{{output}}

</div>

{% endif %}

</div>

</body>

</html>

"""

def generate_markov(text):

    words = text.split()

    if len(words) < 2:
        return "Please enter more text."

    markov = {}

    for i in range(len(words)-1):

        word = words[i]
        next_word = words[i+1]

        if word not in markov:
            markov[word] = []

        markov[word].append(next_word)

    current = random.choice(words)

    sentence = [current]

    for i in range(30):

        if current not in markov:
            break

        current = random.choice(markov[current])

        sentence.append(current)

    return " ".join(sentence)

@app.route("/",methods=["GET","POST"])

def home():

    output=None

    if request.method=="POST":

        text=request.form["text"]

        output=generate_markov(text)

    return render_template_string(html,output=output)

if __name__=="__main__":

    app.run(debug=True)