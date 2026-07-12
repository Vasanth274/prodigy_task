from flask import Flask, request, render_template_string
import os

from PIL import Image

import torch
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Running on:", device)

cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

loader = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

def load_image(path):

    image = Image.open(path).convert("RGB")

    image = loader(image).unsqueeze(0)

    return image.to(device,dtype=torch.float)

unloader = transforms.ToPILImage()

def save_image(tensor,path):

    image=tensor.cpu().clone()

    image=image.squeeze(0)

    image=unloader(image)

    image.save(path)

html="""

<!DOCTYPE html>

<html>

<head>

<title>Neural Style Transfer</title>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:Arial;
}

body{

background:linear-gradient(135deg,#4facfe,#00f2fe,#667eea);

display:flex;

justify-content:center;

align-items:center;

min-height:100vh;

padding:30px;

}

.container{

width:800px;

background:rgba(255,255,255,.15);

backdrop-filter:blur(10px);

padding:40px;

border-radius:20px;

color:white;

box-shadow:0 10px 30px rgba(0,0,0,.3);

}

h1{

text-align:center;

margin-bottom:20px;

}

input{

width:100%;

padding:15px;

margin:10px 0;

border:none;

border-radius:10px;

}

button{

width:100%;

padding:15px;

margin-top:20px;

background:#0072ff;

color:white;

font-size:18px;

border:none;

border-radius:10px;

cursor:pointer;

}

button:hover{

background:#0055cc;

}

img{

width:100%;

margin-top:25px;

border-radius:15px;

}

</style>

</head>

<body>

<div class="container">

<h1>🎨 Neural Style Transfer</h1>

<form method="POST" enctype="multipart/form-data">

<label>Content Image</label>

<input type="file" name="content" required>

<label>Style Image</label>

<input type="file" name="style" required>

<button>Generate Styled Image</button>

</form>

{% if image %}

<img src="{{image}}">

{% endif %}

</div>

</body>

</html>

"""

def gram_matrix(input):

    batch_size, channels, height, width = input.size()

    features = input.view(channels, height * width)

    G = torch.mm(features, features.t())

    return G.div(channels * height * width)

class ContentLoss(torch.nn.Module):

    def __init__(self, target):

        super().__init__()

        self.target = target.detach()

    def forward(self, input):

        self.loss = torch.nn.functional.mse_loss(input, self.target)

        return input
    
class StyleLoss(torch.nn.Module):

    def __init__(self, target_feature):

        super().__init__()

        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):

        G = gram_matrix(input)

        self.loss = torch.nn.functional.mse_loss(G, self.target)

        return input

content_layers = ['conv_4']

style_layers = [

'conv_1',

'conv_2',

'conv_3',

'conv_4',

'conv_5'

]

def get_style_model_and_losses(

cnn,

style_img,

content_img

):

    model = torch.nn.Sequential()

    content_losses = []

    style_losses = []

    i = 0

    for layer in cnn.children():

        if isinstance(layer, torch.nn.Conv2d):

            i += 1

            name = f'conv_{i}'

        elif isinstance(layer, torch.nn.ReLU):

            name = f'relu_{i}'

            layer = torch.nn.ReLU(inplace=False)

        elif isinstance(layer, torch.nn.MaxPool2d):

            name = f'pool_{i}'

        elif isinstance(layer, torch.nn.BatchNorm2d):

            name = f'bn_{i}'

        else:

            continue

        model.add_module(name, layer)

        if name in content_layers:

            target = model(content_img).detach()

            content_loss = ContentLoss(target)

            model.add_module(

                "content_loss_"+str(i),

                content_loss

            )

            content_losses.append(content_loss)

        if name in style_layers:

            target_feature = model(style_img).detach()

            style_loss = StyleLoss(target_feature)

            model.add_module(

                "style_loss_"+str(i),

                style_loss

            )

            style_losses.append(style_loss)

    return model, style_losses, content_losses
    
def get_input_optimizer(input_img):

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    return optimizer

def run_style_transfer(

    cnn,

    content_img,

    style_img,

    input_img,

    num_steps=150,

    style_weight=1000000,

    content_weight=1

):

    model, style_losses, content_losses = get_style_model_and_losses(

        cnn,

        style_img,

        content_img

    )

    optimizer = get_input_optimizer(input_img)

    run = [0]

    while run[0] <= num_steps:

        def closure():

            input_img.data.clamp_(0,1)

            optimizer.zero_grad()

            model(input_img)

            style_score = 0

            content_score = 0

            for sl in style_losses:

                style_score += sl.loss

            for cl in content_losses:

                content_score += cl.loss

            loss = style_score * style_weight + content_score * content_weight

            loss.backward()

            run[0] += 1

            if run[0] % 25 == 0:

                print(

                    "Step:",

                    run[0],

                    " Style:",

                    style_score.item(),

                    " Content:",

                    content_score.item()

                )

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0,1)

    return input_img

def create_stylized_image(

    content_path,

    style_path,

    output_path

):

    content_img = load_image(content_path)

    style_img = load_image(style_path)

    input_img = content_img.clone()

    output = run_style_transfer(

        cnn,

        content_img,

        style_img,

        input_img

    )

    save_image(output, output_path)

@app.route("/", methods=["GET", "POST"])
def home():

    image = None

    if request.method == "POST":

        content = request.files["content"]
        style = request.files["style"]

        content_path = os.path.join(
            "uploads",
            "content.jpg"
        )

        style_path = os.path.join(
            "uploads",
            "style.jpg"
        )

        output_path = os.path.join(
            "outputs",
            "styled.png"
        )

        content.save(content_path)
        style.save(style_path)

        print("Generating image... Please wait.")

        create_stylized_image(
            content_path,
            style_path,
            output_path
        )

        image = "/" + output_path

    return render_template_string(
        html,
        image=image
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )
