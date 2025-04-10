from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import re
import os
import PIL.Image
from google import genai
from dotenv import dotenv_values
from typing import List

app = FastAPI()
allowed_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load environment variables
config = dotenv_values(".env")
key = config.get('GEMINI_API_KEY')
if not key:
    raise ValueError("gemini_key not found in .env file")

# Upload folder
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store garbage descriptions and suggested product
garbage_descriptions: List[str] = []
suggested_product: str = ""

# Configure Gemini API
import google.generativeai as genai
genai.configure(api_key=key)
client = genai.GenerativeModel(model_name="gemini-pro-vision")

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Serves the index.html file.
    """
    # Read the content of index.html
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/upload_image")
async def upload_image(files: List[UploadFile] = File(...)):
    """
    Handles multiple image uploads and processes each image to generate descriptions.
    """
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)

    global garbage_descriptions
    descriptions = []

    for file in files:
        try:
            # Save the uploaded file
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Open the image
            image = PIL.Image.open(file_path)

            # Convert the image to bytes
            image_bytes = BytesIO()
            image.save(image_bytes, format='JPEG')
            image_bytes.seek(0)

            # Define the content prompt
            content = """
If the image contains human or animal biological waste (e.g., poop, urine, blood, hair, nails, body parts, vomit, or a dead animal), respond with: "Not garbage".

Otherwise:
1. Identify the item and classify it into one of these categories: Plastic, Metal, Organic, Paper, Glass, E-waste, Textile, or Unknown.
2. Suggest: "Reuse it"
3. Provide detailed information in this format:
   "Category: <category>"
   "Description: <brief and clear physical description>, Unit: <unit>, Shape: <shape>, Color: <color>"

Make sure the description is precise and consistent to support image regeneration with AI tools. Use reliable, repeatable phrasing to ensure the same or similar images can be recreated.
"""

            # Generate content using the Gemini API
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[content, image]
            )

            description = response.text

            # If it's garbage, store the description
            if "Not garbage" not in description:
                garbage_descriptions.append(description)

            descriptions.append(description)

        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    return {'descriptions': descriptions}

def generate_product_from_waste(descriptions: List[str]):
    """
    Generates a product suggestion from a list of waste descriptions.
    """
    client = genai.GenerativeModel(model_name="gemini-pro")
    prompt = f"""
Using these waste materials: {descriptions}, suggest a simple eco-friendly product that can be made easily.

Include:
1. Product Name  
2. Use/Purpose  
3. How to Make (in 3-4 steps)  
4. Extra Items Needed (like glue, scissors, etc.)  
5. Environmental Benefit (1 line)
"""
    response = client.generate_content(
        contents=[prompt]
    )
    
    input_string = response.text
    
    # Remove Markdown symbols using regex
    output_string = re.sub(r"\*\*|\*", "", input_string)
    return output_string

@app.get("/predict_product")
async def predict_product():
    """
    Predicts a product based on the collected garbage descriptions.
    """
    global suggested_product
    if not garbage_descriptions:
        return JSONResponse({'message': "No garbage detected in uploaded images."})

    # Call your function to generate a product from garbage descriptions
    suggested_product = generate_product_from_waste(garbage_descriptions)

    return {'suggested_product': suggested_product}

@app.get("/generate_product_image")
async def generate_product_image():
    """
    Generates an image of the suggested product using Gemini.
    """
    global suggested_product
    if not suggested_product:
        return JSONResponse({"error": "No product suggested yet. Please run /predict_product first."}), 400
    
    client = genai.GenerativeModel(model_name="gemini-pro-vision")
    
    contents = [f"Generate an image of: {suggested_product}"]
    
    response = client.generate_content(
        contents=contents,
    )
    
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            pass
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            image.save('static/gemini-native-image.png')
            return {"image_url": "/static/gemini-native-image.png"}
        else:
            return JSONResponse({"error": "No image data found"}), 500
    return JSONResponse({"error": "No image data found"}), 500
