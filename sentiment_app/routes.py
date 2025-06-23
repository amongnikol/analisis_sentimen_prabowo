from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model import predict_sentiment

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/predict-form", response_class=HTMLResponse)
def handle_form(request: Request, text: str = Form(...)):
    result = predict_sentiment(text)
    return templates.TemplateResponse("index.html", {"request": request, "text": text, "sentiment": result})
