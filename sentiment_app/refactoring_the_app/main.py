from fastapi import FastAPI
from router.router_sentiment import router

app = FastAPI()

app.include_router(router)
