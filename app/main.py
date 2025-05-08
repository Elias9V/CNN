from fastapi import FastAPI
from app.routes import router

app = FastAPI()

# Registrar rutas
app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "API CNN Encoder Activa"}
