from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.database import engine, Base
from src.routers import auth_router, patient_router, doctor_router, ai_router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="MedSarthi API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(patient_router.router)
app.include_router(doctor_router.router)
app.include_router(ai_router.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to MedSarthi API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)