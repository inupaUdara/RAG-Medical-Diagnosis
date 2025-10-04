from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def main():
    print("Starting the FastAPI application...")
    
if __name__ == "__main__":
    main()
