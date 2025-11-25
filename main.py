import uvicorn
import os

if __name__ == "__main__":
    # Run the FastAPI app
    # reload=True for dev is nice, but might restart bot logic. 
    # Let's use reload=False for stability in "production" feel.
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)
