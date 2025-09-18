# Test server to isolate the issue
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/test")
def test_endpoint():
    return {"status": "ok", "message": "Server is working"}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse("test.html", {"request": request})
    except Exception as e:
        return HTMLResponse(f"<h1>Template Error</h1><p>{str(e)}</p>", status_code=500)

@app.get("/original", response_class=HTMLResponse)
async def read_original(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        return HTMLResponse(f"<h1>Template Error</h1><p>{str(e)}</p>", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)