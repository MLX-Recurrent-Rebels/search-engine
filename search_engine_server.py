import fastapi
import dummy


app = fastapi.FastAPI()


@app.on_event("startup")
async def startup_event():
  app.state.maybe_model = "Attach my model to some variables"
  app.state.inference_model = dummy.model



@app.on_event("shutdown")
async def startup_event():
  print("Shutting down")


@app.get("/")
def on_root():
  return { "message": "Hello App" }


@app.post("/learn_to_search")
async def on_language_challenge(request: fastapi.Request):

  # The POST request body has a text filed,
  # take it and tokenize it. Then feed it to
  # the language model and return the result.
  text = (await request.json())["text"]
  text = app.state.inference_model('yeah right')
  return text
  

