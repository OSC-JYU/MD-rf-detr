from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import json
from typing import List, Dict, Any, Union

import numpy as np

from narc.detect import detect_polygons_from_image



app = FastAPI(
    title="RF-DETR models API",
    description="API for RF-DETR models running on MessyDesk",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

OUTPUT_FOLDER = 'output'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.get("/")
async def root():
    return {"message": "RF-DETR models API for MessyDesk"}


@app.post("/process")
async def process_files(message: UploadFile = File(...), content: UploadFile = File(...)):
    if not message or not content:
        raise HTTPException(status_code=400, detail='JSON file and text file are required')
    if message.filename == '' or content.filename == '':
        raise HTTPException(status_code=400, detail='Empty file submitted')
    
    try:
        
        # Parse message JSON
        message_data = await message.read()
        try:
            message_text = message_data.decode('utf-8')
            msg = json.loads(message_text)
            
            # If the parsed result is a string, parse it again to get the actual JSON object
            if isinstance(msg, str):
                msg = json.loads(msg)

        except UnicodeDecodeError as e:
            raise HTTPException(status_code=400, detail=f'Message file encoding error: {str(e)}')
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f'Invalid JSON in message file: {str(e)}')

        # operate based on the task id
        if msg.get("task", {}).get("id") == "line_segmentation":
            response = await detect_polygons(content, msg)
        else:
            return {"response": {}}
        return response

        

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f'Processing failed: {str(e)}')



@app.get("/files/{filename:path}")
def serve_file(filename: str, background_tasks: BackgroundTasks):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail='File not found')

    def remove_file(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"Error deleting file {path}: {e}")

    background_tasks.add_task(remove_file, file_path)
    return FileResponse(file_path, background=background_tasks)



def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(value) for value in obj]
    return obj


async def detect_polygons(content: UploadFile, msg: Dict):
    # generate output uuid
    output_uuid = str(uuid.uuid4())
    # read content as image
    content_data = await content.read()
    original_name = os.path.basename(content.filename) if content.filename else "content"
    stored_filename = f"{output_uuid}_{original_name}"
    stored_path = os.path.join(UPLOAD_FOLDER, stored_filename)
    with open(stored_path, "wb") as upload_file:
        upload_file.write(content_data)
    output_file_path = os.path.join(OUTPUT_FOLDER, f"{output_uuid}.polygons.json")
    response_payload: Dict[str, Any] = {}
    try:
        # detect polygons
        polygons = detect_polygons_from_image(stored_path)
        # write polygons to json file
        serializable_polygons = _to_serializable(polygons)
        with open(output_file_path, "w") as f:
            json.dump(serializable_polygons, f, indent=2)
        response_payload = {
            "response": {
                "type": "stored",
                "uri": [f"/files/{output_uuid}.polygons.json"]
            }
        }
    finally:
        try:
            os.remove(stored_path)
        except OSError:
            pass
    return response_payload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9011) 
