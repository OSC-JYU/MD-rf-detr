
# MD-rf_detr

## Introduction

MD-rf_detr allows running rfdetr based models. This service was build to run segmentation model of the Finnisn National Archive.

The codebase is from https://github.com/Kansallisarkisto-AI/rfdetr_trocr_pipeline

model: https://huggingface.co/Kansallisarkisto/rfdetr_textline_textregion_detection_model

NOTE: very experimental!


### Functionality

The API offers three main services:

1. **Detect lines** - Create line polygons.




## API

endpoint is http://localhost:9011/process

Payload is queue message as json file. 





### Example API call 

Run from MD-rf_detr directory:

Detect line polygons:

	curl -X POST -H "Content-Type: multipart/form-data" \
	  -F "message=@test/detect.json;type=application/json" \
	  -F "content=@test/htr3.jpg;type=image/jpeg" \
	  http://localhost:9011/process






