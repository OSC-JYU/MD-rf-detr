
# MD-rf_detr

## Introduction

MD-rf_detr allows running rfdetr based models. This service was build to run segmentation model of the Finnisn National Archive.

The codebase is from https://github.com/Kansallisarkisto-AI/rfdetr_trocr_pipeline

model: https://huggingface.co/Kansallisarkisto/rfdetr_textline_textregion_detection_model

NOTE: very experimental!

## install and run cpu version 

	python -m venv venv
	source venv/bin/activate
	pip install -r requirements_cpu.txt

run

	python api.py



## API

endpoint is http://localhost:9011/process

Payload is queue message as json file. 




### Direct API call 

Run from MD-rf_detr directory:

	curl -X POST -H "Content-Type: multipart/form-data" \
	  -F "message=@test/message.json;type=application/json" \
	  -F "content=@test/htr3.jpg;type=image/jpeg" \
	  http://localhost:9011/process






