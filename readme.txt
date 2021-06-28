# PO-233 - 01/2021

A estrutura está como:
paper
    arquivos relativos ao artigo
source
    Módulos e scripts personalizados

model_load.py - Arquivo que carrega os modelos encontrados
model_search.py - Arquivo para configurar parâmetros e salvar o modelo

Passo a passo para configurar o ambiente, após clonar do github:

Em um terminal:

    python -m venv .env

    .env/Scripts/python.exe -m pip install -r requirements.txt
    .env/Scripts/python.exe source/setup.py
