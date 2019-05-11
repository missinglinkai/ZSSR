![alt text](https://github.com/missinglinkai/ZSSR/blob/master/ZSSR_1.png)

# Zero Shot Super Resolution

This repository contains a Keras implementation of the ZSSR project.
It produces state of the art (SOTA) single image super resolution (SISR) outputs
for in-the-wild images with real world degragation and compression artifacts.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine and on MissingLink's cloud.
for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


## Prerequisites
You need Python 3.6.8 on your system to run this example.

First, updating pip is advised:
```bash
pip install -U pip
```

## Installing

Clone this repo:
```bash
git clone git@github.com:missinglinkai/ZSSR.git
```
Change directory:
```bash
cd ZSSR
```

You are strongly recommended to use [virtualenv](https://virtualenv.pypa.io/en/stable/) to create a sandboxed environment for individual Python projects:
```bash
pip install virtualenv
```

Create and activate the virtual environment inside the project directory:
```bash
virtualenv .venv
source .venv/bin/activate
```

Install dependency libraries:
```bash
pip install -r requirements.txt
```
* To run local, uncomment "tensorflow" in the requiremnets.txt file before installing.

## Run
Sign up to [MissingLink](https://missinglink.ai/) and follow through the process to view the project on our UI.

Authenticate your username from the CLI:
```bash 
ml auth init
```
### Local
```bash
python main.py
```
Running config for example:
```bash
python main.py --epochs 2000 --subdir 001
```
## MissingLink with Resource Management
Follow instructions here:
[Resource Management](https://missinglink.ai/docs/resource-management/introduction/)

After setting up Resource Management:
Create a data volume through the UI and use its ID number to sync the local dataset:
```bash
ml data sync yourDataVolumeID --data-path ~/ZSSR_Images
```
Edit the ".ml_recipe.yaml" file and fill in your Data Volume ID, Data Volume Version and Organization Name:
```python
command: 'python main.py'
data_volume: yourDataVolumeID
data_query: '@version:yourDataVolumeVersion @path:002/*'
gpu: true
org: 'yourOrganizationName'
```
Then simply run the project:
```bash
ml run xp
```
To change the input image (data query) 
open the ".ml_recipe.yaml" file and edit the DirName after path:
```python
data_query: '@version:yourDataVolumeVersion @path:DirName/*'
```
Optional image directories:
001		002		003		004		005		007		016		019		020		032		034		039		047		052		067		071		098		100
