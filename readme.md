# Zero Shot Super Resolution

This repository contains a Keras implementation of the ZSSR project.
It produces state of the art (SOTA) single image super resolution (SISR) outputs
for in-the-wild images with real world degragation and compression artifacts.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine and on MissingLink's cloud.
for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


## Prerequisites
You need Python 3.6.8 on your system to run this example.

To install the dependency:

First, updating pip is advised:
```bash
pip install -U pip
```
You are also strongly recommended to use [virtualenv](https://virtualenv.pypa.io/en/stable/) to create a sandboxed environment for individual Python projects:
```bash
pip install virtualenv
```

Create and activate the virtual environment:
```bash
virtualenv .venv
source .venv/bin/activate
```
## Installing

Clone this repo:
```bash
git clone git@github.com:missinglinkai/ZSSR.git
```
Install dependency libraries:
```bash
pip install -r requirements.txt
```
## Run
Sign up to [MissingLink](https://missinglink.ai/) to view the project on our UI.

### Local
```bash
python main.py
```
### MissingLink with Resource Management
Follow instructions here:
[Resource Management](https://missinglink.ai/docs/resource-management/introduction/)

After setting up Resource Management:

Authenticate your username from the CLI:
```bash 
ml auth init
```
Then simply run the project:
```bash
ml run xp
```
