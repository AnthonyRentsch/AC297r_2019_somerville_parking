Estimating Parking Capacity in the City of Somerville with Machine Learning
==============================

## Overview 

> We partnered with the City of Somerville to estimate the city's residential off-street parking (i.e., driveway) capacity. To do this we manually labelled a small sample of the 12,000+ residential parcels in the city using satellite and Google Steet View images and then trained a machine learning model to predict the presence of a driveway from assessment data, parking permit data, and geometric features derived from parcel and building footprints. After training this model, we carefully tuned it and derived uncertainty estimates to allow for Somerville to do transportation planning in light of the Green Line Extension and the city's climate change plan. Read more about the project on our [Towards Data Science blog post](https://towardsdatascience.com/machine-learning-for-urban-planning-estimating-parking-capacity-15aabd490cf8).

This repository contains all of the code we wrote to perform our analyses. All of the data we used is publicly available through [SomerStat](https://www.somervillema.gov/somerstat), the City of Somerville's open data portal. This project was completed as a part of Harvard's [AC297r Capstone](https://www.capstone.iacs.seas.harvard.edu/) course in the fall of 2019.

## Installation

To safely use our code, you should create a virtual environment and add it to jupyter notebooks. To do so, perform the following steps:
1. Create virtual environment in this directory if it does not already exist with `virtualenv somerville`
2. Activate virtual env with `source somerville/bin/activate`
3. Install requirements with `pip install -r requirements.txt`
4. Add the virtual environment to jupyter with `python -m ipykernel install --user --name=somerville`

## Project Organization

    ├── README.md
    ├── models
    ├── notebooks
        ├── ...
    ├── requirements.txt
    ├── src
    │   ├── __init__.py
    │   ├── ...
    ├── submissions
    │   ├── final-presentation
    │   ├── final-poster
    │   ├── lighting-talk-1
    │   ├── lighting-talk-2
    │   ├── midterm
    │   ├── milestone-1
    │   ├── milestone-2


## Team Members

- Joshua Feldman
- Lipika Ramawamy
- Anthony Rentsch
- Kevin Rader (mentor)


---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. </small></p>
