# **Robust Journey Planning: Final Project**

## Overview

This project focuses on creating a robust journey planning algorithm that considers not just the shortest travel times, but also the probability of delays and missed connections. By integrating historical data on public transport delays, the project allows users to plan routes that account for uncertainties in travel, making public transport decisions more reliable.

## How-To Run the Project

To run the project, follow these steps:

1. **Pre-process Data**:  
   Run the `pre_processing.ipynb` notebook to filter relevant data and form edges and nodes for your journey planner.
   
2. **Model Delays**:  
   Execute `delay_modelling.ipynb` to gather historical delay data from the `istdaten` dataset. This will help model delays for more accurate route planning.
   
3. **Visualize Results**:  
   Finally, run `visual.ipynb` to get an overview and visualization of your journey planner’s results.

The project contains `.py` files, which provide utility functions, while the `.ipynb` notebooks contain the code necessary to run the project.

---

## Problem Motivation

Imagine being a regular user of public transport and planning your route based on the operator's schedule. The app might show the quickest route to get to your destination, but it doesn't account for potential delays or missed connections.

In reality, your travel decisions would be affected by the likelihood of arriving on time. For instance, one route might be faster but riskier in terms of delays, while another may take longer but have a higher probability of getting you to your destination on time. Current transport applications don’t account for such risk factors. Our journey planner is designed to improve this by providing robust route planning that includes these uncertainties.

---

## Problem Description

This project builds a _robust_ public transport route planner by reusing the Swiss Federal Railways (SBB) dataset. The objective is to develop an algorithm that not only finds the fastest routes between two stops but also accounts for the uncertainty in delays and missed connections.

Given a desired arrival time, the algorithm computes the fastest route between departure and arrival stops within a specified confidence tolerance. The goal is to answer the following question:

> "What is the fastest route from _A_ to _B_ that will get me to _B_ before time _T_ with at least _Q%_ confidence?"

The output is a list of routes from _A_ to _B_ along with their confidence levels. The routes are sorted from latest to earliest departure time, all arriving at _B_ before time _T_ with a confidence level greater than or equal to _Q_. The routes can also be visualized on a map.

### Key Tasks:
- Model the public transport infrastructure for route planning.
- Build a predictive model using historical delay data.
- Implement a robust route planning algorithm.
- Test and validate the results using scientific methods.
- Implement a visualization to showcase the method, using Jupyter widgets.

---

## Project Details

### **Assumptions and Simplifications:**
- We focus on journeys during typical business hours on weekdays.
- Small walking distances (max 500m) between stops are allowed.
- Routes start and end at known station coordinates (bus/train stops).
- Delays or travel times on different public transport lines are treated as independent.
- The algorithm assumes the traveler will follow the computed route until completion or failure (i.e., a missed connection).
- No penalty is applied for failure, and routes with identical travel times and uncertainty tolerances are considered equal.
- We prioritize routes with the minimum walking distance and fewest transfers.

### **Deliverables:**
1. **Public Transport Network Model**:  
   Use the SBB dataset to model transport routes, stops, and connections.
   
2. **Predictive Delay Model**:  
   Build a model to predict delays using historical data from the `istdaten` dataset.
   
3. **Robust Route Planner**:  
   Implement the route planning algorithm with confidence levels to account for delays.
   
4. **Validation**:  
   Test and validate the route planner using scientific methods, ensuring the accuracy of results.
   
5. **Visualization**:  
   Provide a user-friendly visualization in Jupyter notebooks, allowing the user to interact with the route planning tool.

---

## Dataset Description

We use data from the [Open Data Platform Mobility Switzerland](https://opentransportdata.swiss). The primary datasets include:

### **Actual Data:**
- **istdaten**: Provides historical data on transport trips, including:
  - `BETRIEBSTAG`: Trip date.
  - `FAHRT_BEZEICHNER`: Trip identifier.
  - `LINIEN_ID`: Train or bus line number.
  - `ANKUNFTSZEIT`: Scheduled arrival time at the stop.
  - `AN_PROGNOSE`: Actual arrival time (if available).
  - `ABFAHRTSZEIT`: Scheduled departure time from the stop.

### **Timetable Data:**
- **stops.txt**: Information on public transport stops, including stop names and coordinates.
- **stop_times.txt**: Timetable data detailing scheduled arrival and departure times for each trip.
- **trips.txt**: Data on individual transport trips, including routes and headsigns.
- **calendar.txt**: Defines the service days for each trip.