import sys
from numpy import shape
import csv
import numpy as np
import os

def load_wind():
    # Load wind power data from CSV
    with open('datasets/wind.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    
    # Print initial shapes and maximum value
    print("Initial shape:", shape(rows))
    m = np.ndarray.max(rows)
    print("Maximum value of wind:", m)
    
    # Initialize trX as numpy array instead of list
    trX = None
    
    # Process each column
    for x in range(rows.shape[1]):
        # Extract and reshape data, excluding last 288 points
        train = rows[:-288, x].reshape(-1, 576)
        train = train / 16.0  # Normalize
        
        # Handle first column differently from subsequent ones
        if trX is None:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)
    
    print("Shape TrX:", shape(trX))
    
    # Load labels
    with open('datasets/wind label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    label = np.array(rows, dtype=int)
    print("Label shape:", shape(label))
    
    return trX, label

def load_wind_data_spatial():
    with open('datasets/spatial.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    m = np.ndarray.max(rows)
    print("Maximum value of wind:", m)
    rows = rows / m
    return rows

def load_solar_data():
    # Load solar labels
    with open('datasets/solar label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    labels = np.array(rows, dtype=int)
    print("Labels shape:", shape(labels))
    
    # Load solar power data
    with open('datasets/solar.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    rows = rows[:104832,:]  # Truncate to specified time points
    print("Raw data shape:", shape(rows))
    
    # Reshape and normalize
    trX = np.reshape(rows.T, (-1, 576))  # Reshape for GAN input
    print("Reshaped data shape:", shape(trX))
    m = np.ndarray.max(rows)
    print("Maximum value of solar power:", m)
    
    # Prepare labels
    trY = np.tile(labels, (32, 1))
    trX = trX / m
    
    return trX, trY