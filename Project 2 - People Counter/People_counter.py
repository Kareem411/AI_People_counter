import tkinter as tk
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

url = ""
cap = cv2.VideoCapture(url)