import os
import csv
import requests
from datetime import datetime
from pathlib import Path

def get_chicago_temperature():
    """Fetch current temperature in Chicago using OpenWeatherMap API"""
    API_KEY = os.getenv('OPENWEATHER_API_KEY')
    CHICAGO_LAT = 41.8781
    CHICAGO_LON = -87.6298
    
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={CHICAGO_LAT}&lon={CHICAGO_LON}&appid={API_KEY}&units=metric'
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['main']['temp']
    except Exception as e:
        print(f"Error fetching temperature: {e}")
        return None

def log_temperature():
    """Log current temperature to CSV file"""
    temp = get_chicago_temperature()
    if temp is None:
        return
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    file_path = data_dir / 'chicago_temps.csv'
    file_exists = file_path.exists()
    
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'temperature_celsius'])
        writer.writerow([current_time, temp])

if __name__ == '__main__':
    log_temperature()