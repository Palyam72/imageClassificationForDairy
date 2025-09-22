import pandas as pd
from typing import Dict, Generator
from transformers import pipeline
from PIL import Image
import os

# Load the breed classification model once
try:
    breed_classifier = pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )
except Exception as e:
    print(f"Error: Could not load breed detection model. {e}")
    breed_classifier = None

# Load CSV data files (you'll need to provide these files)
try:
    cow_breeds = pd.read_csv("cow_breeds.csv")
    buff_breeds = pd.read_csv("buff_breeds.csv")
except Exception as e:
    print(f"Error: Could not load CSV data files. {e}")
    # Create sample data for demonstration
    cow_breeds = pd.DataFrame({
        "Breed_Type": ["Holstein-Friesian", "Jersey", "Guernsey"],
        "Cost_Of_Cow_INR": [50000, 45000, 40000],
        "Monthly_Income_INR": [8000, 7500, 7000],
        "Popular_Areas": ["Punjab, Haryana", "Global", "UK, US"],
        "Milk_Per_Day_Litres": [20, 15, 12]
    })
    buff_breeds = pd.DataFrame({
        "Breed_Type": ["Murrah", "Nili-Ravi", "Jaffarabadi"],
        "Cost_per_Buffalo_INR": [60000, 55000, 65000],
        "Monthly_Income_per_Buffalo_INR": [9000, 8500, 9500],
        "Popular_Areas": ["Haryana, Punjab", "Pakistan, Punjab", "Gujarat"],
        "Milk_per_Day_Liters": [18, 16, 20]
    })

def detect_breed(image_path: str, animal_type: str) -> str:
    """
    Detects the specific breed of the animal using the pre-loaded vision model.
    Returns the breed name as a string.
    """
    if not breed_classifier:
        return f"Default {animal_type.capitalize()}"

    # Use appropriate column for candidate labels based on animal type
    if animal_type.lower() == "cow" and cow_breeds is not None:
        labels = cow_breeds["Breed_Type"].tolist()
    elif animal_type.lower() == "buffalo" and buff_breeds is not None:
        labels = buff_breeds["Breed_Type"].tolist()
    else:
        # Fallback to a basic list if CSV data is unavailable
        labels = ["Holstein-Friesian", "Jersey", "Murrah", "Nili-Ravi"] if animal_type.lower() == "cow" else ["Murrah", "Nili-Ravi", "Jaffarabadi", "Bhadawari"]

    if not labels:
        return f"Unknown {animal_type.capitalize()}"

    try:
        image = Image.open(image_path)
        results = breed_classifier(image, candidate_labels=labels)
        
        if results and len(results) > 0:
            detected_breed = results[0]["label"]
            return detected_breed
        else:
            return f"Uncertain {animal_type.capitalize()}"
    except Exception as e:
        print(f"Error during breed detection: {e}")
        return f"Error detecting breed"

def get_breed_insights(breed_type: str, animal_type: str) -> Dict:
    """
    Fetches business insights from the appropriate CSV file based on breed and animal type.
    Returns a dictionary with standardized keys for the frontend.
    """
    try:
        if animal_type.lower() == "cow" and cow_breeds is not None:
            # Filter for the specific breed
            breed_data = cow_breeds[cow_breeds["Breed_Type"] == breed_type]
            if not breed_data.empty:
                record = breed_data.iloc[0]
                return {
                    "breed_type": str(record["Breed_Type"]),
                    "starting_expenditure": f"₹{int(record['Cost_Of_Cow_INR']):,}",
                    "annual_income": f"₹{int(record.get('Monthly_Income_INR', 0) * 12):,}",
                    "farmers_percent": "--",
                    "popular_areas": str(record.get("Popular_Areas", "N/A")),
                    "milk_per_day": f"{record.get('Milk_Per_Day_Litres', 'N/A')} Liters",
                    "monthly_income": f"₹{int(record.get('Monthly_Income_INR', 0)):,}"
                }
        
        elif animal_type.lower() == "buffalo" and buff_breeds is not None:
            # Filter for the specific breed
            breed_data = buff_breeds[buff_breeds["Breed_Type"] == breed_type]
            if not breed_data.empty:
                record = breed_data.iloc[0]
                return {
                    "breed_type": str(record["Breed_Type"]),
                    "starting_expenditure": f"₹{int(record.get('Cost_per_Buffalo_INR', 0)):,}",
                    "annual_income": f"₹{int(record.get('Monthly_Income_per_Buffalo_INR', 0) * 12):,}",
                    "farmers_percent": "--",
                    "popular_areas": str(record.get("Popular_Areas", "N/A")),
                    "milk_per_day": f"{record.get('Milk_per_Day_Liters', 'N/A')} Liters",
                    "monthly_income": f"₹{int(record.get('Monthly_Income_per_Buffalo_INR', 0)):,}"
                }
        
        # If no data found, return default values
        return {
            "breed_type": breed_type,
            "starting_expenditure": "Data not available",
            "annual_income": "Data not available",
            "farmers_percent": "--",
            "popular_areas": "Data not available",
            "milk_per_day": "Data not available",
            "monthly_income": "Data not available"
        }
        
    except Exception as e:
        print(f"Error fetching breed insights: {e}")
        return {
            "breed_type": breed_type,
            "starting_expenditure": "Error fetching data",
            "annual_income": "Error fetching data",
            "farmers_percent": "--",
            "popular_areas": "Error fetching data",
            "milk_per_day": "Error fetching data",
            "monthly_income": "Error fetching data"
        }

def get_insights_stream(animal: str, image_path: str) -> Generator[Dict, None, None]:
    """
    Generator function that yields progress updates and final data.
    This drives the real-time progress bar in the UI.
    """
    # Stage 1: Initialization
    yield {'progress': 5, 'message': 'Initializing analysis...'}
    
    # Stage 2: Breed Detection
    yield {'progress': 30, 'message': 'Analyzing image to detect specific breed...'}
    detected_breed = detect_breed(image_path, animal)
    yield {'progress': 50, 'message': f'Breed detected: {detected_breed}. Fetching business insights...'}
    
    # Stage 3: Data Lookup
    yield {'progress': 75, 'message': 'Retrieving detailed business analytics from database...'}
    insights_data = get_breed_insights(detected_breed, animal)
    
    # Stage 4: Data Processing
    yield {'progress': 90, 'message': 'Formatting and validating results...'}
    
    # Stage 5: Complete
    yield {'progress': 100, 'message': '✅ Analysis complete!', 'data': insights_data}