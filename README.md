# Forest Fire Prediction Project

This project is designed to predict forest fires using machine learning algorithms. It provides a web interface for users to upload datasets, preprocess them, run machine learning algorithms, and visualize the results.

## Project Structure

```
forest_fire_prediction_project/
├── forest_fire_prediction_app/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   │   └── __init__.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   ├── views.py
│   └── templates/
│       └── forest_fire_prediction_app/
│           ├── home.html
│           ├── upload_dataset.html
│           ├── preprocess_dataset.html
│           ├── run_ml_algorithm.html
│           ├── show_graphical_report.html
│           └── make_prediction.html
├── forest_fire_prediction_project/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── manage.py
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd forest_fire_prediction_project
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

4. **Run migrations:**
   ```
   python manage.py migrate
   ```

5. **Start the development server:**
   ```
   python manage.py runserver
   ```

## Usage Guidelines

- Navigate to `http://127.0.0.1:8000/` to access the home page.
- Use the provided links to upload datasets, preprocess them, run machine learning algorithms, and view graphical reports.
- Follow the instructions on each page for specific functionalities.

## Requirements

- Python 3.x
- Django
- pandas
- scikit-learn
- matplotlib

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.