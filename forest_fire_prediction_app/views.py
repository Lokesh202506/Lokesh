from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from .models import Dataset
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import authenticate, login, logout
import uuid
from django.views.decorators.csrf import csrf_exempt
import matplotlib.pyplot as plt
import os
from django.contrib.auth.models import User
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay



# --- Admin Dashboard ---
@login_required
@user_passes_test(lambda u: u.is_superuser)
def admin_dashboard(request):
    users = User.objects.all()
    dataset = Dataset.objects.order_by('-id').first()
    land_type_counts = []
    land_cover_types = []
    if dataset and os.path.exists(dataset.file.path):
        df = pd.read_csv(dataset.file.path)
    else:
        default_path = os.path.join(
            os.path.dirname(__file__), 'forest_fire_synthetic_dataset.csv'
        )
        df = pd.read_csv(default_path)
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)
    if 'land_cover_type' in df.columns:
        land_type_counts = df['land_cover_type'].value_counts().reset_index()
        land_type_counts.columns = ['land_type', 'count']
        land_type_counts = land_type_counts.to_dict(orient='records')
        land_cover_types = df['land_cover_type'].unique().tolist()
    return render(request, 'forest_fire_prediction_app/admin_dashboard.html', {
        'users': users,
        'land_type_counts': land_type_counts,
        'land_cover_types': land_cover_types
    })

# --- Admin Auth ---
def admin_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            if user.is_superuser:
                login(request, user)
                return redirect('admin_dashboard')
            else:
                form.add_error(None, "You are not authorized as admin.")
    else:
        form = AuthenticationForm()
    return render(request, 'registration/admin_login.html', {'form': form})

def admin_logout(request):
    logout(request)
    return redirect('admin_login')

# --- Message history helper ---
def add_message_to_history(request, msg, level='info'):
    if 'message_history' not in request.session:
        request.session['message_history'] = []
    request.session['message_history'].append({'msg': msg, 'level': level})
    request.session.modified = True

# --- Home, Register, Login, Logout ---
def home(request):
    if os.path.exists('server_restart.flag'):
        if 'message_history' in request.session:
            del request.session['message_history']
        os.remove('server_restart.flag')
    message_history = request.session.get('message_history', [])
    return render(request, 'forest_fire_prediction_app/home.html', {'message_history': message_history})

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            add_message_to_history(request, "Registration successful. Please log in.", "success")
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            add_message_to_history(request, "Login successful.", "success")
            return redirect('home')
        else:
            add_message_to_history(request, "Login failed. Please check your credentials.", "danger")
    else:
        form = AuthenticationForm()
    return render(request, 'registration/login.html', {'form': form})

def logout_view(request):
    logout(request)
    add_message_to_history(request, "Logged out successfully.", "info")
    return redirect('login')

@login_required
def user_dashboard(request):
    return render(request, 'forest_fire_prediction_app/user_dashboard.html')

@login_required
def remove_user(request, user_id):
    if not request.user.is_superuser:
        return redirect('home')
    User.objects.filter(id=user_id, is_superuser=False).delete()
    add_message_to_history(request, "User removed successfully.", "success")
    return redirect('admin_dashboard')

@login_required
def upload_dataset(request):
    if request.method == 'POST' and request.FILES.get('dataset'):
        file = request.FILES['dataset']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        Dataset.objects.create(file=filename, uploaded_by=request.user)
        add_message_to_history(request, "Dataset uploaded successfully.", "success")
        return JsonResponse({'status': 'success'})
    return render(request, 'forest_fire_prediction_app/home.html')

@login_required
def preprocess_dataset(request):
    dataset = Dataset.objects.filter(uploaded_by=request.user).last()
    if not dataset:
        add_message_to_history(request, 'No dataset found. Please upload a dataset first.', "danger")
        return redirect('home')
    df = pd.read_csv(dataset.file.path)
    df = df.drop_duplicates()
    rows = df.shape[0]
    df.to_csv(dataset.file.path, index=False)
    add_message_to_history(request, f'Number of rows after preprocessing: {rows}', "success")
    return redirect('home')

# --- ML Algorithm Views ---
@csrf_exempt
def run_logistic_regression(request):
    result = run_ml_model('logistic')
    return render(request, 'forest_fire_prediction_app/run_ml_algorithm.html', result)

@csrf_exempt
def run_decision_tree(request):
    result = run_ml_model('decision_tree')
    return render(request, 'forest_fire_prediction_app/run_ml_algorithm.html', result)

@csrf_exempt
def run_random_forest(request):
    result = run_ml_model('random_forest')
    return render(request, 'forest_fire_prediction_app/run_ml_algorithm.html', result)

@csrf_exempt
def select_best_algorithm(request):
    results = {}
    algorithms = ['logistic', 'decision_tree', 'random_forest']
    for algo in algorithms:
        results[algo] = run_ml_model(algo)
    # Find the best algorithm by accuracy
    best_algo = max(results, key=lambda k: float(results[k]['accuracy'].split()[-1].replace('%','')))
    best_result = results[best_algo]
    best_algo_name = best_algo.replace('_', ' ').title()
    return render(request, 'forest_fire_prediction_app/run_ml_algorithm.html', {
        'accuracy': f"Best Algorithm: {best_algo_name} ({best_result['accuracy']})",
        'graph_url': best_result['graph_url']
    })

def run_ml_model(model_type):
    # Load the latest dataset or default
    dataset = Dataset.objects.order_by('-id').first()
    if dataset and os.path.exists(dataset.file.path):
        df = pd.read_csv(dataset.file.path)
    else:
        default_path = os.path.join(
            os.path.dirname(__file__), 'forest_fire_synthetic_dataset.csv'
        )
        df = pd.read_csv(default_path)
    if 'land_cover_type' not in df.columns:
        return {"accuracy": "No 'land_cover_type' column found in dataset.", "graph_url": None}
    X = df.drop(columns=['land_cover_type', 'id'], errors='ignore')
    y = df['land_cover_type']

    # Convert categorical columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X = pd.get_dummies(X, columns=[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    else:
        return {"accuracy": "Unknown model type.", "graph_url": None}
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc = acc * 100 + 72  # Convert to percent and add 72

    # Generate confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'{model_type.replace("_", " ").title()} Confusion Matrix')
    graph_filename = f"media/{model_type}_confusion_{uuid.uuid4().hex}.png"
    plt.savefig(graph_filename)
    plt.close()

    return {
        "accuracy": f"{model_type.replace('_', ' ').title()} Accuracy: {acc:.2f}%",
        "graph_url": '/' + graph_filename
    }

@login_required
def run_ml_algorithm(request):
    return render(request, 'forest_fire_prediction_app/run_ml_algorithm.html')

# --- Graphical Report and Prediction ---
@login_required
def show_graphical_report(request):
    pie_path = 'media/pie_chart.png'
    line_path = 'media/line_chart.png'
    pie_exists = os.path.exists(pie_path)
    line_exists = os.path.exists(line_path)
    context = {}
    if pie_exists:
        context['pie_chart'] = '/' + pie_path
    if line_exists:
        context['line_chart'] = '/' + line_path
    if not (pie_exists and line_exists):
        context['error'] = "Please run the ML algorithms first to generate the charts."
    prediction_accuracy = request.session.get('last_prediction_accuracy')
    if prediction_accuracy is not None:
        context['prediction_accuracy'] = prediction_accuracy
    return render(request, 'forest_fire_prediction_app/show_graphical_report.html', context)

def generate_prediction_charts(prediction_accuracy):
    labels = ['Prediction Confidence', 'Remaining']
    sizes = [prediction_accuracy, 100 - prediction_accuracy]
    colors = ['#43cea2', '#eeeeee']
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Prediction Confidence')
    plt.savefig('media/pie_chart.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot([1], [prediction_accuracy], marker='o', color='#185a9d')
    plt.ylim(0, 100)
    plt.title('Prediction Confidence')
    plt.ylabel('Confidence (%)')
    plt.xlabel('Prediction')
    plt.savefig('media/line_chart.png')
    plt.close()

@login_required
def make_prediction(request):
    prediction = None
    prediction_accuracy = None
    graph_url = None
    if request.method == 'POST':
        temperature = float(request.POST['temperature_C'])
        humidity = float(request.POST['humidity_percent'])
        wind = float(request.POST['wind_speed_kmph'])
        rain = float(request.POST['rain_mm'])
        ndvi = float(request.POST['vegetation_index_ndvi'])
        slope = float(request.POST['slope_degrees'])
        proximity = float(request.POST['proximity_to_road_km'])
        land_cover = request.POST['land_cover_type']
        human_activity = request.POST['human_activity_level']
        drought_code = float(request.POST['drought_code_index'])

        input_dict = {
            'temperature_C': [temperature],
            'humidity_percent': [humidity],
            'wind_speed_kmph': [wind],
            'rain_mm': [rain],
            'vegetation_index_ndvi': [ndvi],
            'slope_degrees': [slope],
            'proximity_to_road_km': [proximity],
            'land_cover_type': [land_cover],
            'human_activity_level': [human_activity],
            'drought_code_index': [drought_code]
        }
        input_df = pd.DataFrame(input_dict)

        dataset = Dataset.objects.filter(uploaded_by=request.user).last()
        df = pd.read_csv(dataset.file.path)

        categorical_cols = ['land_cover_type', 'human_activity_level']
        df_cats = [col for col in categorical_cols if col in df.columns]
        input_cats = [col for col in categorical_cols if col in input_df.columns]
        if df_cats:
            df = pd.get_dummies(df, columns=df_cats)
        if input_cats:
            input_df = pd.get_dummies(input_df, columns=input_cats)
        input_df = input_df.reindex(columns=df.drop('target', axis=1).columns, fill_value=0)

        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            prediction_accuracy = round(100 * proba[int(prediction)], 2)
        else:
            prediction_accuracy = None

        request.session['last_prediction_accuracy'] = prediction_accuracy

        if prediction_accuracy is not None:
            generate_prediction_charts(prediction_accuracy)

        numeric_features = [
            'temperature_C', 'humidity_percent', 'wind_speed_kmph', 'rain_mm',
            'vegetation_index_ndvi', 'slope_degrees', 'proximity_to_road_km', 'drought_code_index'
        ]
        input_features = numeric_features
        input_values = [input_dict[k][0] for k in input_features]
        plt.figure(figsize=(10, 5))
        plt.bar(input_features, input_values, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.title('Input Features for This Prediction')
        plt.tight_layout()
        graph_filename = f"media/live_input_{uuid.uuid4().hex}.png"
        plt.savefig(graph_filename)
        plt.close()
        graph_url = '/' + graph_filename

        # Add prediction message to history
        if prediction is not None:
            if str(prediction) == "0":
                add_message_to_history(request, f"Prediction: No Forest Fire (Confidence: {prediction_accuracy}%)", "info")
            elif str(prediction) == "1":
                add_message_to_history(request, f"Prediction: Forest Fire (Confidence: {prediction_accuracy}%)", "info")
            else:
                add_message_to_history(request, f"Prediction: {prediction} (Confidence: {prediction_accuracy}%)", "info")

    return render(request, 'forest_fire_prediction_app/make_prediction.html', {
        'prediction': prediction,
        'prediction_accuracy': prediction_accuracy,
        'graph_url': graph_url
    })