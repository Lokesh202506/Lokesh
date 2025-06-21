from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('user_dashboard/', views.user_dashboard, name='user_dashboard'),
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-login/', views.admin_login, name='admin_login'),  
    path('admin-logout/', views.admin_logout, name='admin_logout'),
    path('remove_user/<int:user_id>/', views.remove_user, name='remove_user'),
    path('upload_dataset/', views.upload_dataset, name='upload_dataset'),
    path('preprocess_dataset/', views.preprocess_dataset, name='preprocess_dataset'),
    path('run_ml_algorithm/', views.run_ml_algorithm, name='run_ml_algorithm'),
    path('run_logistic_regression/', views.run_logistic_regression, name='run_logistic_regression'),
    path('run_decision_tree/', views.run_decision_tree, name='run_decision_tree'),
    path('run_random_forest/', views.run_random_forest, name='run_random_forest'),
    path('select_best_algorithm/', views.select_best_algorithm, name='select_best_algorithm'),
    path('show_graphical_report/', views.show_graphical_report, name='show_graphical_report'),
    path('make_prediction/', views.make_prediction, name='make_prediction'),
]