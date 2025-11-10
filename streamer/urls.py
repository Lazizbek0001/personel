from django.urls import path
from . import views

app_name = 'streamer'

urlpatterns = [
    # Main pages
    path('', views.CameraListView.as_view(), name='camera_list'),
    
    # Camera management
    path('camera/add/', views.CameraCreateView.as_view(), name='camera_add'),
    path('camera/<int:pk>/edit/', views.CameraUpdateView.as_view(), name='camera_edit'),
    path('camera/<int:pk>/delete/', views.CameraDeleteView.as_view(), name='camera_delete'),
    
    # Streaming API endpoints
    path('start/<int:camera_id>/', views.start_stream, name='start_stream'),
    path('stop/', views.stop_stream, name='stop_stream'),
    path('status/', views.stream_status, name='stream_status'),
    path('mjpeg/<int:camera_id>/', views.mjpeg_stream, name='mjpeg_stream'),
]