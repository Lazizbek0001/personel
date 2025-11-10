from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views


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
    
    # HLS streaming endpoints
    path('media/hls/camera_<int:camera_id>/<str:filename>', views.hls_playlist, name='hls_playlist'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)