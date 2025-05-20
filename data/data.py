import kagglehub

# Download latest version
path = kagglehub.dataset_download("khushikyad001/impact-of-screen-time-on-mental-health")

print("Path to dataset files:", path)